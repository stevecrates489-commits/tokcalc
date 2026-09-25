/**
 * TensorRT-LLM benchmark parser.
 *
 * TRT-LLM's `benchmark_serving.py` produces JSON output with fields like:
 *   - throughput (tokens/s)
 *   - latency_avg (ms) — end-to-end latency
 *   - first_token_latency_avg (ms) — TTFT
 *   - inter_token_latency_avg (ms) — ITL
 *   - total_input_tokens, total_output_tokens
 *   - num_requests
 *   - model
 *
 * Refs: https://nvidia.github.io/TensorRT-LLM/1.3.0rc4/developer-guide/perf-benchmarking.html
 */

import type {
  BenchmarkParser,
  BenchmarkRecord,
  ConfidenceTier,
} from "./benchmark-schema";
import { generateBenchmarkId } from "./benchmark-schema";

export const trtllmParser: BenchmarkParser = {
  name: "trtllm-json",
  version: "v1",
  formatLabel: "TensorRT-LLM benchmark JSON",

  detect(rawText: string): boolean {
    try {
      const data = JSON.parse(rawText);
      return !!(
        data &&
        (data.throughput ||
          data.first_token_latency_avg ||
          data.inter_token_latency_avg ||
          data.latency_avg ||
          data.total_input_tokens ||
          data.total_output_tokens)
      );
    } catch {
      return false;
    }
  },

  parse(rawText: string): BenchmarkRecord | null {
    try {
      const data = JSON.parse(rawText);
      if (!data) return null;

      const modelId = data.model || data.model_id || data.model_name || "unknown";
      const modelDisplayName = modelId.split("/").pop() || modelId;

      const outputThroughput = data.throughput ?? data.output_throughput ?? 0;
      const meanTtft =
        data.first_token_latency_avg ?? data.ttft_mean ?? data.mean_ttft_ms;
      const meanItl =
        data.inter_token_latency_avg ?? data.itl_mean ?? data.mean_itl_ms;
      const meanE2E = data.latency_avg ?? data.e2e_latency_mean;

      const numRequests = data.num_requests ?? data.completed ?? 0;

      const totalInPerReq =
        data.total_input_tokens && numRequests > 0
          ? data.total_input_tokens / numRequests
          : undefined;
      const inputLen = Number(
        data.input_len ?? data.mean_input_len ?? totalInPerReq ?? 0,
      );

      const totalOutPerReq =
        data.total_output_tokens && numRequests > 0
          ? data.total_output_tokens / numRequests
          : undefined;
      const outputLen = Number(
        data.output_len ?? data.mean_output_len ?? totalOutPerReq ?? 0,
      );

      const concurrency = data.concurrency ?? data.num_procs ?? 1;

      const confidenceTier: ConfidenceTier =
        modelId !== "unknown" && outputThroughput > 0 && numRequests > 0
          ? "community_reproducible"
          : "community_unverified";

      return {
        benchmark_id: generateBenchmarkId(),
        source_name: "tensorrt_llm",
        source_url: data.source_url || "https://nvidia.github.io/TensorRT-LLM/",
        source_license: "Apache-2.0",
        ingestion_method: "uploaded_raw_artifact",
        retrieved_at: new Date().toISOString(),
        parser_name: "trtllm-json",
        parser_version: "v1",
        confidence_tier: confidenceTier,
        verification_status: "schema_valid",

        model_id: modelId,
        model_display_name: modelDisplayName,
        model_revision: data.model_revision || "unspecified",
        architecture: data.architecture,
        parameter_count_b: data.parameter_count_b,
        context_window_tokens: data.max_model_len,
        quantization_format: data.quantization || data.dtype || "unspecified",
        weight_dtype: data.dtype,
        kv_cache_dtype: data.kv_cache_dtype || "float16",

        accelerator_vendor: "NVIDIA",
        accelerator_model: data.gpu_model || "unspecified",
        gpu_count: data.tensor_parallel_size || data.gpu_count || 1,
        memory_per_gpu_gb: data.gpu_memory || 0,
        driver_version: data.driver_version,
        cuda_rocm_metal_version: data.cuda_version,

        serving_engine: "TensorRT-LLM",
        engine_version: data.trtllm_version || "unspecified",
        engine_command: data.command,
        engine_config: {
          tensor_parallel_size: data.tensor_parallel_size,
          max_model_len: data.max_model_len,
          max_num_seqs: data.max_num_seqs,
          max_num_batched_tokens: data.max_num_batched_tokens,
          prefix_cache_enabled: data.enable_prefix_caching,
          chunked_prefill_enabled: data.enable_chunked_prefill,
          attention_backend: "TensorRT-LLM",
        },

        workload_type: data.workload_type || "online_poisson",
        dataset_name: data.dataset_name || data.dataset,
        request_count: numRequests,
        concurrency,
        input_tokens_mean: inputLen,
        output_tokens_mean: outputLen,
        streaming_enabled: data.streaming ?? true,

        successful_requests: numRequests,
        failed_requests: data.failed ?? 0,
        output_token_throughput_tps: outputThroughput,
        total_token_throughput_tps: data.total_throughput,
        request_throughput_rps: data.request_throughput,
        ttft_mean_ms: meanTtft,
        ttft_p50_ms: data.first_token_latency_p50,
        ttft_p95_ms: data.first_token_latency_p95,
        ttft_p99_ms: data.first_token_latency_p99,
        itl_mean_ms: meanItl,
        itl_p50_ms: data.inter_token_latency_p50,
        itl_p95_ms: data.inter_token_latency_p95,
        itl_p99_ms: data.inter_token_latency_p99,
        e2e_latency_p50_ms: data.latency_p50,
        e2e_latency_p95_ms: data.latency_p95,
        e2e_latency_p99_ms: data.latency_p99,

        gpu_memory_peak_gb: data.peak_gpu_memory,

        metric_definition_version: "trtllm-v1",
        citation_text: `TRT-LLM benchmark: ${outputThroughput.toFixed(2)} tok/s, TTFT mean ${meanTtft?.toFixed(1) || "?"} ms, ${numRequests} requests`,
        attribution_required: true,
        commercial_redistribution_allowed: "yes",
      };
    } catch {
      return null;
    }
  },
};