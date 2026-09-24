/**
 * vLLM benchmark JSON parser.
 *
 * vLLM's `benchmark_serving.py` produces JSON output with fields like:
 *   - output_throughput: output tokens/s
 *   - mean_ttft_ms: time to first token
 *   - p99_ttft_ms
 *   - mean_itl_ms: inter-token latency
 *   - mean_tpot_ms: time per output token
 *   - total_token_throughput
 *   - duration
 *   - completed / total_input_tokens / total_output_tokens
 *
 * Example vLLM benchmark output:
 * {
 *   "elapsed_time": 100.5,
 *   "num_input_tokens": 1024000,
 *   "num_output_tokens": 256000,
 *   "request_throughput": 1.73,
 *   "output_throughput": 382.89,
 *   "total_token_throughput": 619.85,
 *   "mean_ttft_ms": 71.54,
 *   "median_ttft_ms": 70.21,
 *   "p99_ttft_ms": 79.49,
 *   "mean_tpot_ms": 12.45,
 *   "median_tpot_ms": 12.30,
 *   "p99_tpot_ms": 14.87,
 *   "mean_itl_ms": 7.74,
 *   "median_itl_ms": 7.68,
 *   "p99_itl_ms": 8.92,
 *   "input_len": "1024",
 *   "output_len": "256",
 *   "num_requests": 1000,
 *   "model_id": "meta-llama/Llama-3.1-70B-Instruct",
 *   ...
 * }
 *
 * The actual field names vary by vLLM version. This parser handles the
 * common field aliases across v0.5.x → v0.29.x.
 */

import type {
  BenchmarkParser,
  BenchmarkRecord,
  ConfidenceTier,
} from "./benchmark-schema";
import { generateBenchmarkId } from "./benchmark-schema";

export const vllmParser: BenchmarkParser = {
  name: "vllm-json",
  version: "v1",
  formatLabel: "vLLM benchmark JSON",

  detect(rawText: string): boolean {
    try {
      const data = JSON.parse(rawText);
      // Check for vLLM-specific fields (any combination)
      return !!(
        data &&
        (data.output_throughput ||
          data.total_token_throughput ||
          data.mean_ttft_ms ||
          data.mean_itl_ms ||
          data.mean_tpot_ms ||
          data.num_requests ||
          data.request_throughput)
      );
    } catch {
      return false;
    }
  },

  parse(rawText: string): BenchmarkRecord | null {
    try {
      const data = JSON.parse(rawText);
      if (!data) return null;

      // Extract model info
      const modelId = data.model_id || data.model || "unknown";
      const modelDisplayName = modelId.split("/").pop() || modelId;

      // Extract throughput metrics (handle field name variations)
      const outputThroughput = data.output_throughput ?? data.output_token_throughput ?? 0;
      const totalThroughput = data.total_token_throughput ?? 0;
      const requestThroughput = data.request_throughput ?? 0;

      // Extract latency metrics
      const meanTtft = data.mean_ttft_ms ?? data.mean_ttft;
      const p50Ttft = data.median_ttft_ms ?? data.p50_ttft_ms ?? data.ttft_p50;
      const p95Ttft = data.p95_ttft_ms ?? data.ttft_p95;
      const p99Ttft = data.p99_ttft_ms ?? data.p99_ttft;

      const meanItl = data.mean_itl_ms ?? data.mean_itl;
      const p50Itl = data.median_itl_ms ?? data.p50_itl_ms ?? data.itl_p50;
      const p95Itl = data.p95_itl_ms ?? data.itl_p95;
      const p99Itl = data.p99_itl_ms ?? data.p99_itl;

      const meanTpot = data.mean_tpot_ms ?? data.mean_tpot;
      const p99Tpot = data.p99_tpot_ms ?? data.p99_tpot;

      // Extract workload info
      const numRequests = data.num_requests ?? data.completed ?? 0;
      const inputLen = Number(data.input_len ?? data.mean_input_len ?? data.input_tokens_mean ?? 0);
      const outputLen = Number(data.output_len ?? data.mean_output_len ?? data.output_tokens_mean ?? 0);
      const concurrency = data.num_procs ?? data.concurrency ?? 1;

      // Extract duration
      const elapsed = data.elapsed_time ?? data.duration ?? 0;

      // Determine confidence tier
      const hasFullConfig = !!data.model_id && !!data.output_throughput && numRequests > 0;
      const confidenceTier: ConfidenceTier = hasFullConfig
        ? "community_reproducible"
        : "community_unverified";

      const record: BenchmarkRecord = {
        benchmark_id: generateBenchmarkId(),
        source_name: "vllm",
        source_url: data.source_url || "https://docs.vllm.ai/en/latest/benchmarking/cli/",
        source_license: "Apache-2.0",
        ingestion_method: "uploaded_raw_artifact",
        retrieved_at: new Date().toISOString(),
        benchmark_started_at: data.timestamp || data.run_time,
        parser_name: "vllm-json",
        parser_version: "v1",
        confidence_tier: confidenceTier,
        verification_status: "schema_valid",

        model_id: modelId,
        model_display_name: modelDisplayName,
        model_revision: data.model_revision || data.commit || "unspecified",
        architecture: data.architecture || "dense",
        parameter_count_b: data.parameter_count_b,
        context_window_tokens: data.max_model_len,
        quantization_format: data.quantization || data.dtype || "unspecified",
        weight_dtype: data.dtype || data.weight_dtype,
        kv_cache_dtype: data.kv_cache_dtype || "float16",

        accelerator_vendor: data.gpu_vendor || "NVIDIA",
        accelerator_model: data.gpu_model || "unspecified",
        gpu_count: data.tensor_parallel_size || data.gpu_count || 1,
        memory_per_gpu_gb: data.gpu_memory || 0,
        driver_version: data.driver_version,
        cuda_rocm_metal_version: data.cuda_version,

        serving_engine: "vLLM",
        engine_version: data.vllm_version || "unspecified",
        engine_command: data.command,
        engine_config: {
          tensor_parallel_size: data.tensor_parallel_size,
          max_model_len: data.max_model_len,
          max_num_seqs: data.max_num_seqs,
          max_num_batched_tokens: data.max_num_batched_tokens,
          prefix_cache_enabled: data.enable_prefix_caching ?? data.prefix_cache_enabled,
          chunked_prefill_enabled: data.enable_chunked_prefill,
          attention_backend: data.attention_backend,
        },

        workload_type: data.workload_type || "online_poisson",
        dataset_name: data.dataset_name || data.dataset,
        request_count: numRequests,
        concurrency,
        input_tokens_mean: inputLen,
        output_tokens_mean: outputLen,
        streaming_enabled: data.streaming ?? true,

        successful_requests: numRequests,
        failed_requests: data.failed ?? (data.total_requests ? data.total_requests - numRequests : 0),
        request_throughput_rps: requestThroughput,
        output_token_throughput_tps: outputThroughput,
        total_token_throughput_tps: totalThroughput,
        ttft_mean_ms: meanTtft,
        ttft_p50_ms: p50Ttft,
        ttft_p95_ms: p95Ttft,
        ttft_p99_ms: p99Ttft,
        itl_mean_ms: meanItl,
        itl_p50_ms: p50Itl,
        itl_p95_ms: p95Itl,
        itl_p99_ms: p99Itl,
        tpot_mean_ms: meanTpot,
        e2e_latency_p99_ms: data.mean_lantecy_ms || data.e2e_latency_p99_ms,

        gpu_memory_peak_gb: data.peak_gpu_memory || data.gpu_memory_peak_gb,
        gpu_utilization_mean_pct: data.gpu_utilization,

        metric_definition_version: "vllm-v1",
        citation_text: `vLLM benchmark: ${outputThroughput.toFixed(2)} output tok/s, TTFT mean ${meanTtft?.toFixed(1) || "?"} ms, ${numRequests} requests`,
        attribution_required: true,
        commercial_redistribution_allowed: "yes",
      };

      return record;
    } catch (e) {
      console.error("vLLM parse error:", e);
      return null;
    }
  },
};
