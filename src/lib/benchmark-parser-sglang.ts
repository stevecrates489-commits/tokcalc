/**
 * SGLang benchmark parser.
 *
 * SGLang's benchmark scripts produce JSON/log output with fields like:
 *   - total_throughput or output_throughput (tokens/s)
 *   - mean_ttft or ttft_avg (ms)
 *   - mean_itl or itl_avg (ms)
 *   - total_input_tokens, total_output_tokens
 *   - num_requests
 *   - concurrency
 *   - model_id
 *
 * SGLang output format varies by benchmark script version. This parser
 * handles common aliases across SGLang v0.3.x → v0.5.x.
 */

import type {
  BenchmarkParser,
  BenchmarkRecord,
  ConfidenceTier,
} from "./benchmark-schema";
import { generateBenchmarkId } from "./benchmark-schema";

export const sglangParser: BenchmarkParser = {
  name: "sglang-json",
  version: "v1",
  formatLabel: "SGLang benchmark JSON",

  detect(rawText: string): boolean {
    try {
      const data = JSON.parse(rawText);
      return !!(
        data &&
        (data.total_throughput ||
          data.output_throughput ||
          data.mean_ttft ||
          data.ttft_avg ||
          data.mean_itl ||
          data.itl_avg ||
          data.num_requests ||
          data.total_input_tokens)
      );
    } catch {
      // Also check for SGLang log format (not JSON)
      return rawText.includes("sglang") && rawText.includes("throughput");
    }
  },

  parse(rawText: string): BenchmarkRecord | null {
    try {
      // Try JSON first
      const data = JSON.parse(rawText);
      if (!data) return null;

      const modelId = data.model_id || data.model || "unknown";
      const modelDisplayName = modelId.split("/").pop() || modelId;

      const outputThroughput =
        data.output_throughput ?? data.total_throughput ?? data.throughput ?? 0;
      const meanTtft = data.mean_ttft ?? data.ttft_avg ?? data.ttft_mean;
      const p99Ttft = data.p99_ttft ?? data.ttft_p99;
      const meanItl = data.mean_itl ?? data.itl_avg ?? data.itl_mean;
      const p99Itl = data.p99_itl ?? data.itl_p99;

      const numRequests = data.num_requests ?? data.completed ?? 0;
      const inputLen = Number(data.input_len ?? data.mean_input_len ?? 0);
      const outputLen = Number(data.output_len ?? data.mean_output_len ?? 0);
      const concurrency = data.concurrency ?? data.num_procs ?? 1;

      const confidenceTier: ConfidenceTier =
        modelId !== "unknown" && outputThroughput > 0 && numRequests > 0
          ? "community_reproducible"
          : "community_unverified";

      return {
        benchmark_id: generateBenchmarkId(),
        source_name: "sglang",
        source_url: data.source_url || "https://github.com/sgl-project/sglang",
        source_license: "Apache-2.0",
        ingestion_method: "uploaded_raw_artifact",
        retrieved_at: new Date().toISOString(),
        parser_name: "sglang-json",
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

        serving_engine: "SGLang",
        engine_version: data.sglang_version || "unspecified",
        engine_command: data.command,
        engine_config: {
          tensor_parallel_size: data.tensor_parallel_size,
          max_model_len: data.max_model_len,
          max_num_seqs: data.max_num_seqs,
          prefix_cache_enabled: data.enable_prefix_caching ?? data.radix_cache,
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
        failed_requests: data.failed ?? 0,
        output_token_throughput_tps: outputThroughput,
        total_token_throughput_tps: data.total_token_throughput,
        request_throughput_rps: data.request_throughput,
        ttft_mean_ms: meanTtft,
        ttft_p50_ms: data.p50_ttft ?? data.median_ttft,
        ttft_p95_ms: data.p95_ttft,
        ttft_p99_ms: p99Ttft,
        itl_mean_ms: meanItl,
        itl_p50_ms: data.p50_itl ?? data.median_itl,
        itl_p95_ms: data.p95_itl,
        itl_p99_ms: p99Itl,

        gpu_memory_peak_gb: data.peak_gpu_memory,

        metric_definition_version: "sglang-v1",
        citation_text: `SGLang benchmark: ${outputThroughput.toFixed(2)} tok/s, TTFT mean ${meanTtft?.toFixed(1) || "?"} ms, ${numRequests} requests`,
        attribution_required: true,
        commercial_redistribution_allowed: "yes",
      };
    } catch {
      // Not JSON — try log format (future: parse SGLang text logs)
      return null;
    }
  },
};
