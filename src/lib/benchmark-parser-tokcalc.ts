/**
 * Community benchmark submission template.
 *
 * Users download this JSON file, fill in their benchmark results,
 * and paste it into tokcalc's BenchmarkImport textarea.
 *
 * The tokcalc-standard parser recognizes this format and creates
 * a BenchmarkRecord with confidence_tier = "community_reproducible"
 * (if all required fields are filled) or "community_unverified" (if some are missing).
 *
 * This is the "tier 4" ingestion path from the Perplexity research:
 * "A tokcalc-standard JSON schema for community uploads."
 */

export const BENCHMARK_TEMPLATE = `{
  "format": "tokcalc-standard",
  "version": "1.0",

  "source": {
    "submitter": "your-github-username",
    "submit_date": "2026-09-24",
    "raw_artifact_url": "https://github.com/your-username/your-repo/blob/main/benchmark-results.json",
    "notes": "Describe how you ran this benchmark. Include the exact command you used."
  },

  "model": {
    "model_id": "meta-llama/Llama-3.3-70B-Instruct",
    "model_revision": "commit hash or release date",
    "quantization_format": "FP8",
    "weight_dtype": "float8_e4m3",
    "kv_cache_dtype": "float16"
  },

  "hardware": {
    "accelerator_vendor": "NVIDIA",
    "accelerator_model": "H100 SXM5 80GB",
    "gpu_count": 1,
    "memory_per_gpu_gb": 80,
    "interconnect": "NVLink",
    "cloud_provider": "RunPod",
    "instance_type": "H100 80GB SXM",
    "region": "us-east-1",
    "driver_version": "550.54.15",
    "cuda_version": "12.4"
  },

  "engine": {
    "serving_engine": "vLLM",
    "engine_version": "0.29.0",
    "tensor_parallel_size": 1,
    "max_model_len": 32768,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "prefix_cache_enabled": true,
    "chunked_prefill_enabled": true,
    "attention_backend": "FlashInfer",
    "command": "vllm serve meta-llama/Llama-3.3-70B-Instruct --quantization fp8 --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 256"
  },

  "workload": {
    "workload_type": "online_poisson",
    "dataset_name": "ShareGPT",
    "request_count": 1000,
    "concurrency": 64,
    "input_tokens_mean": 1024,
    "input_tokens_p95": 4096,
    "output_tokens_mean": 256,
    "output_tokens_p95": 1024,
    "streaming_enabled": true,
    "sampling_config": {
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 512
    }
  },

  "metrics": {
    "successful_requests": 1000,
    "failed_requests": 0,
    "output_token_throughput_tps": 2450.5,
    "total_token_throughput_tps": 11000.0,
    "request_throughput_rps": 3.91,
    "ttft_mean_ms": 112.3,
    "ttft_p50_ms": 108.5,
    "ttft_p95_ms": 145.2,
    "ttft_p99_ms": 189.7,
    "itl_mean_ms": 5.67,
    "itl_p50_ms": 5.45,
    "itl_p95_ms": 7.23,
    "itl_p99_ms": 8.91,
    "gpu_memory_peak_gb": 78.5,
    "gpu_utilization_mean_pct": 92.5
  }
}`;

/**
 * Parse a tokcalc-standard benchmark JSON.
 * This is the community submission format — users fill in the template
 * and paste it into the BenchmarkImport textarea.
 */
import type {
  BenchmarkParser,
  BenchmarkRecord,
  ConfidenceTier,
} from "./benchmark-schema";
import { generateBenchmarkId } from "./benchmark-schema";

export const tokcalcStandardParser: BenchmarkParser = {
  name: "tokcalc-standard",
  version: "v1",
  formatLabel: "tokcalc community benchmark",

  detect(rawText: string): boolean {
    try {
      const data = JSON.parse(rawText);
      return data?.format === "tokcalc-standard" || !!data?.metrics?.output_token_throughput_tps;
    } catch {
      return false;
    }
  },

  parse(rawText: string): BenchmarkRecord | null {
    try {
      const data = JSON.parse(rawText);
      if (!data) return null;

      const model = data.model || {};
      const hardware = data.hardware || {};
      const engine = data.engine || {};
      const workload = data.workload || {};
      const metrics = data.metrics || {};
      const source = data.source || {};

      const modelId = model.model_id || "unknown";
      const modelDisplayName = modelId.split("/").pop() || modelId;

      // Determine confidence tier based on completeness
      const hasFullConfig =
        modelId !== "unknown" &&
        hardware.accelerator_model &&
        engine.serving_engine &&
        metrics.output_token_throughput_tps > 0 &&
        workload.request_count > 0;
      const confidenceTier: ConfidenceTier = hasFullConfig
        ? "community_reproducible"
        : "community_unverified";

      return {
        benchmark_id: generateBenchmarkId(),
        source_name: "community",
        source_url: source.raw_artifact_url || "",
        source_license: "CC-BY-SA-4.0",
        ingestion_method: "uploaded_raw_artifact",
        retrieved_at: new Date().toISOString(),
        benchmark_started_at: source.submit_date,
        raw_artifact_url: source.raw_artifact_url,
        parser_name: "tokcalc-standard",
        parser_version: "v1",
        confidence_tier: confidenceTier,
        verification_status: "schema_valid",

        model_id: modelId,
        model_display_name: modelDisplayName,
        model_revision: model.model_revision || "unspecified",
        architecture: model.architecture,
        parameter_count_b: model.parameter_count_b,
        context_window_tokens: engine.max_model_len,
        quantization_format: model.quantization_format || "unspecified",
        weight_dtype: model.weight_dtype,
        kv_cache_dtype: model.kv_cache_dtype || "float16",

        accelerator_vendor: hardware.accelerator_vendor || "NVIDIA",
        accelerator_model: hardware.accelerator_model || "unspecified",
        gpu_count: hardware.gpu_count || engine.tensor_parallel_size || 1,
        memory_per_gpu_gb: hardware.memory_per_gpu_gb || 0,
        host_cpu: hardware.host_cpu,
        host_memory_gb: hardware.host_memory_gb,
        interconnect: hardware.interconnect,
        topology: hardware.topology,
        cloud_provider: hardware.cloud_provider,
        instance_type: hardware.instance_type,
        region: hardware.region,
        driver_version: hardware.driver_version,
        cuda_rocm_metal_version: hardware.cuda_version,

        serving_engine: engine.serving_engine || "unspecified",
        engine_version: engine.engine_version || "unspecified",
        engine_command: engine.command,
        engine_config: {
          tensor_parallel_size: engine.tensor_parallel_size,
          max_model_len: engine.max_model_len,
          max_num_seqs: engine.max_num_seqs,
          max_num_batched_tokens: engine.max_num_batched_tokens,
          prefix_cache_enabled: engine.prefix_cache_enabled,
          chunked_prefill_enabled: engine.chunked_prefill_enabled,
          attention_backend: engine.attention_backend,
        },

        workload_type: workload.workload_type || "online_poisson",
        dataset_name: workload.dataset_name,
        request_count: workload.request_count || 0,
        concurrency: workload.concurrency || 1,
        arrival_rate_rps: workload.arrival_rate_rps,
        input_tokens_mean: workload.input_tokens_mean || 0,
        input_tokens_p50: workload.input_tokens_p50,
        input_tokens_p95: workload.input_tokens_p95,
        input_tokens_p99: workload.input_tokens_p99,
        output_tokens_mean: workload.output_tokens_mean || 0,
        output_tokens_p50: workload.output_tokens_p50,
        output_tokens_p95: workload.output_tokens_p95,
        output_tokens_p99: workload.output_tokens_p99,
        shared_prefix_tokens_mean: workload.shared_prefix_tokens_mean,
        shared_prefix_hit_rate: workload.shared_prefix_hit_rate,
        streaming_enabled: workload.streaming_enabled ?? true,
        sampling_config: workload.sampling_config,

        successful_requests: metrics.successful_requests ?? workload.request_count ?? 0,
        failed_requests: metrics.failed_requests ?? 0,
        request_throughput_rps: metrics.request_throughput_rps,
        output_token_throughput_tps: metrics.output_token_throughput_tps || 0,
        total_token_throughput_tps: metrics.total_token_throughput_tps,
        ttft_mean_ms: metrics.ttft_mean_ms,
        ttft_p50_ms: metrics.ttft_p50,
        ttft_p95_ms: metrics.ttft_p95_ms,
        ttft_p99_ms: metrics.ttft_p99_ms,
        itl_mean_ms: metrics.itl_mean_ms,
        itl_p50_ms: metrics.itl_p50,
        itl_p95_ms: metrics.itl_p95_ms,
        itl_p99_ms: metrics.itl_p99_ms,
        e2e_latency_p50_ms: metrics.e2e_latency_p50_ms,
        e2e_latency_p95_ms: metrics.e2e_latency_p95_ms,
        e2e_latency_p99_ms: metrics.e2e_latency_p99_ms,
        gpu_memory_peak_gb: metrics.gpu_memory_peak_gb,
        gpu_utilization_mean_pct: metrics.gpu_utilization_mean_pct,
        power_mean_w: metrics.power_mean_w,
        energy_per_output_token_j: metrics.energy_per_output_token_j,

        metric_definition_version: "tokcalc-v1",
        citation_text: `Community benchmark: ${metrics.output_token_throughput_tps || 0} tok/s, ${workload.request_count || 0} requests, submitted by ${source.submitter || "anonymous"}`,
        attribution_required: true,
        commercial_redistribution_allowed: "yes",
      };
    } catch {
      return null;
    }
  },
};
