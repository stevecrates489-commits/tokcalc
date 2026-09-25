/**
 * Benchmark provenance schema for tokcalc.
 *
 * Based on the Perplexity research brief which specified:
 * "Do not build a single scalar called 'tokens/sec.' A valid observed record
 * should always preserve the workload, serving software, engine flags, hardware
 * topology, model revision, quantization, and metric definition."
 *
 * This file defines:
 *   1. The BenchmarkRecord type (50+ fields across 7 categories)
 *   2. A confidence tier system (official → community → unverified)
 *   3. Parser interface for ingesting raw benchmark artifacts
 *
 * Design principle: tokcalc retains separate estimate types:
 *   - Theoretical model (what we have now — physics formulas)
 *   - Observed benchmark (this file — real measured data)
 *   - Calibrated estimate (theory adjusted by observed data — future)
 *   - User-provided run (user uploads their benchmark — future)
 */

/* ============================================================
   CONFIDENCE TIERS
   ============================================================ */

export type ConfidenceTier =
  | "official_audited"      // MLPerf submission, official vendor benchmark
  | "official_reproducible" // vLLM/SGLang/TRT-LLM docs example with full config
  | "community_reproducible" // User ran it with documented config + raw artifact
  | "community_unverified"   // User-reported number without raw artifact
  | "derived";               // Computed from other records (e.g., calibration)

export type VerificationStatus =
  | "unreviewed"
  | "schema_valid"
  | "reproduced"
  | "curated"
  | "rejected";

/* ============================================================
   CORE BENCHMARK RECORD
   ============================================================ */

export interface BenchmarkRecord {
  /* ---- Core identity and provenance ---- */
  benchmark_id: string;
  source_name: "vllm" | "sglang" | "tensorrt_llm" | "mlperf" | "llmperf" | "community" | "manual_curated";
  source_url: string;
  source_license: string;
  ingestion_method: "official_api" | "official_repo" | "uploaded_raw_artifact" | "manual_curated";
  retrieved_at: string;           // ISO 8601
  benchmark_started_at?: string;  // ISO 8601
  raw_artifact_url?: string;
  artifact_sha256?: string;
  parser_name?: string;
  parser_version?: string;
  confidence_tier: ConfidenceTier;
  verification_status: VerificationStatus;

  /* ---- Model identity ---- */
  model_id: string;
  model_display_name: string;
  model_repository?: string;
  model_revision: string;         // commit hash / version / date
  architecture?: "dense" | "moe" | "vlm" | "encoder" | "decoder_only";
  parameter_count_b?: number;
  context_window_tokens?: number;
  quantization_format: string;   // FP16, FP8, GGUF Q4_K_M, AWQ, etc.
  weight_dtype?: string;          // float16, bfloat16, float8_e4m3, int8, etc.
  kv_cache_dtype?: string;       // "float16" by default

  /* ---- Hardware and topology ---- */
  accelerator_vendor: "NVIDIA" | "AMD" | "Intel" | "Apple" | "Google" | "Groq" | "Cerebras" | string;
  accelerator_model: string;
  gpu_count: number;
  memory_per_gpu_gb: number;
  host_cpu?: string;
  host_memory_gb?: number;
  interconnect?: string;
  topology?: string;
  cloud_provider?: string;
  instance_type?: string;
  region?: string;
  driver_version?: string;
  cuda_rocm_metal_version?: string;

  /* ---- Serving stack and configuration ---- */
  serving_engine: "vLLM" | "SGLang" | "TensorRT-LLM" | "TGI" | "llama.cpp" | "MLX" | string;
  engine_version: string;
  engine_command?: string;
  engine_config?: {
    tensor_parallel_size?: number;
    pipeline_parallel_size?: number;
    max_model_len?: number;
    max_num_seqs?: number;
    max_num_batched_tokens?: number;
    prefix_cache_enabled?: boolean;
    speculative_decoding?: {
      enabled: boolean;
      draft_model?: string;
      acceptance_rate?: number;
    };
    chunked_prefill_enabled?: boolean;
    attention_backend?: string;
  };

  /* ---- Workload definition ---- */
  workload_type: "offline" | "online_poisson" | "fixed_rate" | "closed_loop" | "trace_replay" | "synthetic";
  dataset_name?: string;
  request_count: number;
  concurrency: number;
  arrival_rate_rps?: number;
  input_tokens_mean: number;
  input_tokens_p50?: number;
  input_tokens_p95?: number;
  input_tokens_p99?: number;
  output_tokens_mean: number;
  output_tokens_p50?: number;
  output_tokens_p95?: number;
  output_tokens_p99?: number;
  shared_prefix_tokens_mean?: number;
  shared_prefix_hit_rate?: number;
  streaming_enabled: boolean;
  sampling_config?: {
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    seed?: number;
  };

  /* ---- Observed metrics ---- */
  successful_requests: number;
  failed_requests: number;
  request_throughput_rps?: number;
  output_token_throughput_tps: number;       // THE key metric
  total_token_throughput_tps?: number;
  ttft_mean_ms?: number;
  ttft_p50_ms?: number;
  ttft_p95_ms?: number;
  ttft_p99_ms?: number;
  itl_mean_ms?: number;
  itl_p50_ms?: number;
  itl_p95_ms?: number;
  itl_p99_ms?: number;
  tpot_mean_ms?: number;                     // Time per output token (vLLM's term)
  e2e_latency_p50_ms?: number;
  e2e_latency_p95_ms?: number;
  e2e_latency_p99_ms?: number;
  gpu_memory_peak_gb?: number;
  gpu_utilization_mean_pct?: number;
  power_mean_w?: number;
  energy_per_output_token_j?: number;

  /* ---- Comparability and display ---- */
  metric_definition_version: string;
  slo_ttft_ms?: number;
  slo_itl_ms?: number;
  slo_e2e_ms?: number;
  comparability_group?: string;
  normalization_notes?: string;
  known_limitations?: string;
  citation_text: string;
  attribution_required: boolean;
  commercial_redistribution_allowed: "yes" | "no" | "unknown" | "contract_required";
}

/* ============================================================
   CALIBRATION RESULT — compares theoretical vs observed
   ============================================================ */

export interface CalibrationResult {
  /** What tokcalc's formulas predicted */
  estimated: {
    decodeTokensPerSec: number;
    aggregateTokensPerSec: number;
    prefillTokensPerSec: number;
    ttftMs: number;
    itlMs: number;
    modelSizeGb: number;
  };
  /** What the benchmark actually measured */
  observed: {
    outputTokenThroughputTps?: number;
    ttftMeanMs?: number;
    ttftP95Ms?: number;
    itlMeanMs?: number;
    itlP95Ms?: number;
    gpuMemoryPeakGb?: number;
  };
  /** Calibration error (observed - estimated) / estimated × 100 */
  calibrationErrors: {
    throughputErrorPct?: number;   // positive = observed was faster than predicted
    ttftErrorPct?: number;          // positive = observed TTFT was longer than predicted
    itlErrorPct?: number;
    memoryErrorPct?: number;
  };
  /** Does the observed data validate or contradict our model? */
  verdict: "validated" | "underestimated" | "overestimated" | "insufficient_data";
}

/* ============================================================
   PARSER INTERFACE
   ============================================================ */

export interface BenchmarkParser {
  name: string;
  version: string;
  /** Detect if raw text looks like this engine's output */
  detect(rawText: string): boolean;
  /** Parse raw text into a BenchmarkRecord */
  parse(rawText: string): BenchmarkRecord | null;
  /** Human-readable format name */
  formatLabel: string;
}

/* ============================================================
   HELPER: generate UUID (no dependency)
   ============================================================ */

export function generateBenchmarkId(): string {
  // RFC 4122 v4 UUID — no dependency needed
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/* ============================================================
   HELPER: format calibration verdict for display
   ============================================================ */

export function formatCalibrationVerdict(verdict: CalibrationResult["verdict"]): {
  label: string;
  color: string;
  description: string;
} {
  switch (verdict) {
    case "validated":
      return {
        label: "Validated",
        color: "text-emerald-500",
        description: "Observed benchmark matches theoretical estimate within ±20%. Formula is accurate for this config.",
      };
    case "underestimated":
      return {
        label: "Underestimated",
        color: "text-amber-500",
        description: "Real performance was BETTER than tokcalc predicted. Formula may be too conservative.",
      };
    case "overestimated":
      return {
        label: "Overestimated",
        color: "text-red-500",
        description: "Real performance was WORSE than tokcalc predicted. Formula may be too optimistic.",
      };
    case "insufficient_data":
      return {
        label: "Insufficient data",
        color: "text-muted-foreground",
        description: "Not enough observed metrics to calibrate against the theoretical estimate.",
      };
  }
}