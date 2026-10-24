/**
 * LLM Token/sec Calculator — Core Engine
 * ---------------------------------------
 * All formulas live here so they can be reused & tested independently.
 *
 * Theory reference:
 *   - Decode phase is MEMORY-BANDWIDTH bound (load weights once per token)
 *   - Prefill phase is COMPUTE bound (process prompt in parallel)
 *   - Real-world efficiency factor (~0.65) accounts for kernel overhead, KV cache reads, attention, sampling
 */

/* ============================================================
   CATALOG
   ------------------------------------------------------------
   The model / accelerator / quantization data no longer lives in this
   file. It lives in catalog/*.json (the editable source of truth, reviewed
   in Git) and is loaded through src/catalog/loader.ts — the ONE loader used
   by the website, the MCP server and the CLI.

   The types are re-exported below so every existing import path keeps
   working. There is exactly ONE copy of the catalog data in the repo.
   ============================================================ */
import {
  MODELS,
  MODEL_MAP,
  ACCELERATORS,
  ACCELERATOR_MAP,
  QUANTIZATIONS,
  QUANT_MAP,
} from "../catalog/loader";

import type {
  AcceleratorRecord,
  GpuCategory,
  GpuSpec,
  ModelCategory,
  ModelSpec,
  Quantization,
  QuantMeta,
  QuantizationRecord,
} from "../catalog/types";

export { MODELS, MODEL_MAP, QUANTIZATIONS, QUANT_MAP };
export { ACCELERATORS as GPUS, ACCELERATOR_MAP as GPU_MAP };
export type {
  AcceleratorRecord,
  GpuCategory,
  GpuSpec,
  ModelCategory,
  ModelSpec,
  Quantization,
  QuantMeta,
  QuantizationRecord,
};

/** ---------- EFFICIENCY CONSTANTS ---------- */
export const ETA_MEM = 0.65;       // Memory bandwidth utilization (real-world)
export const ETA_COMPUTE = 0.50;   // Compute utilization
export const ETA_NVLINK = 0.85;    // NVLink tensor-parallel communication efficiency
export const KERNEL_OVERHEAD_MS = 30; // per-request overhead

/**
 * How many tokens a streamed answer needs before a human can act on it.
 *
 * TTFT reports when the first *character* appears. A user cannot read "#" —
 * they need a clause. We model "readable" as roughly one short clause, and
 * state the assumption rather than hiding it inside a formula.
 */
export const FIRST_ANSWER_TOKENS = 12;

/**
 * Confidence levels surfaced in the UI to help users trust (or question)
 * specific numbers. The research brief called this "honest uncertainty" —
 * a key credibility signal for technical audiences (HN/Reddit/ML engineers).
 *
 * The older four-value scheme lumped three genuinely different kinds of
 * evidence under "measured" (a vendor spec sheet, a published price, and a
 * real observed run) and called a rule of thumb "inferred" — so a reader
 * could not tell *where* a number came from. The split below is by
 * provenance, not by how good the number feels:
 *
 *  - "official_spec"    → taken from a vendor spec sheet, model card, or
 *                         HuggingFace config.json / safetensors index.
 *  - "published_price"  → a published price (cloud GPU $/hr, API token price).
 *  - "observed_run"     → an actual measurement: a benchmark run, or a
 *                         reading recorded through `record_measured`.
 *  - "modeled"          → derived by a physics-based formula with cited
 *                         sources (e.g. decode = HBM_BW × η_mem / model_size).
 *  - "heuristic"        → derived by a rule of thumb with known error bars
 *                         (e.g. continuous-batching multiplier 1.5×).
 *  - "user_supplied"    → the user typed it in (e.g. GPU $/hr override).
 */
export type Confidence =
  | "official_spec"
  | "published_price"
  | "observed_run"
  | "modeled"
  | "heuristic"
  | "user_supplied";

/**
 * Map of metric name → confidence level.
 * Surfaced in the UI as a colored dot + tooltip next to each metric.
 */
export type ConfidenceMap = Record<string, Confidence>;

/** Continuous-batching multiplier guidance:
 *  - Research warns NO universal multiplier exists
 *  - vLLM reported 14–24x vs HF Transformers (extreme), 2.2–2.5x vs TGI
 *  - SOSP paper: 2–4x throughput vs FasterTransformer/Orca at same latency
 *  - Conservative default: 1.5x (clearly cited as "benchmark-derived range, not universal")
 *  Refs: https://arxiv.org/abs/2309.06180 (vLLM/PagedAttention paper)
 */
export const DEFAULT_BATCHING_MULTIPLIER = 1.5;

/** Anthropic prompt caching multipliers (verified 2025):
 *  - 5-minute cache write: 1.25x base input
 *  - 1-hour cache write: 2.0x base input
 *  - cache read: 0.1x base input (90% savings)
 *  - OpenAI: 50% off cached input (no separate write fee)
 *  Refs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
 */
export const ANTHROPIC_CACHE_WRITE_5M = 1.25;
export const ANTHROPIC_CACHE_WRITE_1H = 2.0;
export const ANTHROPIC_CACHE_READ = 0.1;
export const OPENAI_CACHE_DISCOUNT = 0.5;

export type CacheTTL = "none" | "5m" | "1h";

export interface CalcInput {
  modelId: string;
  gpuId: string;
  quantization: Quantization;
  numGpus: number;            // tensor parallel degree
  batchSize: number;          // concurrent requests
  promptTokens: number;        // input length (excluding cache prefix)
  outputTokens: number;        // requested output (visible answer)
  gpuHourlyCost?: number;     // optional override
  useSpeculative?: boolean;   // speculative decoding?
  speculativeBoost?: number;  // multiplier e.g. 2.0
  // === Phase 2 additions ===
  useContinuousBatching?: boolean;            // continuous batching toggle
  continuousBatchingMultiplier?: number;     // user-tunable 1.0–4.0x
  reasoningTokens?: number;                   // hidden reasoning budget (o1/R1/thinking)
  // Prompt caching (self-hosted or API)
  cachePrefixTokens?: number;                 // reusable prefix length (e.g. system prompt + RAG context)
  cacheHitRate?: number;                       // 0–1 fraction of requests that hit the cache
  cacheTTL?: CacheTTL;                        // "none" | "5m" | "1h"
  cacheProvider?: "self-hosted" | "anthropic" | "openai";
  // === Engine-aware presets ===
  engineId?: string;          // "vllm" | "sglang" | "trtllm" | "llamacpp" | "generic"
                               // When set, UI passes engine-specific effMem/effCompute
  effMem?: number;             // Engine-specific memory bandwidth utilization (overrides ETA_MEM)
  effCompute?: number;         // Engine-specific compute utilization (overrides ETA_COMPUTE)
}

export interface CalcResult {
  // ---- Raw measurements ----
  /** ALL params that must be resident in VRAM (paramsB × bytes). For MoE this
   *  is the full expert set, not just the per-token active subset. */
  modelSizeGb: number;
  /** ACTIVE params per token (activeParamsB × bytes). Equals modelSizeGb for
   *  dense models; ~10x smaller for MoE. This is the working set the decode
   *  memory-bandwidth math divides by. */
  activeParamsGb: number;
  /** True when the model is Mixture-of-Experts, i.e. modelSizeGb >> activeParamsGb. */
  isMoE: boolean;
  kvCachePerTokenKb: number;
  kvCacheTotalGb: number;
  /**
   * Fixed recurrent state (linear/KDA layers) in GB, for the whole batch.
   * Zero for ordinary full-attention models. This does NOT grow with context.
   */
  stateBytesGb: number;
  totalVramNeededGb: number;
  vramFits: boolean;

  // ---- Throughput ----
  decodeTokensPerSec: number;      // per-stream (batch=1)
  aggregateTokensPerSec: number;   // FLEET total — batched × continuous batching multiplier × numGpus (tensor-parallel)
  prefillTokensPerSec: number;     // prefill throughput
  batchCrossover: number;          // batch size where compute becomes bound

  // ---- Latency (new: split into TTFT + ITL) ----
  prefillTimeMs: number;           // time to read the prompt (prefill phase)
  decodeTimePerTokenMs: number;    // = ITL (inter-token latency)
  totalDecodeTimeMs: number;
  totalLatencyMs: number;
  endToEndTokensPerSec: number;    // output / total time
  ttftMs: number;                  // = prefill + first decode step + overhead
  itlMs: number;                   // alias for decodeTimePerTokenMs — for clarity
  /**
   * Time until the answer is READABLE, not merely started.
   * = prefill + FIRST_ANSWER_TOKENS × ITL + overhead. Usually several times
   * ttftMs — this is the number an SLO should be written against.
   */
  firstAnswerLatencyMs: number;

  // ---- Reasoning ----
  billedOutputTokens: number;     // visible + hidden reasoning tokens

  // ---- Prompt caching ----
  cacheHitRate: number;
  cachePrefixTokens: number;
  cacheTTL: CacheTTL;
  cacheProvider: string;
  prefillTokensAvoided: number;   // how many prefill tokens saved per cache hit
  cacheWriteCostMultiplier: number;  // 1.25 (5m) or 2.0 (1h) for Anthropic; 1.0 for OpenAI
  cacheReadCostMultiplier: number;   // 0.1 (Anthropic) or 0.5 (OpenAI) or 0 (self-hosted)
  effectivePrefillTokens: number;  // after cache hit adjustment
  prefillTimeWithCacheMs: number;  // actual prefill time accounting for cache
  cacheSavingsPct: number;         // % reduction in prefill time from caching

  // ---- Cost ----
  costPerHour: number;
  costPerMillionOutputTokens: number;
  costPerRequest: number;
  costPerRequestWithCache: number; // including cache write/read cost

  // ---- Components for display ----
  effectiveBandwidthGbps: number;
  effectiveFlopsTflops: number;
  quantEfficiency: number;
  speculativeBoost: number;
  multiGpuEfficiency: number;
  continuousBatchingMultiplier: number;  // actual multiplier applied
  longContextWarning?: string;          // set when prompt > 32K
  // Per-metric confidence map — surfaced as colored dots in the UI
  // (research: "honest uncertainty" = credibility signal for HN/Reddit/ML engineers)
  confidence: ConfidenceMap;
}

/**
 * Main calculation function.
 *
 * Math basis:
 *   decode_tokens_per_sec  ≈  (HBM_BW × η_mem × quant_eff) / model_size   (batch=1, mem-bound)
 *   prefill_tokens_per_sec ≈  (FLOPS × η_compute) / (2 × active_params)
 *   batch_crossover        ≈  (model_size × FLOPS × η_compute) / (2 × HBM_BW × η_mem × quant_eff)
 *                            = bytes_per_param × FLOPS × η_compute / (2 × HBM_BW × η_mem × quant_eff)
 *   total_latency          = prefill_time + output × decode_time + overhead
 *   cost_per_M_tokens     = (gpu_hourly_cost / 3600) / (decode_tokens_per_sec × batch) × 1e6
 */
export function calculate(input: CalcInput): CalcResult {
  const model = MODEL_MAP[input.modelId];
  const gpu = ACCELERATOR_MAP[input.gpuId];
  const quant = QUANT_MAP[input.quantization];

  if (!model) throw new Error(`Unknown model: ${input.modelId}`);
  if (!gpu) throw new Error(`Unknown GPU: ${input.gpuId}`);

  // ---- Model size ----
  // Two distinct quantities, easy to conflate:
  //
  //   activeParamsGb — ACTIVE params per token. For MoE only the routers pick
  //     a subset of experts per token, so this is the working set the memory
  //     bandwidth is divided by in the decode math below.
  //
  //   residentWeightsGb — ALL params that must sit in VRAM. MoE still needs
  //     every expert resident even though only some are active per token, so
  //     this is what actually consumes VRAM and what `modelSizeGb` reports.
  //
  // These are equal for dense models and diverge ~10x for MoE. Reporting the
  // active figure as "model weights" made any consumer that divides by it
  // (replica counts, headroom math) overestimate capacity by that factor.
  const activeParamsGb = model.activeParamsB * quant.bytesPerParam;
  const residentWeightsGb = model.paramsB * quant.bytesPerParam;

  // ---- Effective bandwidth / compute ----
  const quantEff = quant.efficiency;
  const tp = Math.max(1, input.numGpus);
  const multiGpuEfficiency = tp > 1 ? ETA_NVLINK : 1.0;
  const speculativeBoost = input.useSpeculative ? (input.speculativeBoost ?? 2.0) : 1.0;
  const continuousBatchingMultiplier = input.useContinuousBatching
    ? (input.continuousBatchingMultiplier ?? DEFAULT_BATCHING_MULTIPLIER)
    : 1.0;

  // Engine-aware efficiency factors: if effMem/effCompute are provided in input
  // (set by the UI based on selected engine), use those instead of global defaults.
  // This makes tokcalc engine-aware — vLLM, SGLang, TRT-LLM, llama.cpp all have
  // different real-world utilization due to kernel optimizations and scheduler design.
  const effMem = input.effMem ?? ETA_MEM;
  const effCompute = input.effCompute ?? ETA_COMPUTE;

  // Aggregate bandwidth across TP GPUs (uses engine-specific effMem if provided)
  const effectiveBandwidthGbps = gpu.memBandwidthGbps * tp * effMem * multiGpuEfficiency;
  // Some accelerators (B200/B300) don't publish dense FP16 — fall back to ~2.5x FP8 as conservative estimate.
  const gpuFlops = gpu.flopsTflops ?? 1500;  // conservative fallback if null
  const effectiveFlopsTflops = gpuFlops * tp * effCompute;

  // ---- KV cache / recurrent state ----
  // For a plain full-attention model: 2 (K&V) × layers × kvHeads × headDim × 2 bytes.
  // For a model with a declared attention layout, each layer is costed by its
  // own kind (sliding windows saturate, linear/KDA layers hold a fixed
  // recurrent state instead of a growing cache, MLA stores a compressed
  // latent). See computeKVBytesPerToken / computeStateBytes.
  const totalContextTokens = input.promptTokens + (input.cachePrefixTokens ?? 0);
  const kvBytesPerToken = computeKVBytesPerToken(model, 2, totalContextTokens);
  const kvCachePerTokenKb = kvBytesPerToken / 1024;
  const stateBytes = computeStateBytes(model);
  // KV cache grows with total context × batch; recurrent state is per request.
  const kvCacheTotalGb =
    (kvBytesPerToken * totalContextTokens * input.batchSize +
      stateBytes * input.batchSize) /
    1e9;

  // ---- Total VRAM ----
  const totalVramNeededGb = residentWeightsGb + kvCacheTotalGb;
  const vramFits = totalVramNeededGb <= gpu.vramGb * tp;

  // ---- Throughput ----
  // Decode is memory-bandwidth bound on the ACTIVE working set (MoE only reads
  // the experts its router selected for this token), so this divides by
  // activeParamsGb, not residentWeightsGb.
  const decodeTokensPerSecRaw = effectiveBandwidthGbps / activeParamsGb * quantEff * speculativeBoost;
  const decodeTokensPerSec = vramFits ? decodeTokensPerSecRaw : 0;

  // Prefill / compute-bound ceiling
  const prefillTokensPerSec = (effectiveFlopsTflops * 1e12) / (2 * model.activeParamsB * 1e9);
  const computeCeiling = prefillTokensPerSec; // same ceiling applies

  // Batch crossover (unit-corrected with ×1000 for TFLOPS/GB-s ratio)
  const batchCrossover =
    (quant.bytesPerParam * gpuFlops * 1000 * ETA_COMPUTE) /
    (2 * gpu.memBandwidthGbps * ETA_MEM * quantEff * (tp > 1 ? multiGpuEfficiency : 1));

  // Aggregate tokens/sec — per-GPU with continuous batching, then scaled to fleet total.
  // For tp>1, the fleet's combined throughput is the per-GPU aggregate × tp (tensor-parallel
  // scales near-linearly for decode-bound workloads). This keeps cost/M output consistent:
  // cost scales with tp, throughput scales with tp, cost/throughput stays correct.
  const memoryBoundAggregate = decodeTokensPerSec * input.batchSize * continuousBatchingMultiplier;
  const aggregateTokensPerSec = vramFits
    ? Math.min(memoryBoundAggregate, computeCeiling) * tp
    : 0;

  // ---- Prompt caching ----
  const cachePrefixTokens = input.cachePrefixTokens ?? 0;
  const cacheHitRate = input.cacheHitRate ?? 0;
  const cacheTTL = input.cacheTTL ?? "none";
  const cacheProvider = input.cacheProvider ?? "self-hosted";

  // Compute cache write/read multipliers based on provider
  let cacheWriteCostMultiplier = 1.0;
  let cacheReadCostMultiplier = 0.0; // self-hosted: cache reads are "free" (just memory access)
  if (cacheProvider === "anthropic") {
    cacheWriteCostMultiplier = cacheTTL === "1h" ? ANTHROPIC_CACHE_WRITE_1H : ANTHROPIC_CACHE_WRITE_5M;
    cacheReadCostMultiplier = ANTHROPIC_CACHE_READ;
  } else if (cacheProvider === "openai") {
    cacheWriteCostMultiplier = 1.0; // OpenAI: no separate write fee
    cacheReadCostMultiplier = OPENAI_CACHE_DISCOUNT;
  }

  // Effective prefill tokens after cache hit (fraction of prefix avoided per request, on average)
  // For self-hosted: full prefix is avoided on hit. For API: cost is reduced, not avoided.
  const prefillTokensAvoided = Math.round(cachePrefixTokens * cacheHitRate);
  const effectivePrefillTokens = Math.max(0, totalContextTokens - prefillTokensAvoided);
  const prefillTimeWithCacheMs = effectivePrefillTokens > 0
    ? (effectivePrefillTokens / prefillTokensPerSec) * 1000
    : 0;
  const cacheSavingsPct = totalContextTokens > 0
    ? (prefillTokensAvoided / totalContextTokens) * 100
    : 0;

  // ---- Latency ----
  // Use cache-adjusted prefill time if caching is enabled
  const prefillTimeMs = input.promptTokens > 0
    ? (input.useContinuousBatching || cachePrefixTokens > 0
        ? prefillTimeWithCacheMs
        : (input.promptTokens / prefillTokensPerSec) * 1000)
    : 0;
  const decodeTimePerTokenMs = decodeTokensPerSec > 0 ? (1000 / decodeTokensPerSec) : Infinity;

  // TTFT is NOT the same as prefill time. The first token only exists after
  // the prompt has been read AND one decode step has run, plus the per-request
  // scheduling overhead. Reporting prefill alone understates TTFT by
  // (itl + overhead) — which is the gap that makes a "sub-100 ms TTFT" claim
  // look better than the user will actually experience.
  const ttftMs = prefillTimeMs + decodeTimePerTokenMs + KERNEL_OVERHEAD_MS;

  // ---- First-answer latency ----
  // TTFT says when the first *character* appears; the user needs a clause
  // before they can act. Modelled explicitly so an SLO written against
  // "first answer" is not silently satisfied by a fast first token.
  const firstAnswerLatencyMs =
    prefillTimeMs + FIRST_ANSWER_TOKENS * decodeTimePerTokenMs + KERNEL_OVERHEAD_MS;

  // ---- Reasoning tokens (billed as output but invisible to user) ----
  const reasoningTokens = input.reasoningTokens ?? 0;
  const billedOutputTokens = input.outputTokens + reasoningTokens;
  const totalDecodeTimeMs = billedOutputTokens * decodeTimePerTokenMs;
  const totalLatencyMs = prefillTimeMs + totalDecodeTimeMs + KERNEL_OVERHEAD_MS;
  const endToEndTokensPerSec =
    totalLatencyMs > 0 ? (billedOutputTokens / totalLatencyMs) * 1000 : 0;

  // ---- Cost ----
  const costPerHour = input.gpuHourlyCost !== undefined ? input.gpuHourlyCost : (gpu.usdPerHour ?? 0);
  const costPerSecond = costPerHour / 3600;
  const costPerRequest =
    (totalLatencyMs / 1000) * costPerSecond * tp;
  const costPerMillionOutputTokens =
    aggregateTokensPerSec > 0 ? (costPerSecond * tp * 1e6) / aggregateTokensPerSec : Infinity;

  // Cost with cache — for self-hosted, same as costPerRequest (latency-driven)
  // For API providers, cache economics would be computed separately in the Build-vs-Buy tab
  const costPerRequestWithCache = costPerRequest;

  // ---- Long-context warning ----
  let longContextWarning: string | undefined;
  if (totalContextTokens > 32768) {
    longContextWarning = `Context ${totalContextTokens.toLocaleString()} tokens > 32K. ` +
      `Attention cost grows superlinearly with context for dense models; ` +
      `real-world throughput may be 20–40% lower than this estimate.`;
  }

  return {
    modelSizeGb: residentWeightsGb,
    activeParamsGb,
    isMoE: model.isMoE,
    kvCachePerTokenKb,
    kvCacheTotalGb,
    stateBytesGb: (stateBytes * input.batchSize) / 1e9,
    totalVramNeededGb,
    vramFits,
    decodeTokensPerSec,
    aggregateTokensPerSec,
    prefillTokensPerSec,
    batchCrossover,
    prefillTimeMs,
    decodeTimePerTokenMs,
    totalDecodeTimeMs,
    totalLatencyMs,
    endToEndTokensPerSec,
    ttftMs,
    itlMs: decodeTimePerTokenMs,
    firstAnswerLatencyMs,
    billedOutputTokens,
    cacheHitRate,
    cachePrefixTokens,
    cacheTTL,
    cacheProvider,
    prefillTokensAvoided,
    cacheWriteCostMultiplier,
    cacheReadCostMultiplier,
    effectivePrefillTokens,
    prefillTimeWithCacheMs,
    cacheSavingsPct,
    costPerHour: costPerHour * tp,
    costPerMillionOutputTokens,
    costPerRequest,
    costPerRequestWithCache,
    effectiveBandwidthGbps,
    effectiveFlopsTflops,
    quantEfficiency: quantEff,
    speculativeBoost,
    multiGpuEfficiency,
    continuousBatchingMultiplier,
    longContextWarning,
    // Per-metric confidence levels — surfaced as colored dots in the UI
    confidence: {
      // Throughput metrics
      decodeTokensPerSec: "modeled",        // Formula: HBM_BW × η_mem × quant_eff / model_size
      aggregateTokensPerSec: continuousBatchingMultiplier > 1 ? "heuristic" : "modeled",
                                            // Multiplier is heuristic, default 1.5× with cited 2-4× range
      prefillTokensPerSec: totalContextTokens > 32768 ? "heuristic" : "modeled",
                                            // Long-context superlinear attention is approximation
      batchCrossover: "modeled",           // Crossover formula is exact

      // Latency metrics
      prefillTimeMs: totalContextTokens > 32768 ? "heuristic" : "modeled",
      decodeTimePerTokenMs: "modeled",     // = 1000 / decode_tok_per_sec
      totalLatencyMs: totalContextTokens > 32768 ? "heuristic" : "modeled",
                                            // Inherits from prefill (long context dominates)
      ttftMs: totalContextTokens > 32768 ? "heuristic" : "modeled",
      itlMs: "modeled",
      firstAnswerLatencyMs: "modeled",   // TTFT + N × ITL; the token count is a stated assumption

      // Memory metrics
      modelSizeGb: "modeled",              // params × bytes_per_param
      kvCacheTotalGb: "modeled",           // Exact formula: 2·L·H_kv·D_h·B·T·batch
      totalVramNeededGb: "modeled",        // Sum of weights + KV
      vramFits: "modeled",

      // Cost metrics
      costPerHour: input.gpuHourlyCost !== undefined ? "user_supplied" : "published_price",
                                            // User override or default cloud price list
      costPerMillionOutputTokens: "modeled",
      costPerRequest: "modeled",

      // Cache metrics
      cacheSavingsPct: "modeled",
      prefillTokensAvoided: "modeled",
      cacheWriteCostMultiplier: "published_price",  // From Anthropic/OpenAI published pricing docs
      cacheReadCostMultiplier: "published_price",

      // Other
      billedOutputTokens: "user_supplied",  // User sets reasoning tokens
      effectiveBandwidthGbps: "modeled",   // GPU spec × utilization constants
      effectiveFlopsTflops: gpu.flopsTflops === null ? "heuristic" : "modeled",
                                            // B200/B300 publish no dense FP16 → conservative fallback
    },
  };
}

/** Format helpers */
export function fmtTokens(n: number): string {
  if (!isFinite(n)) return "—";
  if (n >= 1000) return n.toFixed(0);
  if (n >= 100) return n.toFixed(1);
  if (n >= 10) return n.toFixed(2);
  return n.toFixed(3);
}

export function fmtBytes(gb: number): string {
  if (gb >= 1024) return (gb / 1024).toFixed(2) + " TB";
  if (gb >= 1) return gb.toFixed(2) + " GB";
  return (gb * 1024).toFixed(1) + " MB";
}

/**
 * Format a byte quantity that is expressed in KILOBYTES.
 *
 * `fmtBytes` takes GIGABYTES. Feeding it KB (or KB/1024) and reading the
 * "MB" it prints back is a 1024x error: a 131.2 KB/token KV cache would be
 * reported as "131.2 MB/token", contradicting the "67.1 MB" total shown
 * right next to it. Use this for any per-token rate.
 */
export function fmtKb(kb: number): string {
  if (!isFinite(kb)) return "—";
  if (kb >= 1024 * 1024) return (kb / 1024 / 1024).toFixed(2) + " GB";
  if (kb >= 1024) return (kb / 1024).toFixed(2) + " MB";
  return kb.toFixed(1) + " KB";
}

export function fmtMs(ms: number): string {
  if (!isFinite(ms)) return "—";
  if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
  if (ms >= 1) return ms.toFixed(1) + " ms";
  return (ms * 1000).toFixed(1) + " µs";
}

export function fmtMoney(n: number): string {
  if (!isFinite(n) || n === 0) return "—";
  if (n >= 1000) return "$" + n.toFixed(0);
  if (n >= 1) return "$" + n.toFixed(2);
  if (n >= 0.01) return "$" + n.toFixed(4);
  return "$" + n.toFixed(6);
}

/* ============================================================
   LONG-CONTEXT CAPACITY HELPERS
   ============================================================
   The research brief calls this "the most important formula tokcalc should visibly expose":

     KV bytes/request = 2 · L · T · H_kv · D_h · B

   The factor of 2 stores both keys and values. With GQA (num_kv_heads < num_attention_heads),
   only the kv_heads' worth of cache is stored per layer.

   For dense attention, prefill cost grows SUPERLINEARLY:
     linear_flops    = 2 · N · T         (matmuls through linear layers)
     attention_flops ≈ T²/2 · H_kv · D_h · L   (QK^T + softmax · V)
   At long contexts (≥32K), attention becomes a significant fraction of total prefill FLOPs.
   Refs: RingAttention paper (https://arxiv.org/abs/2310.01889)
*/

/**
 * Bytes of KV cache per single token (for one request).
 *
 * For a plain full-attention model this is the classic formula:
 *   2 (K and V) × layers × kvHeads × headDim × bytesPerKVValue
 *
 * When the model declares an attention layout, each layer is costed by its own
 * kind:
 *   full     linear in context
 *   sliding  linear in context up to the window, then FLAT — the cache holds
 *            at most `windowTokens` entries, so a 128K request on a 128-token
 *            sliding layer costs the same as a 128-token one
 *   mla      one compressed latent (kvLoraRank) + a decoupled RoPE key
 *   linear   no per-token cache at all (see `computeStateBytes`)
 *   kda      no per-token cache at all (see `computeStateBytes`)
 *
 * `contextTokens` is required for sliding layers because their cost depends
 * on it. It is optional everywhere else and defaults to 0, which means "no
 * sliding discount" — the safe direction, since it never under-estimates.
 */
export function computeKVBytesPerToken(
  model: ModelSpec,
  bytesPerKVValue: number = 2, // FP16 = 2 bytes; can be INT8=1, FP4=0.5
  contextTokens: number = 0,
): number {
  if (!model.attentionLayers) {
    // 2 = K and V; model.layers = number of transformer blocks
    return 2 * model.layers * model.kvHeads * model.headDim * bytesPerKVValue;
  }

  let total = 0;
  for (const layer of model.attentionLayers) {
    switch (layer.kind) {
      case "full":
        total += 2 * model.kvHeads * model.headDim * bytesPerKVValue;
        break;

      case "sliding": {
        const perToken = 2 * model.kvHeads * model.headDim * bytesPerKVValue;
        const window = layer.windowTokens ?? 0;
        // The cache saturates at the window size. Below it, cost is linear;
        // above it, the marginal cost of extra context is zero.
        if (contextTokens > 0 && window > 0) {
          total += perToken * Math.min(1, window / contextTokens);
        } else {
          total += perToken;
        }
        break;
      }

      case "mla": {
        // Compressed latent + decoupled RoPE key, both stored once per token.
        const latent = layer.kvLoraRank ?? model.kvHeads * model.headDim;
        const rope = layer.mlaRopeHeadDim ?? 0;
        total += (latent + rope) * bytesPerKVValue;
        break;
      }

      case "linear":
      case "kda":
        // Recurrent state instead of a KV cache — no per-token cost.
        break;
    }
  }
  return total;
}

/**
 * Fixed recurrent state bytes for ONE sequence (not per token).
 *
 * Linear-attention and KDA layers keep a recurrent state matrix per sequence
 * instead of a growing key/value cache. That state is the same size whether
 * the sequence is 1K or 1M tokens long, so it is a fixed addition to the
 * per-request footprint that the old per-token formula could not express.
 */
export function computeStateBytes(model: ModelSpec): number {
  if (!model.attentionLayers) return 0;
  let total = 0;
  for (const layer of model.attentionLayers) {
    if (layer.kind === "linear" || layer.kind === "kda") {
      total += layer.stateBytes ?? 0;
    }
  }
  return total;
}

/**
 * Total KV + recurrent state size in GB for `contextTokens` tokens ×
 * `batchSize` concurrent requests.
 *
 * The state term is per-request, not per-token, so it is multiplied by
 * batchSize only.
 */
export function computeKVCacheGb(
  model: ModelSpec,
  contextTokens: number,
  batchSize: number = 1,
  bytesPerKVValue: number = 2,
): number {
  const perToken = computeKVBytesPerToken(model, bytesPerKVValue, contextTokens);
  const state = computeStateBytes(model);
  const bytes = perToken * contextTokens * batchSize + state * batchSize;
  return bytes / 1e9;
}

/** Maximum concurrent single-slot requests that fit in VRAM at the given context length.
 *
 *  = floor((total_vram − resident_weights) / kv_per_request_at_context)
 *
 * Notes, stated explicitly so callers can reconcile the number by hand:
 *  - `numGpus` is the GPU COUNT for the topology being priced (TP sharding
 *    splits both weights and KV across the group, so total VRAM scales with it).
 *  - `quantBytesPerParam` multiplies FULL `paramsB`, not `activeParamsB`: every
 *    MoE expert must be resident even though only a subset is active per token.
 *  - This counts single-slot requests. For a batch of size B, concurrent
 *    batches = maxConcurrentUsers / B.
 *  - No hidden reserve is applied — the result is exactly the floor of the
 *    ratio above, so it can be reproduced by hand. It excludes runtime
 *    overhead (activations, CUDA context, allocator slack), which real
 *    deployments must budget for on top; treat it as an upper bound.
 */
export function computeMaxConcurrency(
  model: ModelSpec,
  gpu: GpuSpec,
  numGpus: number,
  contextTokens: number,
  quantBytesPerParam: number,
  bytesPerKVValue: number = 2,
): number {
  const weightsBytes = model.paramsB * 1e9 * quantBytesPerParam;
  // Sliding layers saturate at their window, and linear/KDA layers hold a
  // fixed per-request state, so the per-request footprint is not simply
  // "bytes per token × context".
  const kvBytesPerRequest =
    computeKVBytesPerToken(model, bytesPerKVValue, contextTokens) * contextTokens +
    computeStateBytes(model);
  const totalVramBytes = gpu.vramGb * numGpus * 1e9;
  const availableKvBytes = totalVramBytes - weightsBytes;
  if (availableKvBytes <= 0 || kvBytesPerRequest <= 0) return 0;
  return Math.floor(availableKvBytes / kvBytesPerRequest);
}

/** Prefill latency in milliseconds for a given context length, accounting for
 *  superlinear attention cost beyond 32K tokens (dense attention only).
 *  For RingAttention/blockwise-attention, this correction overestimates the cost.
 */
export function computeLongContextPrefillMs(
  model: ModelSpec,
  effectiveFlopsTflops: number,
  promptTokens: number,
): number {
  if (promptTokens <= 0 || effectiveFlopsTflops <= 0) return 0;
  const T = promptTokens;
  const N = model.activeParamsB * 1e9;
  const linearFlops = 2 * N * T;
  // Attention O(N²) — only meaningful for long context
  // Each layer: QK^T = T·T·headDim, softmax·V = T·T·headDim
  const attentionFlops = (T * T) * model.kvHeads * model.headDim * model.layers;
  const totalFlops = linearFlops + attentionFlops;
  const effFlopsPerSec = effectiveFlopsTflops * 1e12;
  return (totalFlops / effFlopsPerSec) * 1000;
}

export interface TopologyRecommendation {
  topology: string;
  reason: string;
  fits: boolean;
  neededGpus: number;
  hasContextParallel: boolean;
}

/** Recommend a multi-GPU topology for the given config.
 *  - Single GPU: model + KV fit in 1 GPU
 *  - TP×2/4/8: shard weights across N GPUs (each holds 1/N of weights + KV)
 *  - TP×8 + Context Parallel: even TP×8 isn't enough — need RingAttention to shard KV across nodes
 */
export function recommendTopology(
  model: ModelSpec,
  gpu: GpuSpec,
  contextTokens: number,
  batchSize: number,
  quantBytesPerParam: number,
  bytesPerKVValue: number = 2,
): TopologyRecommendation {
  const weightsGb = model.paramsB * quantBytesPerParam;
  const kvPerRequestGb = computeKVGb(model, contextTokens, 1, bytesPerKVValue);
  const totalKvGb = kvPerRequestGb * batchSize;
  const totalNeededGb = weightsGb + totalKvGb;

  // Try single-GPU first
  if (totalNeededGb <= gpu.vramGb) {
    return {
      topology: "Single GPU",
      reason: `${totalNeededGb.toFixed(1)} GB ≤ ${gpu.vramGb} GB — fits in 1 ${gpu.name.split(" ")[0]}`,
      fits: true,
      neededGpus: 1,
      hasContextParallel: false,
    };
  }
  // Try TP×2, TP×4, TP×8
  for (const tp of [2, 4, 8]) {
    if (totalNeededGb <= gpu.vramGb * tp) {
      return {
        topology: `Tensor Parallel ×${tp}`,
        reason: `${totalNeededGb.toFixed(1)} GB > ${gpu.vramGb} GB single → shard across ${tp} GPUs (${gpu.vramGb * tp} GB total, weights + KV split evenly)`,
        fits: true,
        neededGpus: tp,
        hasContextParallel: false,
      };
    }
  }
  // TP×8 not enough → need Context Parallel (RingAttention)
  return {
    topology: "TP×8 + Context Parallel",
    reason: `${totalNeededGb.toFixed(1)} GB exceeds ${gpu.vramGb * 8} GB even with TP×8. Use Context Parallel (RingAttention) to shard KV cache across nodes. Ref: arxiv.org/abs/2310.01889`,
    fits: false,
    neededGpus: 8,
    hasContextParallel: true,
  };
}

/** Helper for above — used internally. */
function computeKVGb(
  model: ModelSpec,
  contextTokens: number,
  batchSize: number,
  bytesPerKVValue: number,
): number {
  return computeKVCacheGb(model, contextTokens, batchSize, bytesPerKVValue);
}

/** Format a context length nicely: 8192 → "8K", 131072 → "128K", 1000000 → "1M". */
export function fmtContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    // Show "1M", "1.5M", "2M"
    const m = tokens / 1_000_000;
    if (m >= 1) {
      const rounded = Math.round(m * 10) / 10;
      return Number.isInteger(rounded) ? `${rounded}M` : `${rounded.toFixed(1)}M`;
    }
  }
  if (tokens >= 1000) {
    // Floor for power-of-2 context sizes (4096→"4K", 131072→"131K")
    // Use 1 decimal only if not within 1 of an integer
    const k = tokens / 1000;
    const rounded = Math.round(k);
    if (Math.abs(k - rounded) < 0.5) {
      return `${rounded}K`;
    }
    return `${k.toFixed(1)}K`;
  }
  return String(tokens);
}
