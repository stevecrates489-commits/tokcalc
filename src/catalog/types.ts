/**
 * Catalog types — the contract between the JSON records and the engine.
 *
 * The catalog lives in `catalog/*.json`. This file describes what a record
 * IS; `src/catalog/loader.ts` loads them; `src/catalog/schemas.ts` validates
 * them. The calculation engine (`src/lib/token-calc.ts`) consumes the loaded
 * records and knows nothing about where they came from.
 *
 * Every literal union is declared as a const array + derived type, so the
 * Zod schema in schemas.ts and the TypeScript type here can never drift:
 * both read from the same array.
 */

// ---------------------------------------------------------------------------
// Shared vocabulary
// ---------------------------------------------------------------------------

/**
 * Where a record's numbers came from.
 *
 * `unspecified` is a first-class value on purpose: a record that predates
 * the provenance requirement can say so honestly instead of borrowing
 * credibility from a source it does not have.
 */
export const PROVENANCE_SOURCES = [
  "vendor-spec",
  "model-card",
  "hf-config",
  "hf-safetensors",
  "published-price",
  "benchmark-run",
  "community",
  "derived",
  "unspecified",
] as const;
export type ProvenanceSource = (typeof PROVENANCE_SOURCES)[number];

export const LIFECYCLES = ["active", "deprecated", "experimental", "planned"] as const;
export type Lifecycle = (typeof LIFECYCLES)[number];

/**
 * How far a record's values can be relied on for a given question.
 *
 * This is deliberately SEPARATE from `ProvenanceSource`. Provenance answers
 * "where did this number come from?"; applicability answers "does that source
 * actually establish this particular claim?".
 *
 * The distinction matters because a source being authoritative for one
 * dimension says nothing about another: a vendor spec sheet may state memory
 * capacity precisely while saying nothing about serving throughput, and a real
 * observed run may be perfectly genuine while being irrelevant to the context
 * length you are planning for.
 *
 *  - "exact"    → the source directly and completely establishes the value.
 *  - "partial"  → the source establishes part of it, or only under conditions
 *                 that do not necessarily hold here (see `limitations`).
 *  - "lower-bound" → the modelled value is a floor: real usage is at least this.
 */
export const APPLICABILITIES = ["exact", "partial", "lower-bound"] as const;
export type Applicability = (typeof APPLICABILITIES)[number];

export interface Provenance {
  /** Which kind of source the values came from. */
  source: ProvenanceSource;
  /** Where to verify. Required for any source other than "unspecified". */
  sourceUrl?: string;
  /** ISO date the values were read from the source (YYYY-MM-DD). */
  retrievedAt?: string;
  /** Free-form note: caveats, disagreements, what is still unverified. */
  notes?: string;
  /**
   * How far the source establishes the values, per dimension. Defaults to
   * "exact" when not stated — the common case is a source that simply does
   * establish the number.
   */
  applicability?: Applicability;
  /**
   * Specific conditions under which the values do not hold as stated.
   *
   * Free-form strings, because the reasons are genuinely heterogeneous
   * ("DSA index cache not modelled", "measured at a different token
   * distribution"). The point is that a reader can see the caveat without
   * having to infer it.
   */
  limitations?: string[];
}

// ---------------------------------------------------------------------------
// Quantizations
// ---------------------------------------------------------------------------

export const QUANTIZATION_IDS = [
  "fp32", "fp16", "bf16", "int8", "int4",
  "gguf-q2k", "gguf-q3km", "gguf-q4km", "gguf-q5km", "gguf-q6k", "gguf-q8",
  "gptq4", "awq4", "exl2-6bpw", "fp8", "nvfp4",
] as const;
export type Quantization = (typeof QUANTIZATION_IDS)[number];

export const QUANT_FAMILIES = [
  "float", "gguf", "gptq", "awq", "exl2", "fp8", "nvfp4",
] as const;
export type QuantFamily = (typeof QUANT_FAMILIES)[number];

export interface QuantMeta {
  id: Quantization;
  label: string;
  /** Effective bytes per parameter (averaged across tensor mix for GGUF). */
  bytesPerParam: number;
  /** Dequantization overhead multiplier (1.0 = none, 0.85 = 15% overhead)
   *  For FP8 on H100: efficiency >1.0 (faster than FP16 due to native FP8 kernels) */
  efficiency: number;
  description: string;
  /** Format family — for engine-matching UI */
  family: QuantFamily;
  /** Typical use case */
  useCase?: string;
}

// ---------------------------------------------------------------------------
// Accelerators
// ---------------------------------------------------------------------------

export const GPU_CATEGORIES = [
  "datacenter", "workstation", "consumer", "mac", "tpu", "lpu", "wse", "rdu", "legacy",
] as const;
export type GpuCategory = (typeof GPU_CATEGORIES)[number];

export interface GpuSpec {
  id: string;
  name: string;
  vendor: string;
  category: GpuCategory;
  /** HBM bandwidth in GB/s */
  memBandwidthGbps: number;
  /** FP16/BF16 compute in TFLOPS (dense, not sparse). null = vendor does not publish dense FP16 (e.g. B200 exposes FP4/FP8 only) */
  flopsTflops: number | null;
  /** VRAM in GB */
  vramGb: number;
  /** NVLink bandwidth per GPU in GB/s (0 if no NVLink) */
  nvlinkGbps: number;
  /** Typical on-demand $/hr in USD (cloud, rough 2025-2026 estimate). null = quote required */
  usdPerHour: number | null;
  /** Release year */
  year: number;
  /** Special note (e.g. 'FP16 dense not publicly reported', 'rack-scale system', etc.) */
  note?: string;
  /** HuggingFace URL or vendor source for verification */
  sourceUrl?: string;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export const MODEL_CATEGORIES = ["text", "vlm", "embedding", "code", "reasoning"] as const;
export type ModelCategory = (typeof MODEL_CATEGORIES)[number];

/**
 * How one transformer layer stores its "past".
 *
 * This is the single most important architectural fact for capacity planning.
 * The naive formula — 2 × layers × kvHeads × headDim bytes per token — assumes
 * every layer is plain full attention. That assumption is now wrong for a
 * meaningful share of production checkpoints:
 *
 *   full     Standard GQA/MHA. Cache grows linearly with context.
 *   sliding  Attention over a bounded window. The cache STOPS GROWING once
 *            context exceeds the window, so long-context cost is bounded.
 *   linear   Linear/recurrent attention (Gated DeltaNet and friends). No KV
 *            cache at all — a fixed-size recurrent state instead, which does
 *            NOT grow with context.
 *   kda      Kimi Delta Attention: a linear-attention variant with its own
 *            state layout.
 *   mla      Multi-head Latent Attention. The cache stores one compressed
 *            latent vector of `kvLoraRank` plus a small decoupled RoPE key —
 *            not kvHeads × headDim.
 */
export const ATTENTION_KINDS = ["full", "sliding", "linear", "kda", "mla"] as const;
export type AttentionKind = (typeof ATTENTION_KINDS)[number];

export interface LayerAttention {
  kind: AttentionKind;
  /** Sliding window size in tokens. `sliding` only. */
  windowTokens?: number;
  /** MLA compressed latent rank. `mla` only. */
  kvLoraRank?: number;
  /** MLA decoupled RoPE key head dim, stored alongside the latent. `mla` only. */
  mlaRopeHeadDim?: number;
  /**
   * Recurrent state bytes for ONE sequence in this layer. `linear` / `kda`
   * only. Constant with respect to context length.
   */
  stateBytes?: number;
}

/**
 * How much of the calculation engine actually supports a model's architecture.
 *
 *   supported     The attention layout is modelled exactly; throughput and
 *                 capacity numbers are usable.
 *   memory-only   We can size the resident weights and the KV/state footprint,
 *                 but the decode-throughput formula assumes full attention, so
 *                 treat throughput as an upper bound.
 *   unsupported   We know the weights exist but not enough to model them.
 *                 Excluded from calculations entirely.
 */
export const ARCHITECTURE_STATUSES = ["supported", "memory-only", "unsupported"] as const;
export type ArchitectureStatus = (typeof ARCHITECTURE_STATUSES)[number];

export interface ModelSpec {
  id: string;
  name: string;
  family: string;
  category: ModelCategory;
  /** Total parameters in billions */
  paramsB: number;
  /** Parameters actually used per token (differs for MoE) */
  activeParamsB: number;
  layers: number;
  hiddenDim: number;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  vocabSize: number;
  maxContext: number;
  /** is MoE */
  isMoE: boolean;

  // ---- Architecture-aware extensions (optional) ----

  /**
   * Per-layer attention layout. Omitted means "every layer is plain full
   * GQA", which keeps every pre-existing catalog entry bit-for-bit identical.
   */
  attentionLayers?: LayerAttention[];
  /**
   * How completely the engine models this architecture.
   * Defaults to "supported" when `attentionLayers` is absent.
   */
  architectureStatus?: ArchitectureStatus;
  /** Native weight format of the official checkpoint, when it is not plain FP16/BF16. */
  nativeWeightFormat?: string;
  /** Upstream HuggingFace repository, for provenance. */
  repository?: string;
  /** Pinned revision of the upstream repository. */
  revision?: string;
  /** SPDX license id. */
  license?: string;
  /**
   * Resident weight bytes measured from the upstream safetensors index.
   * When present it is the authoritative figure for loading cost, and it
   * settles marketing-vs-tensor parameter-count disagreements.
   */
  residentWeightBytes?: number;
  /** Native context window, when it differs from the extended maximum. */
  nativeContext?: number;
  /** How the extended context is reached (RoPE scaling, YaRN, …). */
  contextExtensionMethod?: string;
  /** Context window the runtime supports by default, when below the maximum. */
  runtimeContextLimit?: number;
}

// ---------------------------------------------------------------------------
// Records — what actually lives in the JSON files
// ---------------------------------------------------------------------------

/** A model as it appears in `catalog/models.json`. */
export interface ModelRecord extends ModelSpec {
  /** Lifecycle state. Deprecated records stay in the catalog for historical
   *  URLs but are excluded from new recommendations. */
  lifecycle: Lifecycle;
  provenance: Provenance;
}

/** An accelerator as it appears in `catalog/accelerators.json`. */
export interface AcceleratorRecord extends GpuSpec {
  lifecycle: Lifecycle;
  provenance: Provenance;
}

/** A quantization as it appears in `catalog/quantizations.json`. */
export interface QuantizationRecord extends QuantMeta {
  lifecycle: Lifecycle;
  provenance: Provenance;
}
