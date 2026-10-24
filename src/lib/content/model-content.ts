/**
 * Programmatic content engine for per-model pages.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The three hand-written comparison pages under src/app/compare/ embed their
 * spec tables as literal strings ("141 GB HBM3e", "~$2.50/hr", "700W"). Those
 * are snapshots of the catalog taken by hand, and they silently rot: the day
 * someone updates `usdPerHour` in token-calc.ts, the prose keeps claiming the
 * old number and the page starts lying to search engines.
 *
 * Every fact on a generated page is computed here from `token-calc.ts` — the
 * same module the calculator, the MCP server, and the CLI use. There is no
 * second source of truth to fall out of sync.
 *
 * The rule: never hand-type a spec value into a page. Derive it.
 */

import {
  calculate,
  computeKVBytesPerToken,
  computeStateBytes,
  fmtKb,
  fmtTokens,
  fmtMoney,
  GPU_MAP,
  GPUS,
  MODEL_MAP,
  MODELS,
  QUANT_MAP,
  QUANTIZATIONS,
  type CalcResult,
  type GpuSpec,
  type ModelSpec,
  type Quantization,
  type QuantMeta,
} from "@/lib/token-calc";

/** Reference serving workload used for every "does it fit / how fast / how much" figure. */
export const REF_CONTEXT = 8192;
export const REF_BATCH = 8;
export const REF_OUTPUT = 500;

/** Context used for the "can it even load" minimum-VRAM claim. */
export const MIN_LOAD_CONTEXT = 4096;

/**
 * The three quantizations every model page leads with.
 *
 * These are the three deployment modes people actually choose between:
 * FP16 (no quality loss), FP8 (native on Hopper/Blackwell, faster than FP16),
 * and AWQ 4-bit (the standard cheap-and-good server quant on vLLM).
 */
export const PRIMARY_QUANTS: Quantization[] = ["fp16", "fp8", "awq4"];

/**
 * Quantization recommended as the default for a model, given its size.
 *
 * Under ~15B everything fits comfortably in FP16 on a single mid-range GPU,
 * so FP16 is the honest default. Above that, FP8 is what real deployments run
 * (native kernels on Hopper/Blackwell), and only for the very largest models
 * do we fall back to AWQ 4-bit because FP8 still needs 2 GPUs.
 */
export function defaultQuantFor(model: ModelSpec): Quantization {
  if (model.paramsB <= 15) return "fp16";
  if (model.paramsB <= 100) return "fp8";
  return "awq4";
}

/** GPU counts we consider as candidates for a deployment. */
const GPU_COUNT_LADDER = [1, 2, 4, 8] as const;

export interface QuantRow {
  quant: QuantMeta;
  /** Resident weight footprint in GB (all params × bytes/param). */
  weightsGb: number;
  /** Minimum VRAM to hold weights + a single 4K request's KV cache. */
  minVramGb: number;
  /** Smallest single GPU in the catalog that can hold the weights alone. */
  smallestGpuThatHolds: GpuSpec | null;
}

export interface GpuFit {
  gpu: GpuSpec;
  /** Smallest GPU count that fits the reference workload; null when nothing up to 8 fits. */
  gpuCount: number | null;
  fits: boolean;
  /** Result of running the reference workload on the chosen GPU count. */
  result: CalcResult;
  /** Total VRAM across the chosen GPUs. */
  capacityGb: number;
  /** USD per 1M output tokens at the reference workload. */
  costPerMillion: number;
  /** Aggregate tokens/sec across the chosen GPUs. */
  aggregateTokensPerSec: number;
  /** Output tokens/sec per dollar/hour — the value metric used for ranking. */
  tokensPerDollar: number;
  /** True when a single GPU of this type is enough. */
  singleGpu: boolean;
  /**
   * True when this hardware has a known hourly rental price.
   *
   * `usdPerHour` is null for quote-required hardware (Groq, Cerebras) and 0
   * for purchased hardware (Apple Silicon). The cost model turns both into
   * $0/hr, which would rank them as "the cheapest option" — so they are
   * excluded from every price ranking here.
   */
  rentable: boolean;
}

export interface ModelContent {
  model: ModelSpec;
  defaultQuant: Quantization;
  quantRows: QuantRow[];
  /** Every catalog GPU, ranked: fitting first, then by value (tokens per $). */
  gpuFits: GpuFit[];
  /** Cheapest configuration that fits the reference workload. */
  cheapest: GpuFit | null;
  /** Fastest configuration that fits the reference workload. */
  fastest: GpuFit | null;
  /** Any configuration where one GPU is enough (the headline "can it fit on one?" answer). */
  singleGpuOptions: GpuFit[];
  faqs: { q: string; a: string }[];
}

function calcFor(
  modelId: string,
  gpuId: string,
  quant: Quantization,
  numGpus: number,
  context: number,
  batch: number,
): CalcResult {
  return calculate({
    modelId,
    gpuId,
    quantization: quant,
    numGpus,
    batchSize: batch,
    promptTokens: context,
    outputTokens: REF_OUTPUT,
    useContinuousBatching: true,
  });
}

/**
 * Find the smallest GPU count (1/2/4/8) on which `gpu` can run the model at
 * the reference workload. Returns null when even 8 is not enough.
 */
function findFittingCount(
  modelId: string,
  gpu: GpuSpec,
  quant: Quantization,
): { gpuCount: number; result: CalcResult } | null {
  for (const n of GPU_COUNT_LADDER) {
    const result = calcFor(modelId, gpu.id, quant, n, REF_CONTEXT, REF_BATCH);
    if (result.vramFits) return { gpuCount: n, result };
  }
  return null;
}

/**
 * Compute every figure needed to render a model's page.
 *
 * Pure and deterministic — the same model id always produces the same page,
 * which is what makes the output safe to pre-render at build time.
 */
export function computeModelContent(modelId: string): ModelContent | null {
  const model = MODEL_MAP[modelId];
  if (!model) return null;

  const defaultQuant = defaultQuantFor(model);

  // ---- VRAM per quantization -------------------------------------------
  const quantRows: QuantRow[] = PRIMARY_QUANTS.map((qid) => {
    const quant = QUANT_MAP[qid];
    const weightsGb = model.paramsB * quant.bytesPerParam;
    const kvPerRequestGb =
      (2 * model.layers * model.kvHeads * model.headDim * 2 * MIN_LOAD_CONTEXT) / 1e9;
    const minVramGb = weightsGb + kvPerRequestGb;

    // Smallest single GPU (by VRAM) that can hold the weights plus one request.
    // Wafer-scale systems and LPUs are excluded: they are not per-card units,
    // so naming one here would misstate the hardware requirement.
    const smallestGpuThatHolds =
      [...GPUS]
        .filter(
          (g) =>
            g.vramGb >= minVramGb && g.category !== "wse" && g.category !== "lpu",
        )
        .sort((a, b) => a.vramGb - b.vramGb)[0] ?? null;

    return { quant, weightsGb, minVramGb, smallestGpuThatHolds };
  });

  // ---- Per-GPU fit / throughput / cost ----------------------------------
  const gpuFits: GpuFit[] = GPUS.map((gpu) => {
    const fit = findFittingCount(modelId, gpu, defaultQuant);
    const rentable = (gpu.usdPerHour ?? 0) > 0;

    if (!fit) {
      const oversized = calcFor(modelId, gpu.id, defaultQuant, 8, REF_CONTEXT, REF_BATCH);
      return {
        gpu,
        gpuCount: null,
        fits: false,
        result: oversized,
        capacityGb: gpu.vramGb * 8,
        costPerMillion: 0,
        aggregateTokensPerSec: 0,
        tokensPerDollar: 0,
        singleGpu: false,
        rentable,
      };
    }

    const { gpuCount, result } = fit;
    const capacityGb = gpu.vramGb * gpuCount;
    const hourlyCost = (gpu.usdPerHour ?? 0) * gpuCount;
    const costPerMillion = result.costPerMillionOutputTokens;
    const tokensPerDollar =
      hourlyCost > 0 ? result.aggregateTokensPerSec / hourlyCost : 0;

    return {
      gpu,
      gpuCount,
      fits: true,
      result,
      capacityGb,
      costPerMillion,
      aggregateTokensPerSec: result.aggregateTokensPerSec,
      tokensPerDollar,
      singleGpu: gpuCount === 1,
      rentable,
    };
  });

  const fitting = gpuFits.filter((g) => g.fits);
  const rentableFitting = fitting.filter((g) => g.rentable);

  // Cheapest per million tokens — only among hardware with a known rental
  // price. Quote-required and purchased hardware have no comparable cost, so
  // ranking them would silently report $0.
  const cheapest = rentableFitting.length
    ? rentableFitting.reduce((best, g) => (g.costPerMillion < best.costPerMillion ? g : best))
    : null;
  // Fastest = highest aggregate throughput. This is a performance fact and is
  // meaningful even for hardware without a published price.
  const fastest = fitting.length
    ? fitting.reduce((best, g) =>
        g.aggregateTokensPerSec > best.aggregateTokensPerSec ? g : best,
      )
    : null;

  // Ranking: fitting configs first, then by value (tokens per dollar-hour),
  // with single-GPU configs nudged ahead of multi-GPU ones of equal value.
  const ranked = [...gpuFits].sort((a, b) => {
    if (a.fits !== b.fits) return a.fits ? -1 : 1;
    if (a.fits) {
      const valueDelta = b.tokensPerDollar - a.tokensPerDollar;
      if (Math.abs(valueDelta) > 1e-9) return valueDelta;
      if (a.singleGpu !== b.singleGpu) return a.singleGpu ? -1 : 1;
      return (a.gpuCount ?? 0) - (b.gpuCount ?? 0);
    }
    return b.gpu.vramGb - a.gpu.vramGb;
  });

  const singleGpuOptions = fitting
    .filter((g) => g.singleGpu)
    // Wafer-scale systems and LPUs are not per-card units, so "fits on one
    // card" is a category error for them. They stay in the GPU ranking (where
    // they legitimately are the fastest option) but they never answer the
    // single-GPU question.
    .filter((g) => g.gpu.category !== "wse" && g.gpu.category !== "lpu")
    .sort((a, b) => a.gpu.vramGb - b.gpu.vramGb);

  /** Every single-card fit, including wafer-scale/LPU systems. */
  const allSingleGpu = fitting.filter((g) => g.singleGpu);

  const fp16Row = quantRows.find((r) => r.quant.id === "fp16")!;
  const defRow = quantRows.find((r) => r.quant.id === defaultQuant)!;

  // ---- FAQs, generated from the computed numbers ------------------------
  const faqs = buildFaqs(
    model,
    defaultQuant,
    fp16Row,
    defRow,
    singleGpuOptions,
    allSingleGpu,
    cheapest,
    fastest,
  );

  return {
    model,
    defaultQuant,
    quantRows,
    gpuFits: ranked,
    cheapest,
    fastest,
    singleGpuOptions,
    faqs,
  };
}

function buildFaqs(
  model: ModelSpec,
  defaultQuant: Quantization,
  fp16Row: QuantRow,
  defRow: QuantRow,
  singleGpuOptions: GpuFit[],
  allSingleGpu: GpuFit[],
  cheapest: GpuFit | null,
  fastest: GpuFit | null,
): { q: string; a: string }[] {
  const faqs: { q: string; a: string }[] = [];

  // "Can it fit on one GPU?"
  if (singleGpuOptions.length > 0) {
    const smallest = singleGpuOptions[0];
    const fp16Smallest = fp16Row.smallestGpuThatHolds;
    faqs.push({
      q: `Can ${model.name} run on a single GPU?`,
      a: `Yes. At ${defRow.quant.label} the model needs ${defRow.minVramGb.toFixed(1)} GB of VRAM, so it fits on a single ${smallest.gpu.name} or anything with more VRAM.${
        fp16Smallest
          ? ` At full FP16 it needs ${fp16Row.minVramGb.toFixed(1)} GB, which fits on one ${fp16Smallest.name} or larger.`
          : ""
      }`,
    });
  } else if (allSingleGpu.length > 0) {
    // Only wafer-scale / LPU systems fit — say so precisely rather than
    // claiming the model does not fit anywhere.
    const only = allSingleGpu[0];
    faqs.push({
      q: `Can ${model.name} run on a single GPU?`,
      a: `Not on a single card. At ${defRow.quant.label} the model needs ${defRow.minVramGb.toFixed(1)} GB of VRAM, so it does not fit in any per-GPU accelerator in the catalog. It does fit on a ${only.gpu.name} — a ${only.gpu.category === "wse" ? "wafer-scale" : "LPU"} system, which is a different class of deployment.`,
    });
  } else {
    const smallestHolder = fp16Row.smallestGpuThatHolds;
    faqs.push({
      q: `Can ${model.name} run on a single GPU?`,
      a: `Not at ${defRow.quant.label} with any GPU in the current catalog — the weights alone need ${defRow.weightsGb.toFixed(0)} GB. ${
        smallestHolder
          ? `The smallest GPU in the catalog with enough VRAM is the ${smallestHolder.name} (${smallestHolder.vramGb} GB).`
          : "You will need multiple GPUs, or a quantisation below AWQ 4-bit."
      }`,
    });
  }

  // "How many GPUs?"
  const minGpus = cheapest?.gpuCount ?? null;
  faqs.push({
    q: `How many GPUs does ${model.name} need?`,
    a: minGpus
      ? `At ${defRow.quant.label} with ${REF_CONTEXT.toLocaleString()}-token context and ${REF_BATCH} concurrent users, the cheapest configuration that fits is ${minGpus}× ${cheapest!.gpu.name} (${cheapest!.capacityGb} GB total VRAM). ${fmtMoney(cheapest!.costPerMillion)} per 1M output tokens.`
      : fastest
        ? `At ${defRow.quant.label} with ${REF_CONTEXT.toLocaleString()}-token context and ${REF_BATCH} concurrent users, the fastest configuration that fits is ${fastest.gpuCount}× ${fastest.gpu.name} (${fastest.capacityGb} GB total VRAM). No configuration in the catalog has a published rental price, so a per-million-token figure is not available.`
        : `${model.name} at ${defRow.quant.label} does not fit within 8× any single GPU type in the catalog. Use context parallelism across nodes, or quantise further.`,
  });

  // "How fast?"
  if (fastest) {
    faqs.push({
      q: `How fast can ${model.name} be served?`,
      a: `On ${fastest.gpuCount}× ${fastest.gpu.name} at ${defRow.quant.label}, the aggregate throughput is ${fmtTokens(fastest.aggregateTokensPerSec)} tokens/sec with continuous batching${
        fastest.rentable
          ? `, at ${fmtMoney(fastest.costPerMillion)} per 1M output tokens.`
          : `. That hardware has no published rental price, so its cost per million tokens is not comparable.`
      }`,
    });
  }

  // "How much VRAM?"
  faqs.push({
    q: `How much VRAM does ${model.name} need?`,
    a: `Weights alone: ${fp16Row.weightsGb.toFixed(1)} GB at FP16, ${defRow.weightsGb.toFixed(1)} GB at ${defRow.quant.label}. Add KV cache for your context length and concurrency — every concurrent request at ${REF_CONTEXT.toLocaleString()} tokens costs extra VRAM, so budget headroom above the weight figure.`,
  });

  // "FP8 vs FP16?"
  if (model.paramsB > 15) {
    const fp8WeightsGb = model.paramsB * QUANT_MAP["fp8"].bytesPerParam;
    faqs.push({
      q: `Should I run ${model.name} in FP8 or FP16?`,
      a: `FP8 halves the weight footprint (${fp16Row.weightsGb.toFixed(0)} GB → ${fp8WeightsGb.toFixed(0)} GB) and is faster than FP16 on Hopper and Blackwell, where it has native kernels. FP16 remains the reference quality bar; FP8 loses a small amount of accuracy in exchange for roughly doubling the batch size you can serve per GPU.`,
    });
  }

  // "MoE?"
  if (model.isMoE) {
    faqs.push({
      q: `Why does ${model.name} need more VRAM than its active parameter count suggests?`,
      a: `${model.name} is a mixture-of-experts model: ${model.activeParamsB.toFixed(0)}B parameters are active per token, but all ${model.paramsB.toFixed(0)}B experts must stay resident in VRAM. Capacity is bounded by the resident set, while decode throughput is bounded by the active set — so it serves fast per token but needs the full footprint.`,
    });
  }

  // "Is it cheaper than an API?"
  if (cheapest && cheapest.costPerMillion > 0) {
    faqs.push({
      q: `Is it cheaper to self-host ${model.name} than to call an API?`,
      a: `At ${cheapest.gpuCount}× ${cheapest.gpu.name} the marginal cost is ${fmtMoney(cheapest.costPerMillion)} per 1M output tokens, before you add idle capacity, ops staff, or egress. That is typically cheaper than frontier API pricing at sustained volume, but API calls win when your load is bursty or too small to keep a GPU busy. The Build vs Buy tab on tokcalc prices this against a specific provider for you.`,
    });
  }

  return faqs;
}

/** Every model id that has a page. */
export const MODEL_PAGE_IDS = MODELS.map((m) => m.id);

/**
 * Human-readable description of a model's attention layout, and the KV cost
 * that layout actually produces.
 *
 * The old generic formula (2 × layers × kvHeads × headDim) is only right for
 * a model whose every layer is plain full attention. Qwen3.8 mixes in
 * linear-attention layers that hold no KV cache at all, gpt-oss alternates
 * sliding and full layers, and Kimi K3 / GLM-5.3 use compressed latent
 * attention. So the number has to come from the engine, not from a formula
 * retyped into a page.
 */
export interface AttentionSummary {
  /** Short label, e.g. "Hybrid: 16 full + 48 linear attention". */
  label: string;
  /** One-line explanation of how the cache behaves. */
  detail: string;
  /** KV bytes per token at the reference context. */
  kvPerTokenKb: number;
  /** Fixed recurrent state per sequence, in GB. Zero for ordinary models. */
  stateGb: number;
}

export function attentionSummary(model: ModelSpec): AttentionSummary {
  const kvPerTokenKb = computeKVBytesPerToken(model, 2, REF_CONTEXT) / 1024;
  const stateGb = computeStateBytes(model) / 1e9;

  if (!model.attentionLayers) {
    return {
      label: "Full attention (GQA)",
      detail:
        "Every layer caches keys and values for the whole context, so the KV cache grows linearly with conversation length.",
      kvPerTokenKb,
      stateGb,
    };
  }

  const counts = new Map<string, number>();
  for (const l of model.attentionLayers) {
    counts.set(l.kind, (counts.get(l.kind) ?? 0) + 1);
  }
  const parts: string[] = [];
  const NAMES: Record<string, string> = {
    full: "full",
    sliding: "sliding",
    linear: "linear",
    kda: "KDA",
    mla: "MLA",
  };
  // Fixed order, so the label reads the same however the layers interleave.
  for (const kind of ["full", "sliding", "linear", "kda", "mla"]) {
    const n = counts.get(kind);
    if (n) parts.push(`${n} ${NAMES[kind]}`);
  }
  const label = `Hybrid: ${parts.join(" + ")}`;

  const details: string[] = [];
  if (counts.has("sliding")) {
    details.push(
      "sliding layers stop growing once the context passes their window, so long-context cost is bounded",
    );
  }
  if (counts.has("linear") || counts.has("kda")) {
    details.push(
      "linear-attention layers keep a fixed recurrent state instead of a growing key/value cache",
    );
  }
  if (counts.has("mla")) {
    details.push(
      "MLA layers store one compressed latent plus a small decoupled RoPE key, not a full key/value pair",
    );
  }

  return {
    label,
    detail: details.length
      ? `${details.join("; ")}.`
      : "Every layer caches keys and values for the whole context.",
    kvPerTokenKb,
    stateGb,
  };
}

/** Convenience map for fast lookup across the app. */
export const MODEL_CONTENT_MAP: Record<string, ModelContent> = Object.fromEntries(
  MODEL_PAGE_IDS.map((id) => {
    const c = computeModelContent(id);
    return [id, c!];
  }),
);

/**
 * Model × GPU combinations worth a dedicated page.
 *
 * Two rules, both about not publishing junk:
 *   1. Every model gets a page for each GPU it can run on by itself — that
 *      is the question people actually type into a search box.
 *   2. Large models additionally get pages for the datacenter GPUs people
 *      deploy on, even when the model needs more than one of them.
 *
 * Everything else (a 405B model on a T4) is not a page anyone would read.
 */
export const PRIMARY_GPU_IDS = [
  "h100-sxm",
  "h200-sxm",
  "a100-80",
  "l40s",
  "b200-sxm",
  "rtx-5090",
];

export interface ModelGpuPair {
  modelId: string;
  gpuId: string;
}

export function modelGpuPairs(): ModelGpuPair[] {
  const pairs: ModelGpuPair[] = [];
  for (const modelId of MODEL_PAGE_IDS) {
    const content = MODEL_CONTENT_MAP[modelId];
    if (!content) continue;

    const seen = new Set<string>();
    for (const fit of content.gpuFits) {
      const include =
        (fit.fits && fit.singleGpu) ||
        (PRIMARY_GPU_IDS.includes(fit.gpu.id) && fit.fits);
      if (!include || seen.has(fit.gpu.id)) continue;
      seen.add(fit.gpu.id);
      pairs.push({ modelId, gpuId: fit.gpu.id });
    }
  }
  return pairs;
}

/**
 * Set of "modelId|gpuId" keys that have a dedicated model-on-GPU page.
 *
 * The pages are curated (see `modelGpuPairs`), but the model page's GPU table
 * lists every configuration that fits. Anything outside this set must render
 * as plain text rather than a link, or the link 404s.
 */
export const MODEL_GPU_PAGE_KEYS = new Set(
  modelGpuPairs().map(({ modelId, gpuId }) => `${modelId}|${gpuId}`),
);

/** True when a dedicated model-on-GPU page exists for this combination. */
export function hasModelGpuPage(modelId: string, gpuId: string): boolean {
  return MODEL_GPU_PAGE_KEYS.has(`${modelId}|${gpuId}`);
}

/** Human label for a model, e.g. "Llama 3.3 70B". */
export function modelLabel(modelId: string): string {
  return MODEL_MAP[modelId]?.name ?? modelId;
}

/**
 * How many concurrent requests fit in VRAM at a given context length.
 *
 * Returns null when the model does not fit at all (not even one request),
 * which is the "red" state on the long-context chart.
 */
export function concurrencyAtContext(
  modelId: string,
  gpuId: string,
  gpuCount: number,
  quant: Quantization,
  contextTokens: number,
): number | null {
  const gpu = GPU_MAP[gpuId];
  if (!gpu) return null;

  // Walk batch size down from a generous ceiling until it stops fitting.
  // This mirrors what the calculator's long-context chart does.
  const MAX_BATCH = 4096;
  let lo = 0;
  let hi = MAX_BATCH;
  let best = 0;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (mid === 0) break;
    const r = calcFor(modelId, gpuId, quant, gpuCount, contextTokens, mid);
    if (r.vramFits) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
    if (hi < 0) break;
  }
  return best > 0 ? best : null;
}

/** Context lengths used across the long-context tables and charts. */
export const CONTEXT_LADDER = [4096, 8192, 32768, 65536, 131072] as const;

/**
 * Caveat for a multi-GPU configuration on hardware without NVLink.
 *
 * The shared capacity model applies a flat 0.85× multi-GPU efficiency factor
 * whenever tensor-parallel degree > 1. That is a reasonable stand-in for
 * NVLink-equipped datacenter cards, but it is optimistic for consumer GPUs,
 * which scale out over PCIe — all-to-all traffic there is substantially
 * slower than the factor implies.
 *
 * We surface this on the generated pages rather than changing the shared
 * model, because the calculator, the MCP server, and the CLI all consume the
 * same numbers and changing them is a separate, validated decision.
 */
export function multiGpuCaveat(fit: GpuFit | null): string | null {
  if (!fit || !fit.fits || fit.gpuCount === 1) return null;
  if (fit.gpu.nvlinkGbps > 0) return null;
  return `${fit.gpu.name} has no NVLink, so ${fit.gpuCount}-way tensor parallelism runs over PCIe. tokcalc applies a flat 0.85× multi-GPU efficiency factor for this, which is realistic for NVLink-equipped datacenter cards but optimistic for PCIe scale-out — treat the throughput and cost figures for this configuration as an upper bound, and measure before committing.`;
}


/** Human label for a GPU, e.g. "H100 SXM5 80GB". */
export function gpuLabel(gpuId: string): string {
  return GPU_MAP[gpuId]?.name ?? gpuId;
}

/** All quantizations available for the model pages' comparison tables. */
export const ALL_QUANTS: QuantMeta[] = QUANTIZATIONS;
