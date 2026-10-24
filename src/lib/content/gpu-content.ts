/**
 * Programmatic content engine for per-GPU pages.
 *
 * Same contract as model-content.ts: every figure is computed from
 * token-calc.ts, never hand-typed. A page that says "$2.50/hr" says it
 * because `GPU_MAP["h100-sxm"].usdPerHour` says so.
 */

import {
  calculate,
  fmtTokens,
  fmtMoney,
  GPUS,
  GPU_MAP,
  MODELS,
  MODEL_MAP,
  type CalcResult,
  type GpuSpec,
  type ModelSpec,
} from "@/lib/token-calc";
import {
  MODEL_CONTENT_MAP,
  REF_BATCH,
  REF_CONTEXT,
  defaultQuantFor,
} from "./model-content";

/**
 * Reference model used for the cross-GPU comparison chart.
 *
 * Llama 3.3 70B at FP8 is the workload most buyers of datacenter GPUs
 * actually run, and it is the same model the hand-written H100-vs-H200 page
 * uses — so the numbers line up across the site.
 */
export const GPU_REFERENCE_MODEL = "llama3-3-70b";
export const GPU_REFERENCE_QUANT = "fp8";

export interface GpuModelFit {
  model: ModelSpec;
  quant: string;
  /** GPU count needed for this model on this GPU at the reference workload. */
  gpuCount: number;
  fits: boolean;
  result: CalcResult;
  costPerMillion: number;
  aggregateTokensPerSec: number;
}

export interface GpuContent {
  gpu: GpuSpec;
  /** Models that fit on a single one of this GPU, cheapest-first. */
  singleGpuModels: GpuModelFit[];
  /** Reference-model throughput on this GPU (1× if it fits, else 8×). */
  reference: {
    model: ModelSpec;
    quant: string;
    gpuCount: number;
    fits: boolean;
    result: CalcResult;
  };
  /** Every other GPU's reference-model throughput, for the comparison chart. */
  comparison: { gpu: GpuSpec; aggregateTokensPerSec: number; costPerMillion: number; fits: boolean }[];
  faqs: { q: string; a: string }[];
}

function calcOn(
  modelId: string,
  gpuId: string,
  quant: string,
  numGpus: number,
): CalcResult {
  return calculate({
    modelId,
    gpuId,
    quantization: quant as never,
    numGpus,
    batchSize: REF_BATCH,
    promptTokens: REF_CONTEXT,
    outputTokens: 500,
    useContinuousBatching: true,
  });
}

/** Smallest GPU count (1/2/4/8) that fits the model on this GPU. */
function fitCount(modelId: string, gpuId: string, quant: string): number | null {
  for (const n of [1, 2, 4, 8]) {
    if (calcOn(modelId, gpuId, quant, n).vramFits) return n;
  }
  return null;
}

export function computeGpuContent(gpuId: string): GpuContent | null {
  const gpu = GPU_MAP[gpuId];
  if (!gpu) return null;

  // ---- Models that fit on a single GPU of this type --------------------
  const singleGpuModels: GpuModelFit[] = [];
  for (const model of MODELS) {
    const quant = defaultQuantFor(model);
    const n = fitCount(model.id, gpu.id, quant);
    if (n !== 1) continue;
    const result = calcOn(model.id, gpu.id, quant, 1);
    singleGpuModels.push({
      model,
      quant,
      gpuCount: 1,
      fits: true,
      result,
      costPerMillion: result.costPerMillionOutputTokens,
      aggregateTokensPerSec: result.aggregateTokensPerSec,
    });
  }
  singleGpuModels.sort((a, b) => a.model.paramsB - b.model.paramsB);

  // ---- Reference model on this GPU --------------------------------------
  const refModel = MODEL_MAP[GPU_REFERENCE_MODEL];
  const refGpuCount = fitCount(GPU_REFERENCE_MODEL, gpu.id, GPU_REFERENCE_QUANT) ?? 8;
  const refResult = calcOn(GPU_REFERENCE_MODEL, gpu.id, GPU_REFERENCE_QUANT, refGpuCount);

  // ---- Cross-GPU comparison on the reference model -----------------------
  const comparison = GPUS.map((other) => {
    const n = fitCount(GPU_REFERENCE_MODEL, other.id, GPU_REFERENCE_QUANT) ?? 8;
    const r = calcOn(GPU_REFERENCE_MODEL, other.id, GPU_REFERENCE_QUANT, n);
    return {
      gpu: other,
      aggregateTokensPerSec: r.aggregateTokensPerSec,
      costPerMillion: r.costPerMillionOutputTokens,
      fits: r.vramFits,
    };
  })
    .filter((c) => c.fits)
    .sort((a, b) => b.aggregateTokensPerSec - a.aggregateTokensPerSec);

  const faqs = buildGpuFaqs(gpu, singleGpuModels, refModel, refGpuCount, refResult);

  return {
    gpu,
    singleGpuModels,
    reference: {
      model: refModel,
      quant: GPU_REFERENCE_QUANT,
      gpuCount: refGpuCount,
      fits: refResult.vramFits,
      result: refResult,
    },
    comparison,
    faqs,
  };
}

function buildGpuFaqs(
  gpu: GpuSpec,
  singleGpuModels: GpuModelFit[],
  refModel: ModelSpec,
  refGpuCount: number,
  refResult: CalcResult,
): { q: string; a: string }[] {
  const faqs: { q: string; a: string }[] = [];

  faqs.push({
    q: `What is ${gpu.name} used for?`,
    a: `${gpu.name} is a ${gpu.category} accelerator from ${gpu.vendor} with ${gpu.vramGb} GB of VRAM and ${gpu.memBandwidthGbps.toLocaleString()} GB/s of memory bandwidth${
      gpu.nvlinkGbps ? ` and ${gpu.nvlinkGbps} GB/s of NVLink` : ""
    }${gpu.usdPerHour ? `, typically around $${gpu.usdPerHour}/hr on cloud rental` : ""}.${
      gpu.note ? ` ${gpu.note}.` : ""
    } It is aimed at ${gpu.category === "datacenter" ? "production LLM serving" : gpu.category === "workstation" ? "on-prem and workstation inference" : gpu.category === "consumer" ? "local inference and development" : "specialised inference workloads"}.`,
  });

  if (singleGpuModels.length > 0) {
    const biggest = singleGpuModels[singleGpuModels.length - 1];
    faqs.push({
      q: `Which models fit on a single ${gpu.name}?`,
      a: `${singleGpuModels.length} models in the tokcalc catalog fit on one ${gpu.name}, up to ${biggest.model.name} (${biggest.model.paramsB.toFixed(0)}B params) at ${biggest.quant}. Below that, throughput and cost per million tokens improve as the model gets smaller.`,
    });
  } else {
    faqs.push({
      q: `Which models fit on a single ${gpu.name}?`,
      a: `No model in the tokcalc catalog fits on a single ${gpu.name} at its recommended quantization — even the smallest models here need more than ${gpu.vramGb} GB once weights and KV cache are loaded, or they only fit at a much lower precision than recommended.`,
    });
  }

  if (refResult.vramFits) {
    faqs.push({
      q: `How fast is ${refModel.name} on ${gpu.name}?`,
      a: `At ${GPU_REFERENCE_QUANT} with ${REF_BATCH} concurrent users, ${refGpuCount}× ${gpu.name} serves ${fmtTokens(refResult.aggregateTokensPerSec)} tokens/sec aggregate (${fmtTokens(refResult.decodeTokensPerSec)} tok/s per stream), with a ${fmtMsSafe(refResult.ttftMs)} time to first token. That works out to ${fmtMoney(refResult.costPerMillionOutputTokens)} per 1M output tokens.`,
    });
  }

  if (gpu.usdPerHour != null) {
    faqs.push({
      q: `How much does ${gpu.name} cost per hour?`,
      a: `Around $${gpu.usdPerHour}/hr for on-demand cloud rental (a typical 2025–2026 estimate across RunPod, Lambda, Modal, AWS, GCP and similar providers). Actual pricing varies by region, commitment, and spot availability — the Build vs Buy tab in the calculator pulls live quotes from several providers.`,
    });
  }

  faqs.push({
    q: `How does ${gpu.name} compare to other accelerators?`,
    a: `Memory bandwidth drives decode speed and VRAM drives capacity. ${gpu.name} has ${gpu.memBandwidthGbps.toLocaleString()} GB/s of bandwidth and ${gpu.vramGb} GB of VRAM${gpu.flopsTflops ? `, plus ${gpu.flopsTflops} TFLOPS of dense FP16/BF16 compute` : ""}. The comparison table on this page ranks every accelerator in the catalog on the same reference workload so you can see where it lands.`,
  });

  return faqs;
}

/** Local helper — mirrors fmtMs without importing it twice. */
function fmtMsSafe(ms: number): string {
  if (!isFinite(ms)) return "—";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** Every GPU id that has a page. */
export const GPU_PAGE_IDS = GPUS.map((g) => g.id);

export const GPU_CONTENT_MAP: Record<string, GpuContent> = Object.fromEntries(
  GPU_PAGE_IDS.map((id) => [id, computeGpuContent(id)!]),
);
