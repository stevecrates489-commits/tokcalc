/**
 * tokcalc MCP Server — server factory + tool registrations.
 *
 * Defines the 7 read-only planning tools and a factory function
 * that wires them into a fresh Server instance.
 *
 * Used by both the stdio entry point (index.ts) and the HTTP
 * entry point (http.ts) — each creates its own server instance
 * via createMcpServer() to avoid shared transport state.
 *
 * Per v0.2.0 research (Perplexity brief, 2025-09-25):
 *   "Do not connect the same McpServer instance simultaneously
 *    to stdio and multiple HTTP transports. Instead, use a factory."
 *
 * Tools:
 *   1. estimate_capacity — VRAM/KV/throughput/latency/cost for one config
 *   2. compare_gpus — ranked GPU comparison for one workload
 *   3. recommend_topology — TP/CP topology recommendation
 *   4. estimate_api_vs_self_host — break-even analysis
 *   5. list_models — discover supported model IDs
 *   6. list_gpus — discover supported GPU IDs
 *   7. get_mlperf_benchmarks — curated MLPerf v4.1 reference configs
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// Import tokcalc's calculation engine + catalog (shared with the web app)
import {
  calculate,
  fmtTokens,
  fmtBytes,
  fmtMs,
  fmtMoney,
  GPUS,
  MODELS,
  QUANT_MAP,
  GPU_MAP,
  MODEL_MAP,
  computeMaxConcurrency,
  computeKVCacheGb,
  recommendTopology,
  fmtContext,
  type Quantization,
} from "../../src/lib/token-calc";

// Import MLPerf curated reference configs (shared with the web app)
import { MLPERF_CURATED } from "../../src/lib/mlperf-curated";

// Helper function to format Zod schema for MCP protocol compliance.
// Uses Zod 4's native z.toJSONSchema() instead of zod-to-json-schema@3.x,
// which can't parse Zod 4 ASTs and silently returns empty `{}` schemas.
function formatInputSchema(schema: z.ZodTypeAny) {
  const json = z.toJSONSchema(schema) as Record<string, any>;
  delete json["$schema"];
  // Strip defaults from required array — clients shouldn't be required to
  // send fields that already have a default.
  if (Array.isArray(json.required) && json.properties) {
    json.required = json.required.filter((field: string) => {
      const prop = json.properties[field];
      return prop && prop.default === undefined;
    });
    if (json.required.length === 0) delete json.required;
  }
  return json;
}

// ============================================================
// TOOL SCHEMAS
// ============================================================

const ModelIdSchema = z.string().describe("Canonical tokcalc model ID (e.g. 'llama3-8b'). Call list_models first if unknown.");
const GpuIdSchema = z.string().describe("Canonical tokcalc GPU ID (e.g. 'h100-sxm'). Call list_gpus first if unknown.");
const QuantSchema = z.enum(["fp32","fp16","bf16","int8","int4","gguf-q2k","gguf-q3km","gguf-q4km","gguf-q5km","gguf-q6k","gguf-q8","gptq4","awq4","exl2-6bpw","fp8","nvfp4"]);

const EstimateCapacitySchema = z.object({
  model: ModelIdSchema,
  gpu: GpuIdSchema,
  quantization: QuantSchema.default("fp16"),
  gpuCount: z.number().int().min(1).max(256).default(1),
  batchSize: z.number().int().min(1).max(10000).default(1),
  promptTokens: z.number().int().min(1).max(2000000).default(500),
  outputTokens: z.number().int().min(1).max(200000).default(200),
  engine: z.enum(["generic","vllm","sglang","trtllm","llamacpp"]).default("generic"),
  continuousBatching: z.boolean().default(false),
  continuousBatchingMultiplier: z.number().min(1).max(10).default(1.5),
  reasoningTokens: z.number().int().min(0).max(1000000).default(0),
  contextTokens: z.number().int().min(1).max(10000000).optional()
    .describe("Full context length for KV cache computation (e.g., 8192 for RAG). If omitted, KV is computed for promptTokens only. Must not exceed the model's maxContext — call list_models to check."),
}).strict();

const CompareGpusSchema = z.object({
  model: ModelIdSchema,
  quantization: QuantSchema.default("fp16"),
  gpuCount: z.number().int().min(1).max(8).default(1),
  batchSize: z.number().int().min(1).max(1000).default(8),
  promptTokens: z.number().int().min(1).max(2000000).default(500),
  outputTokens: z.number().int().min(1).max(200000).default(200),
  sortBy: z.enum(["lowest_cost","highest_throughput","best_value"]).default("best_value"),
  limit: z.number().int().min(1).max(30).default(10),
  gpus: z.array(z.string()).optional().describe("Restrict comparison to specific GPU IDs (e.g. ['h100-sxm','h200-sxm']). Use list_gpus first to find IDs."),
}).strict();

const RecommendTopologySchema = z.object({
  model: ModelIdSchema,
  quantization: QuantSchema.default("fp16"),
  contextTokens: z.number().int().min(1).max(10000000).default(8192)
    .describe("Context length to size the KV cache for. Must not exceed the model's maxContext — call list_models to check."),
  batchSize: z.number().int().min(1).max(1000).default(1),
}).strict();

const EstimateApiVsSelfHostSchema = z.object({
  model: ModelIdSchema,
  gpu: GpuIdSchema,
  quantization: QuantSchema.default("fp16"),
  gpuCount: z.number().int().min(1).max(64).default(1),
  inputTokens: z.number().int().min(1).max(2000000).default(500),
  outputTokens: z.number().int().min(1).max(200000).default(200),
  requestsPerDay: z.number().int().min(1).max(100000000).default(1000),
  utilization: z.number().min(0.05).max(1).default(0.5),
  apiModel: z.string().default("gpt-4o-mini"),
  apiInputPrice: z.number().min(0).default(0.15),
  apiOutputPrice: z.number().min(0).default(0.60),
}).strict();

const ListModelsSchema = z.object({
  family: z.string().optional().describe("Filter by model family (e.g. 'Llama', 'Qwen')"),
  category: z.enum(["text","vlm","embedding","code","reasoning"]).optional(),
  isMoE: z.boolean().optional(),
}).strict();

const ListGpusSchema = z.object({
  vendor: z.string().optional().describe("Filter by vendor (e.g. 'NVIDIA', 'AMD')"),
  category: z.enum(["datacenter","workstation","consumer","mac","tpu","lpu","wse","legacy"]).optional(),
  minVramGb: z.number().optional(),
}).strict();

const GetMlperfBenchmarksSchema = z.object({
  gpuModel: z.string().optional().describe("Filter by GPU model substring (e.g. 'H100', 'H200', 'A100')"),
  workload: z.string().optional().describe("Filter by model ID substring (e.g. 'llama3-70b', 'llama3-8b')"),
  scenario: z.enum(["Offline", "Server"]).optional().describe("Filter by MLPerf scenario"),
}).strict();

// ============================================================
// TOOL HANDLERS
// ============================================================

function handleEstimateCapacity(input: z.infer<typeof EstimateCapacitySchema>) {
  // 1. Calculate baseline metrics using promptTokens (preserves correct prefill and TTFT)
  const result = calculate({
    modelId: input.model,
    gpuId: input.gpu,
    quantization: input.quantization as Quantization,
    numGpus: input.gpuCount,
    batchSize: input.batchSize,
    promptTokens: input.promptTokens,
    outputTokens: input.outputTokens,
    engineId: input.engine,
    effMem: undefined,
    useContinuousBatching: input.continuousBatching,
    continuousBatchingMultiplier: input.continuousBatchingMultiplier,
    reasoningTokens: input.reasoningTokens,
  });

  // 2. Override KV cache and VRAM if explicit contextTokens provided.
  // calculate() uses promptTokens for both prefill and KV — for RAG/long-context
  // scenarios, callers know the full context length (prompt + retrieved docs).
  // We selectively recompute KV and VRAM without touching prefill-derived TTFT.
  if (input.contextTokens && input.contextTokens > input.promptTokens) {
    const ctxModel = MODEL_MAP[input.model];
    const ctxGpu = GPU_MAP[input.gpu];
    if (ctxModel) {
      if (input.contextTokens > ctxModel.maxContext) {
        return {
          error: `contextTokens ${input.contextTokens.toLocaleString("en-US")} exceeds ${ctxModel.name}'s maxContext of ${ctxModel.maxContext.toLocaleString("en-US")}.`,
          modelMaxContext: ctxModel.maxContext,
          requestedContext: input.contextTokens,
          suggestion: `Clamp to the model's real ceiling. Advertising a larger window does not make it servable — KV cache grows linearly with context and would exhaust VRAM.`,
        };
      }
      const kvAtContext = computeKVCacheGb(ctxModel, input.contextTokens, input.batchSize);
      const totalNeeded = result.modelSizeGb + kvAtContext;
      const availableGb = (ctxGpu?.vramGb ?? 0) * input.gpuCount;
      (result as any).kvCacheTotalGb = kvAtContext;
      (result as any).totalVramNeededGb = totalNeeded;
      (result as any).vramFits = totalNeeded <= availableGb;
    }
  }

  const model = MODEL_MAP[input.model];
  const gpu = GPU_MAP[input.gpu];

  return {
    summary: `${model?.name || input.model} on ${gpu?.name || input.gpu} (${input.quantization}): ${fmtTokens(result.decodeTokensPerSec)} tok/s decode, ${fmtMs(result.ttftMs)} TTFT, ${fmtBytes(result.totalVramNeededGb)} VRAM needed, ${fmtMoney(result.costPerMillionOutputTokens)}/M tokens`,
    feasibility: {
      modelFits: result.vramFits,
      fitsWithKvCache: result.vramFits,
      blockingReasons: result.vramFits ? [] : [`Needs ${fmtBytes(result.totalVramNeededGb)} but only ${gpu?.vramGb * input.gpuCount} GB available`],
      warnings: result.longContextWarning ? [result.longContextWarning] : [],
    },
    performance: {
      outputTokensPerSecond: {
        low: Math.round(result.decodeTokensPerSec * 0.7),
        expected: Math.round(result.decodeTokensPerSec),
        high: Math.round(result.decodeTokensPerSec * 1.3),
      },
      aggregateTokensPerSecond: Math.round(result.aggregateTokensPerSec),
      ttftMs: { expected: Math.round(result.ttftMs) },
      itlMs: { expected: Math.round(result.itlMs) },
      totalLatencyMs: Math.round(result.totalLatencyMs),
    },
    memory: {
      modelWeightsGb: +result.modelSizeGb.toFixed(2),
      activeParamsGb: +(result as any).activeParamsGb?.toFixed?.(2) ?? null,
      isMoE: (result as any).isMoE ?? model?.isMoE ?? false,
      weightsNote: (result as any).isMoE ?? model?.isMoE
        ? "MoE: modelWeightsGb is ALL experts resident in VRAM. activeParamsGb is the per-token working set used for the bandwidth math. They differ by ~10x — use modelWeightsGb for capacity, activeParamsGb for speed."
        : "Dense model: modelWeightsGb equals activeParamsGb.",
      kvCacheGb: +result.kvCacheTotalGb.toFixed(2),
      totalRequiredGb: +result.totalVramNeededGb.toFixed(2),
      availableGb: gpu ? gpu.vramGb * input.gpuCount : 0,
      utilizationPct: +((result.totalVramNeededGb / ((gpu?.vramGb || 1) * input.gpuCount)) * 100).toFixed(1),
    },
    cost: {
      gpuHourlyUsd: result.costPerHour,
      costPerMillionTokens: result.costPerMillionOutputTokens,
      costPerRequest: result.costPerRequest,
    },
    confidence: {
      throughput: result.confidence.decodeTokensPerSec,
      latency: result.confidence.totalLatencyMs,
      memory: result.confidence.totalVramNeededGb,
    },
    assumptions: [
      `η_mem = 0.65 (typical real-world memory utilization)`,
      `η_compute = 0.50 (typical compute utilization)`,
      `KV cache in FP16 (2 bytes per value)`,
      `Engine: ${input.engine} (affects efficiency factors)`,
      `Continuous batching: ${input.continuousBatching ? `${input.continuousBatchingMultiplier}× multiplier` : "disabled"}`,
      `These are planning estimates, not deployment guarantees`,
    ],
    catalogVersion: "0.3.0",
  };
}

function handleCompareGpus(input: z.infer<typeof CompareGpusSchema>) {
  const model = MODEL_MAP[input.model];
  const quant = QUANT_MAP[input.quantization as Quantization];
  if (!model || !quant) return { error: "Unknown model or quantization", models: Object.keys(MODEL_MAP).slice(0, 10) };

  // Apply explicit GPU filter if provided, else all GPUs
  let candidatePool = GPUS;
  if (input.gpus && input.gpus.length > 0) {
    const requested = new Set(input.gpus);
    candidatePool = GPUS.filter(g => requested.has(g.id));
    if (candidatePool.length === 0) {
      return {
        error: `No GPUs matched gpus=[${input.gpus.join(", ")}]. Call list_gpus to find valid IDs.`,
        validGpuIds: GPUS.slice(0, 10).map(g => g.id),
      };
    }
  }

  // Filter: must have a known price AND meet memory requirement
  const skippedNoPrice: string[] = [];
  const candidates = candidatePool.filter(g => {
    const fitsMemory = g.vramGb * input.gpuCount >= model.activeParamsB * quant.bytesPerParam;
    const hasPrice = g.usdPerHour !== null && g.usdPerHour !== undefined && g.usdPerHour > 0;
    if (!hasPrice) skippedNoPrice.push(g.id);
    return fitsMemory && hasPrice;
  });

  const results = candidates.map(g => {
    const r = calculate({
      modelId: input.model,
      gpuId: g.id,
      quantization: input.quantization as Quantization,
      numGpus: input.gpuCount,
      batchSize: input.batchSize,
      promptTokens: input.promptTokens,
      outputTokens: input.outputTokens,
    });
    return {
      gpuId: g.id,
      gpuName: g.name,
      vendor: g.vendor,
      vramGb: g.vramGb * input.gpuCount,
      modelFits: r.vramFits,
      outputTokensPerSecond: Math.round(r.decodeTokensPerSec),
      aggregateTokensPerSecond: Math.round(r.aggregateTokensPerSec),
      ttftMs: Math.round(r.ttftMs),
      costPerMillionTokens: +r.costPerMillionOutputTokens.toFixed(2),
      gpuHourlyUsd: r.costPerHour,
      confidence: r.confidence.decodeTokensPerSec,
    };
  });

  const sorted = results.sort((a, b) => {
    if (input.sortBy === "lowest_cost") return a.costPerMillionTokens - b.costPerMillionTokens;
    if (input.sortBy === "highest_throughput") return b.aggregateTokensPerSecond - a.aggregateTokensPerSecond;
    return (b.aggregateTokensPerSecond / Math.max(b.costPerMillionTokens, 0.001)) - (a.aggregateTokensPerSecond / Math.max(a.costPerMillionTokens, 0.001));
  }).slice(0, input.limit);

  // Describe the winner according to the requested ordering. The previous
  // wording always said "Cheapest" while rows were ordered by sortBy, so a
  // highest_throughput run opened with a sentence that contradicted its own
  // table (e.g. "Cheapest: B200 at $0.64/M" when MI300X was cheaper at $0.40/M).
  const best = sorted[0];
  const cheapestByCost = results.length
    ? results.reduce((acc, r) => (r.costPerMillionTokens < acc.costPerMillionTokens ? r : acc))
    : undefined;
  const fastest = results.length
    ? results.reduce((acc, r) => (r.aggregateTokensPerSecond > acc.aggregateTokensPerSecond ? r : acc))
    : undefined;
  const bestValue = results.length
    ? results.reduce((acc, r) => {
        const score = (x: typeof r) =>
          x.aggregateTokensPerSecond / Math.max(x.costPerMillionTokens, 0.001);
        return score(r) > score(acc) ? r : acc;
      })
    : undefined;

  const lead =
    input.sortBy === "lowest_cost"
      ? `Cheapest: ${cheapestByCost?.gpuName} at $${cheapestByCost?.costPerMillionTokens.toLocaleString("en-US")}/M tokens`
      : input.sortBy === "highest_throughput"
        ? `Fastest: ${fastest?.gpuName} at ${fastest?.aggregateTokensPerSecond.toLocaleString("en-US")} aggregate tok/s ($/${fastest?.costPerMillionTokens.toLocaleString("en-US")}/M)`
        : `Best value: ${bestValue?.gpuName} — ${bestValue?.aggregateTokensPerSecond.toLocaleString("en-US")} tok/s at $${bestValue?.costPerMillionTokens.toLocaleString("en-US")}/M`;

  return {
    summary:
      `Compared ${results.length} GPU${results.length === 1 ? "" : "s"} for ${model.name} (${quant.label}, ` +
      `${input.gpuCount}× ${input.gpuCount > 1 ? "GPUs" : "GPU"}, batch ${input.batchSize}). ` +
      `${lead}. Top ${input.sortBy === "lowest_cost" ? "cheapest" : input.sortBy === "highest_throughput" ? "fastest" : "value"} shown first.`,
    comparisons: sorted,
    totalCandidates: results.length,
    sortBy: input.sortBy,
    leaders: {
      cheapestByCost: cheapestByCost
        ? { gpuId: cheapestByCost.gpuId, gpuName: cheapestByCost.gpuName, costPerMillionTokens: cheapestByCost.costPerMillionTokens }
        : undefined,
      fastest: fastest
        ? { gpuId: fastest.gpuId, gpuName: fastest.gpuName, aggregateTokensPerSecond: fastest.aggregateTokensPerSecond }
        : undefined,
      bestValue: bestValue
        ? { gpuId: bestValue.gpuId, gpuName: bestValue.gpuName, aggregateTokensPerSecond: bestValue.aggregateTokensPerSecond, costPerMillionTokens: bestValue.costPerMillionTokens }
        : undefined,
    },
    firstRow: best && {
      gpuId: best.gpuId,
      gpuName: best.gpuName,
      rankBasis: input.sortBy,
      note: "First row reflects sortBy; it is only the cheapest when sortBy='lowest_cost'.",
    },
    skippedNoPrice: skippedNoPrice.length > 0 ? skippedNoPrice : undefined,
    assumptions: [`Region: us-east-1 (default)`, `On-demand pricing`, `η_mem = 0.65`, `Catalog version: 0.3.0`],
  };
}

function handleRecommendTopology(input: z.infer<typeof RecommendTopologySchema>) {
  const model = MODEL_MAP[input.model];
  const quant = QUANT_MAP[input.quantization as Quantization];
  if (!model || !quant) return { error: "Unknown model or quantization" };

  // Per-batch KV cache. All batchSize requests hold KV simultaneously in a
  // static batch, so the fit check must account for batchSize × kvPerRequest.
  const kvPerRequest = computeKVCacheGb(model, input.contextTokens, 1);
  const kvPerBatch = computeKVCacheGb(model, input.contextTokens, input.batchSize);
  const residentWeightsGb = model.paramsB * quant.bytesPerParam;
  const results = GPUS.map(g => {
    const rec = recommendTopology(model, g, input.contextTokens, input.batchSize, quant.bytesPerParam);
    // computeMaxConcurrency's 3rd arg is numGpus (GPU COUNT), not batchSize.
    // Use the topology's actual GPU count so the number reconciles with fits.
    const maxConcurrent = rec.neededGpus > 0
      ? computeMaxConcurrency(model, g, rec.neededGpus, input.contextTokens, quant.bytesPerParam)
      : 0;
    const freeGb = g.vramGb * rec.neededGpus - residentWeightsGb;
    return {
      gpuId: g.id,
      gpuName: g.name,
      vramGb: g.vramGb,
      topology: rec.topology,
      neededGpus: rec.neededGpus,
      totalVramGb: +(g.vramGb * rec.neededGpus).toFixed(0),
      vramNeededGb: +(residentWeightsGb + kvPerBatch).toFixed(1),
      residentWeightsGb: +residentWeightsGb.toFixed(1),
      freeAfterWeightsGb: +freeGb.toFixed(1),
      fits: rec.fits,
      maxConcurrentUsers: maxConcurrent,
      // The exact division behind maxConcurrentUsers, so the figure can be
      // checked by hand instead of taken on faith. No hidden reserve is applied.
      maxConcurrentUsersMath: kvPerRequest > 0
        ? `floor((${g.vramGb} GB × ${rec.neededGpus} − ${residentWeightsGb.toFixed(1)} GB weights) ÷ ${kvPerRequest.toFixed(2)} GB/req) = ${maxConcurrent}`
        : "n/a",
      maxConcurrentBatchesAtContext: Math.floor(maxConcurrent / Math.max(input.batchSize, 1)),
      kvPerRequestGb: +kvPerRequest.toFixed(2),
      kvPerBatchGb: +kvPerBatch.toFixed(2),
      reason: rec.reason,
    };
  }).filter(r => r.fits).slice(0, 5);

  return {
    summary: `For ${model.name} at ${fmtContext(input.contextTokens)} context (batch=${input.batchSize}): ${results.length} feasible topology options across ${GPUS.length} GPUs.`,
    model: { name: model.name, paramsB: model.paramsB, activeParamsB: model.activeParamsB, isMoE: model.isMoE },
    context: { tokens: input.contextTokens, label: fmtContext(input.contextTokens), batchSize: input.batchSize },
    recommendations: results,
    formula: `KV per request = 2 × ${model.layers} layers × ${model.kvHeads} KV heads × ${model.headDim} head_dim × 2 bytes × ${input.contextTokens} tokens = ${kvPerRequest.toFixed(2)} GB; × ${input.batchSize} batch = ${kvPerBatch.toFixed(2)} GB`,
    assumptions: [
      `Single GPU unless TP needed`,
      `KV cache in FP16`,
      `Model weights (${residentWeightsGb.toFixed(1)} GB, full paramsB${model.isMoE ? " — all MoE experts resident" : ""}) + KV × batchSize must fit in total VRAM`,
      `maxConcurrentUsers is concurrent single-slot requests at this context; concurrent batches = maxConcurrentUsers ÷ ${input.batchSize}`,
      `maxConcurrentUsers is an upper bound: it excludes runtime overhead (activations, CUDA context, allocator slack)`,
      `Catalog version: 0.3.0`,
    ],
  };
}

function handleEstimateApiVsSelfHost(input: z.infer<typeof EstimateApiVsSelfHostSchema>) {
  // `utilization` is a FRACTION in [0.05, 1] (0.5 = 50%), not a percentage.
  const utilizationPct = input.utilization * 100;
  const sh = calculate({
    modelId: input.model,
    gpuId: input.gpu,
    quantization: input.quantization as Quantization,
    numGpus: input.gpuCount,
    batchSize: 8,
    promptTokens: input.inputTokens,
    outputTokens: input.outputTokens,
  });

  const gpu = GPU_MAP[input.gpu];
  const model = MODEL_MAP[input.model];
  if (!sh.vramFits) {
    return {
      error: `Self-host infeasible: ${model?.name ?? input.model} at ${input.quantization} on ${input.gpuCount}× ${gpu?.name ?? input.gpu} needs ${sh.totalVramNeededGb.toFixed(1)} GB but only ${((gpu?.vramGb ?? 0) * input.gpuCount).toFixed(0)} GB is available.`,
      suggestion: `Increase gpuCount, lower the quantization (e.g. fp8/int4), or pick a smaller model. Break-even is undefined until the config fits.`,
      feasibility: { fits: false, vramNeededGb: +sh.totalVramNeededGb.toFixed(2), vramAvailableGb: +((gpu?.vramGb ?? 0) * input.gpuCount).toFixed(0) },
    };
  }

  const effGpuPrice = (gpu?.usdPerHour ?? 0) * input.gpuCount;
  // utilization is already a fraction — do NOT divide by 100 again.
  const effTokens = sh.aggregateTokensPerSec * input.utilization;
  const selfHostCostPerM = effTokens > 0 ? (effGpuPrice / 3600 / effTokens) * 1e6 : Infinity;
  const selfHostMonthly = effGpuPrice * 730;

  const apiCostPerRequest = (input.inputTokens / 1e6) * input.apiInputPrice + (input.outputTokens / 1e6) * input.apiOutputPrice;
  const apiMonthly = apiCostPerRequest * input.requestsPerDay * 30;
  const selfHostMonthlyTotal = selfHostMonthly;

  const breakEven = apiCostPerRequest > 0 ? selfHostMonthly / (30 * apiCostPerRequest) : Infinity;

  // Per-token comparisons must be like-for-like.
  //
  // `selfHostCostPerM` is a BLENDED figure: one $/M over all tokens the fleet
  // emits. The API side has separate input and output rates, so the blended API
  // cost is (inputTokens·in + outputTokens·out) / totalTokens per 1M — NOT
  // apiOutputPrice alone. Comparing blended self-host against output-only API
  // flattered self-hosting whenever input tokens are a large share of traffic.
  const tokensPerRequest = input.inputTokens + input.outputTokens;
  const apiBlendedCostPerM =
    tokensPerRequest > 0
      ? (apiCostPerRequest / tokensPerRequest) * 1e6
      : input.apiOutputPrice;

  const cheaper = selfHostCostPerM < apiBlendedCostPerM;
  const meetsVolume = input.requestsPerDay > breakEven;

  return {
    summary: cheaper && meetsVolume
      ? `Self-hosting is cheaper at ${input.requestsPerDay.toLocaleString("en-US")} req/day. ` +
        `$${selfHostCostPerM.toFixed(2)}/M blended (self-host) vs $${apiBlendedCostPerM.toFixed(2)}/M blended (${input.apiModel}).`
      : !meetsVolume
        ? `Not enough volume. Need ${Math.round(breakEven).toLocaleString("en-US")} req/day to break even (currently ${input.requestsPerDay.toLocaleString("en-US")}).`
        : `API is cheaper. $${selfHostCostPerM.toFixed(2)}/M blended (self-host) vs $${apiBlendedCostPerM.toFixed(2)}/M blended (${input.apiModel}).`,
    selfHost: {
      costPerMillionTokensBlended: +selfHostCostPerM.toFixed(2),
      costPerMillionTokens: +selfHostCostPerM.toFixed(2),
      blendedNote: "One rate across all emitted tokens (input + output).",
      monthlyInfraUsd: +selfHostMonthly.toFixed(2),
      monthlyTotalUsd: +selfHostMonthlyTotal.toFixed(2),
      utilization: `${utilizationPct.toLocaleString("en-US")}%`,
      utilizationNote: "Share of peak fleet throughput actually sold/utilized.",
      throughput: `${fmtTokens(sh.aggregateTokensPerSec)} tok/s`,
      effectiveThroughput: `${fmtTokens(effTokens)} tok/s at ${utilizationPct.toLocaleString("en-US")}% utilization`,
    },
    api: {
      costPerMillionTokensBlended: +apiBlendedCostPerM.toFixed(2),
      costPerMillionInputTokens: input.apiInputPrice,
      costPerMillionOutputTokens: input.apiOutputPrice,
      costPerMillionTokens: +apiBlendedCostPerM.toFixed(2),
      blendedNote:
        "Blended = (inputTokens × in + outputTokens × out) ÷ total tokens, scaled to 1M. " +
        "Compare against selfHost.costPerMillionTokensBlended, not the output-only rate.",
      monthlyUsd: +apiMonthly.toFixed(2),
      model: input.apiModel,
    },
    breakEven: {
      requestsPerDay: Math.round(breakEven),
      reached: meetsVolume,
      explanation:
        `Self-host fixed cost $${selfHostMonthly.toFixed(2)}/mo ÷ ` +
        `($${apiCostPerRequest.toFixed(5)}/request × 30 days) = ` +
        `${Math.round(breakEven).toLocaleString("en-US")} req/day. ` +
        `Assumes ${utilizationPct.toLocaleString("en-US")}% utilization on ${input.gpuCount}× ${gpu?.name || input.gpu}, ` +
        `${input.inputTokens.toLocaleString("en-US")} input + ${input.outputTokens.toLocaleString("en-US")} output tokens per request.`,
      requestsPerDayLabel: Math.round(breakEven).toLocaleString("en-US"),
    },
    assumptions: [
      `Self-host peak throughput: ${fmtTokens(sh.aggregateTokensPerSec)} tok/s; effective at ${utilizationPct.toLocaleString("en-US")}% utilization = ${fmtTokens(effTokens)} tok/s`,
      `GPU price: $${effGpuPrice}/hr`,
      `API pricing: $${input.apiInputPrice}/M input, $${input.apiOutputPrice}/M output → $${apiBlendedCostPerM.toFixed(2)}/M blended at this token mix`,
      `Break-even uses API cost only; it excludes self-host ops, power beyond the GPU-hour rate, and engineering time`,
      `730 hours/month`,
      `Catalog version: 0.3.0`,
    ],
  };
}

function handleListModels(input: z.infer<typeof ListModelsSchema>) {
  let filtered = MODELS;
  if (input.family) filtered = filtered.filter(m => m.family === input.family);
  if (input.category) filtered = filtered.filter(m => m.category === input.category);
  if (input.isMoE !== undefined) filtered = filtered.filter(m => m.isMoE === input.isMoE);

  return {
    count: filtered.length,
    models: filtered.map(m => ({
      id: m.id,
      name: m.name,
      family: m.family,
      category: m.category,
      paramsB: m.paramsB,
      activeParamsB: m.activeParamsB,
      isMoE: m.isMoE,
      layers: m.layers,
      maxContext: m.maxContext,
    })),
    catalogVersion: "0.3.0",
  };
}

function handleListGpus(input: z.infer<typeof ListGpusSchema>) {
  let filtered = GPUS;
  if (input.vendor) filtered = filtered.filter(g => g.vendor === input.vendor);
  if (input.category) filtered = filtered.filter(g => g.category === input.category);
  if (input.minVramGb !== undefined) {
    const minVram = input.minVramGb;
    filtered = filtered.filter(g => g.vramGb >= minVram);
  }

  return {
    count: filtered.length,
    gpus: filtered.map(g => ({
      id: g.id,
      name: g.name,
      vendor: g.vendor,
      category: g.category,
      memBandwidthGbps: g.memBandwidthGbps,
      flopsTflops: g.flopsTflops,
      vramGb: g.vramGb,
      nvlinkGbps: g.nvlinkGbps,
      usdPerHour: g.usdPerHour,
      year: g.year,
      note: g.note,
    })),
    catalogVersion: "0.3.0",
  };
}

function handleGetMlperfBenchmarks(input: z.infer<typeof GetMlperfBenchmarksSchema>) {
  let results = MLPERF_CURATED;
  if (input.gpuModel) {
    const q = input.gpuModel.toLowerCase();
    results = results.filter(r => r.accelerator_model?.toLowerCase().includes(q));
  }
  if (input.workload) {
    const q = input.workload.toLowerCase();
    results = results.filter(r => r.model_id.toLowerCase().includes(q));
  }
  if (input.scenario) {
    results = results.filter(r => r.comparability_group?.includes(input.scenario!));
  }
  return {
    count: results.length,
    benchmarks: results.map(r => ({
      system: r.instance_type,
      model: r.model_display_name,
      quantization: r.quantization_format,
      gpuCount: r.gpu_count,
      gpu: r.accelerator_model,
      scenario: r.comparability_group,
      throughput_tps: r.output_token_throughput_tps,
      ttft_ms: r.ttft_mean_ms,
      itl_ms: r.itl_mean_ms,
      confidence_tier: r.confidence_tier,
      verification: r.verification_status,
      citation: r.citation_text,
      source_url: r.source_url,
    })),
    note: "Throughput values are COMPUTED by tokcalc formulas (confidence_tier=derived). System configurations are sourced from MLPerf Inference v4.1 audited submissions.",
    catalogVersion: "0.3.0",
  };
}

// ============================================================
// TOOL DEFINITIONS
// ============================================================

const TOOL_DEFINITIONS = [
  {
    name: "estimate_capacity",
    description: "Estimate whether an LLM-serving configuration fits in memory and can meet throughput and latency targets. Returns VRAM/KV-cache breakdown, throughput and latency ranges, concurrency, cost, confidence, assumptions, and sources.",
    inputSchema: formatInputSchema(EstimateCapacitySchema),
  },
  {
    name: "compare_gpus",
    description: "Compare supported GPU or cloud SKU options for the same LLM workload. Use before recommending hardware.",
    inputSchema: formatInputSchema(CompareGpusSchema),
  },
  {
    name: "recommend_topology",
    description: "Recommend feasible GPU/topology designs including tensor parallelism and context parallel (RingAttention) when needed.",
    inputSchema: formatInputSchema(RecommendTopologySchema),
  },
  {
    name: "estimate_api_vs_self_host",
    description: "Compare monthly token-based API costs with self-hosted GPU infrastructure under stated utilization assumptions.",
    inputSchema: formatInputSchema(EstimateApiVsSelfHostSchema),
  },
  {
    name: "list_models",
    description: "List tokcalc-supported model IDs and metadata. Use before estimating if the requested model is ambiguous or unknown.",
    inputSchema: formatInputSchema(ListModelsSchema),
  },
  {
    name: "list_gpus",
    description: "List GPU and cloud SKU IDs, memory, bandwidth, pricing, and categories.",
    inputSchema: formatInputSchema(ListGpusSchema),
  },
  {
    name: "get_mlperf_benchmarks",
    description: "Retrieve curated MLPerf Inference v4.1 LLM benchmark reference configurations. Throughput is computed by tokcalc formulas (confidence_tier=derived). Use to cross-validate theoretical estimates against audited system configurations.",
    inputSchema: formatInputSchema(GetMlperfBenchmarksSchema),
  },
];

/**
 * Server build fingerprint.
 *
 * Bump this in lockstep with mini-services/mcp-server/package.json and expose
 * it in serverInfo (tools/list) AND in every tool response's `provenance`.
 *
 * Why: when an agent reports behavior that does not match the code you just
 * shipped, the only question that matters is "which build is this endpoint
 * actually serving?". Stamping every response makes that answerable in a single
 * call instead of five round-trips. It also caught serverInfo being hardcoded
 * to 0.2.0 across three releases.
 */
export const SERVER_VERSION = "0.2.4";

/** Stamped into every tool result so callers can detect stale deployments. */
function withProvenance<T extends Record<string, unknown>>(result: T): T & {
  provenance: { serverVersion: string; catalogVersion: string; buildStamp: string };
} {
  return {
    ...result,
    provenance: {
      serverVersion: SERVER_VERSION,
      catalogVersion: "0.3.0",
      // Machine-greppable token: a single `grep fd5ba32` style diff tells you
      // whether the running deployment includes a given fix.
      buildStamp: `tokcalc-mcp/${SERVER_VERSION}`,
    },
  };
}

/** The schema backing each tool, so we can echo valid names on a bad call. */
const TOOL_SCHEMAS: Record<string, z.ZodType> = {
  estimate_capacity: EstimateCapacitySchema,
  compare_gpus: CompareGpusSchema,
  recommend_topology: RecommendTopologySchema,
  estimate_api_vs_self_host: EstimateApiVsSelfHostSchema,
  list_models: ListModelsSchema,
  list_gpus: ListGpusSchema,
  get_mlperf_benchmarks: GetMlperfBenchmarksSchema,
};

/**
 * Turn a thrown validation error into something a caller can act on.
 *
 * The important case is an unrecognized key. These schemas declare
 * `additionalProperties: false` in their exported JSON Schema, but Zod's
 * default object behaviour is to *strip* unknown keys silently — so a caller
 * inventing a plausible name (`tokensPerMonth`, `apiInputPricePerMillion`)
 * previously got a confident answer computed from defaults with their input
 * quietly discarded. That is the worst possible failure mode: a number that
 * looks authoritative and is wrong.
 *
 * With `.strict()` those calls now throw, and we answer with the exact list of
 * accepted names plus a nearest-match suggestion, so the caller self-corrects
 * in one turn.
 */
function describeCallFailure(
  toolName: string,
  error: unknown,
): Record<string, unknown> {
  const base: Record<string, unknown> = {
    error: error instanceof Error ? error.message : String(error),
    tool: toolName,
    provenance: {
      serverVersion: SERVER_VERSION,
      buildStamp: `tokcalc-mcp/${SERVER_VERSION}`,
    },
  };

  const issues =
    error && typeof error === "object" && "issues" in error
      ? ((error as { issues: unknown }).issues as Array<{ code?: string; path?: unknown[] }>)
      : null;

  if (!issues || issues.length === 0) return base;

  const schema = TOOL_SCHEMAS[toolName];
  const validKeys = schema
    ? Object.keys((schema as unknown as { shape: Record<string, unknown> }).shape ?? {})
    : [];

  const unrecognized = issues.filter((i) => i.code === "unrecognized_keys");
  const outOfRange = issues.filter((i) => i.code === "invalid_value");
  const wrongType = issues.filter((i) => i.code === "invalid_type");

  if (unrecognized.length > 0 && validKeys.length > 0) {
    const bad = unrecognized.flatMap((i) => (Array.isArray(i.path) ? i.path : [])) as string[];

    // Cheap nearest-name suggestion: prefix/substring overlap beats nothing.
    const suggestions = bad.map((b) => {
      const lower = b.toLowerCase();
      const near = validKeys.find(
        (k) =>
          k.toLowerCase().includes(lower) ||
          lower.includes(k.toLowerCase()) ||
          k.toLowerCase().replace(/[^a-z]/g, "") === lower.replace(/[^a-z]/g, ""),
      );
      return near ? `${b} -> did you mean '${near}'?` : `${b} is not a parameter of ${toolName}`;
    });

    return {
      ...base,
      error: `Unrecognized parameter(s): ${bad.join(", ")}. These schemas are strict — unknown keys are rejected rather than ignored, so nothing was silently dropped.`,
      unrecognizedParameters: bad,
      suggestions,
      validParameters: validKeys,
      hint: `Call tools/list (or read the inputSchema) to see every accepted parameter and its default.`,
    };
  }

  if (wrongType.length > 0 || outOfRange.length > 0) {
    return {
      ...base,
      error: `Invalid value for ${toolName}.`,
      invalidParameters: issues
        .filter((i) => i.code === "invalid_type" || i.code === "invalid_value")
        .map((i) => (Array.isArray(i.path) ? i.path.join(".") : "?")),
      validParameters: validKeys,
      hint: `Check the type and range constraints declared in the inputSchema for ${toolName}.`,
    };
  }

  return base;
}

// ============================================================
// SERVER FACTORY
// ============================================================

/**
 * Create a fresh MCP Server instance with all 7 tools wired.
 *
 * Used by:
 *   - stdio entry point (index.ts) — one server per process
 *   - HTTP entry point (http.ts) — one server per request (stateless mode)
 *
 * Each caller creates its own instance via this factory to avoid
 * shared transport state (per v0.2.0 research recommendation #5:
 * "Do not connect the same McpServer instance to multiple transports").
 */
export function createMcpServer(): Server {
  const server = new Server(
    { name: "tokcalc", version: SERVER_VERSION },
    {
      capabilities: {
        tools: {},
      },
      instructions:
        `tokcalc MCP server ${SERVER_VERSION} — LLM serving capacity planner. ` +
        `Use list_models / list_gpus to discover canonical IDs before calling ` +
        `estimate_capacity, compare_gpus, recommend_topology, or ` +
        `estimate_api_vs_self_host. All estimates are planning projections, not ` +
        `deployment guarantees. Every response carries provenance.serverVersion.`,
    }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOL_DEFINITIONS,
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      let result: unknown;

      switch (name) {
        case "estimate_capacity": {
          const input = EstimateCapacitySchema.parse(args);
          result = handleEstimateCapacity(input);
          break;
        }
        case "compare_gpus": {
          const input = CompareGpusSchema.parse(args);
          result = handleCompareGpus(input);
          break;
        }
        case "recommend_topology": {
          const input = RecommendTopologySchema.parse(args);
          result = handleRecommendTopology(input);
          break;
        }
        case "estimate_api_vs_self_host": {
          const input = EstimateApiVsSelfHostSchema.parse(args);
          result = handleEstimateApiVsSelfHost(input);
          break;
        }
        case "list_models": {
          const input = ListModelsSchema.parse(args);
          result = handleListModels(input);
          break;
        }
        case "list_gpus": {
          const input = ListGpusSchema.parse(args);
          result = handleListGpus(input);
          break;
        }
        case "get_mlperf_benchmarks": {
          const input = GetMlperfBenchmarksSchema.parse(args);
          result = handleGetMlperfBenchmarks(input);
          break;
        }
        default:
          return {
            content: [{ type: "text", text: `Unknown tool: ${name}` }],
            isError: true,
          };
      }

      // Handlers signal recoverable problems (unknown ID, infeasible config,
      // no GPU match) by returning an `error` key rather than throwing.
      // Surface those as isError:true so agent clients can detect failure
      // programmatically instead of parsing prose.
      const isHandlerError =
        !!result && typeof result === "object" && "error" in (result as Record<string, unknown>);

      // Stamp the running build into every response. One glance tells the
      // caller whether the endpoint they're talking to includes a given fix.
      const stamped = withProvenance(result as Record<string, unknown>);

      return {
        content: [
          { type: "text", text: JSON.stringify(stamped, null, 2) },
        ],
        structuredContent: stamped,
        ...(isHandlerError ? { isError: true } : {}),
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              describeCallFailure(name, error),
              null,
              2,
            ),
          },
        ],
        structuredContent: describeCallFailure(name, error),
        isError: true,
      };
    }
  });

  return server;
}
