/**
 * tokcalc MCP server — UNIFIED tool core.
 *
 * This module is the single implementation of every tokcalc tool. Both
 * surfaces import it:
 *   - mini-services/mcp-server/{index,server,http}.ts  → npm stdio/HTTP package
 *   - src/lib/mcp/server.ts (shim)                      → website /api/mcp route
 *
 * Why one core: the tool handlers previously lived as two near-identical
 * copies (one per surface), which is precisely how fix drift happened during
 * the 0.2.2–0.2.4 stale-dist episodes — a fix landed in one file and not the
 * other. Handlers now live here once; the surface files only wire transports.
 *
 * Tools (11):
 *   1. estimate_capacity        — VRAM/KV/throughput/latency/cost for one config
 *   2. compare_gpus             — ranked GPU comparison for one workload
 *   3. recommend_topology       — TP/CP topology recommendation
 *   4. estimate_api_vs_self_host— break-even analysis
 *   5. list_models              — discover supported model IDs
 *   6. list_gpus                — discover supported GPU IDs
 *   7. get_mlperf_benchmarks    — curated MLPerf v4.1 reference configs
 *   8. find_config_for_slo      — INVERSE planner: constraints → feasible configs
 *   9. plan_deployment          — one-call decision brief (memory+perf+cost+risks)
 *  10. fetch_model_spec         — live HuggingFace config.json vs catalog diff
 *  11. record_measured          — store a real measurement; calibrates future estimates
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

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
} from "../token-calc";
import { MLPERF_CURATED } from "../mlperf-curated";
import { MCP_SERVER_VERSION, CATALOG_VERSION } from "../mcp-version";
import {
  readSpecCache,
  writeSpecCache,
  storeIsPersistent,
  writeCalibrationRecord,
  latestCalibrationFor,
} from "../mcp-store";

// ============================================================
// VERSIONS + PROVENANCE
// ============================================================

/** Kept as a re-export so existing imports keep working. */
export const SERVER_VERSION = MCP_SERVER_VERSION;

/** Stamped into every tool result so callers can detect stale deployments. */
function withProvenance<T extends Record<string, unknown>>(result: T): T & {
  provenance: { serverVersion: string; catalogVersion: string; buildStamp: string };
} {
  return {
    ...result,
    provenance: {
      serverVersion: MCP_SERVER_VERSION,
      catalogVersion: CATALOG_VERSION,
      // Machine-greppable token: a single `grep buildStamp` diff tells you
      // whether the running deployment includes a given fix.
      buildStamp: `tokcalc-mcp/${MCP_SERVER_VERSION}`,
    },
  };
}

// ============================================================
// SCHEMA FORMATTING
// ============================================================

// Helper function to format Zod schema for MCP protocol compliance.
// Uses Zod 4's native z.toJSONSchema() instead of zod-to-json-schema@3.x,
// which can't parse Zod 4 ASTs and silently returns empty `{}` schemas.
function formatInputSchema(schema: z.ZodTypeAny) {
  const json = z.toJSONSchema(schema) as Record<string, unknown>;
  delete json["$schema"];
  // Strip defaults from required array — clients shouldn't be required to
  // send fields that already have a default.
  if (Array.isArray(json.required) && json.properties) {
    const required = json.required as string[];
    const props = json.properties as Record<string, Record<string, unknown>>;
    const filtered = required.filter((field: string) => {
      const prop = props[field];
      return prop && prop.default === undefined;
    });
    if (filtered.length === 0) delete json.required;
    else json.required = filtered;
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
  // --- New optional knobs, all backed by the core engine ---
  kvQuantization: z.enum(["fp16","int8","int4"]).default("fp16")
    .describe("KV cache dtype: fp16 (2 B/value), int8 (1 B) or int4 (0.5 B). Lower precision quarters KV size — decisive at 128K+ context. vLLM supports --kv-cache-dtype fp8; llama.cpp q8_0/q4_0."),
  cachePrefixTokens: z.number().int().min(0).max(2000000).optional()
    .describe("Reusable prefix length (system prompt + RAG context) that is cached across requests."),
  cacheHitRate: z.number().min(0).max(1).optional()
    .describe("Fraction of requests that hit the prefix cache (0–1)."),
  useSpeculative: z.boolean().optional()
    .describe("Model speculative decoding (draft model accepts ~2x tokens per step)."),
  speculativeBoost: z.number().min(1).max(10).optional()
    .describe("Speculative decoding speedup multiplier (default 2.0 when useSpeculative is true)."),
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

const FindConfigForSloSchema = z.object({
  model: ModelIdSchema,
  quantization: QuantSchema.default("fp16"),
  contextTokens: z.number().int().min(1).max(10000000).default(8192)
    .describe("Context length the workload must serve."),
  requestsPerDay: z.number().int().min(1).max(100000000).default(10000)
    .describe("Expected daily request volume (drives the monthly cost estimate)."),
  batchSize: z.number().int().min(1).max(1000).default(8)
    .describe("Concurrent requests the config must hold simultaneously."),
  maxTtftMs: z.number().min(1).optional()
    .describe("Hard ceiling on time-to-first-token in ms. Configs above it are excluded."),
  maxCostPerMillion: z.number().min(0).optional()
    .describe("Hard ceiling on $/M output tokens. Configs above it are excluded."),
  minTokensPerSecond: z.number().min(1).optional()
    .describe("Minimum aggregate tok/s the config must deliver at the given batchSize."),
  sortBy: z.enum(["lowest_cost","highest_throughput","best_value"]).default("best_value"),
}).strict();

const PlanDeploymentSchema = z.object({
  model: ModelIdSchema,
  gpu: GpuIdSchema,
  quantization: QuantSchema.default("fp16"),
  gpuCount: z.number().int().min(1).max(64).default(1),
  batchSize: z.number().int().min(1).max(1000).default(8),
  promptTokens: z.number().int().min(1).max(2000000).default(2000),
  outputTokens: z.number().int().min(1).max(200000).default(500),
  requestsPerDay: z.number().int().min(1).max(100000000).default(10000),
  engine: z.enum(["generic","vllm","sglang","trtllm","llamacpp"]).default("vllm"),
  continuousBatching: z.boolean().default(true),
  continuousBatchingMultiplier: z.number().min(1).max(10).default(1.5),
  reasoningTokens: z.number().int().min(0).max(1000000).default(0),
  apiModel: z.string().default("gpt-4o-mini"),
  apiInputPrice: z.number().min(0).default(0.15),
  apiOutputPrice: z.number().min(0).default(0.60),
}).strict();

const FetchModelSpecSchema = z.object({
  modelId: z.string().describe("tokcalc model ID (e.g. 'qwen3-32b'). Call list_models first if unknown."),
  hfRepo: z.string().optional().describe("HuggingFace repo override (e.g. 'Qwen/Qwen3-32B'). Guessed from the catalog when omitted."),
  forceRefresh: z.boolean().optional().describe("Bypass the 24h cache and re-fetch from HuggingFace."),
}).strict();

const RecordMeasuredSchema = z.object({
  model: ModelIdSchema,
  gpu: GpuIdSchema,
  quantization: QuantSchema.default("fp16"),
  gpuCount: z.number().int().min(1).max(64).default(1),
  batchSize: z.number().int().min(1).max(1000).default(1),
  promptTokens: z.number().int().min(1).max(2000000).optional(),
  outputTokens: z.number().int().min(1).max(200000).optional(),
  engine: z.enum(["generic","vllm","sglang","trtllm","llamacpp"]).optional(),
  observed: z.object({
    decodeTokensPerSecond: z.number().positive().optional(),
    aggregateTokensPerSecond: z.number().positive().optional(),
    ttftMs: z.number().positive().optional(),
    itlMs: z.number().positive().optional(),
  }).strict().describe("Measured values from your serving run (vLLM logs, benchmark harness, etc.). At least one field."),
  source: z.string().max(300).optional().describe("Where the measurement came from (e.g. 'vllm bench_serving, 2026-09-28')."),
  notes: z.string().max(2000).optional(),
}).strict();

// ============================================================
// CALIBRATION ANNOTATION (used by estimate_capacity + find_config_for_slo)
// ============================================================

function calibrationAnnotation(
  modelId: string,
  gpuId: string,
  quantization: string,
): Record<string, unknown> | null {
  const records = latestCalibrationFor(modelId, gpuId, quantization);
  const rec = records.find(
    (r) => r.ratios?.decodeTps != null || r.ratios?.aggregateTps != null,
  );
  if (!rec) return null;
  return {
    calibratedFrom: {
      recordedAt: rec.recordedAt,
      ...(rec.source ? { source: rec.source } : {}),
    },
    observedDecodeTokensPerSecond: rec.observed.decodeTps ?? null,
    observedAggregateTokensPerSecond: rec.observed.aggregateTps ?? null,
    measuredOverPredicted: rec.ratios.decodeTps ?? rec.ratios.aggregateTps ?? null,
    note: "Your recorded measurement for this model+GPU+quantization cell. Modeled numbers above are uncorrected — scale them by measuredOverPredicted for a calibrated view.",
  };
}

// ============================================================
// TOOL HANDLERS — original 7
// ============================================================

function handleEstimateCapacity(input: z.infer<typeof EstimateCapacitySchema>) {
  const quantMeta = QUANT_MAP[input.quantization as Quantization];
  const kvBytesPerValue = input.kvQuantization === "int8" ? 1 : input.kvQuantization === "int4" ? 0.5 : 2;

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
    useSpeculative: input.useSpeculative,
    speculativeBoost: input.speculativeBoost,
    cachePrefixTokens: input.cachePrefixTokens,
    cacheHitRate: input.cacheHitRate,
  });

  // 2. Per-model maxContext guard — applies whenever contextTokens is given,
  //    not only when it exceeds promptTokens (previously a caller could pass a
  //    within-prompt but over-ceiling context and skip the check).
  const model = MODEL_MAP[input.model];
  const gpu = GPU_MAP[input.gpu];
  if (input.contextTokens !== undefined && model) {
    if (input.contextTokens > model.maxContext) {
      return {
        error: `contextTokens ${input.contextTokens.toLocaleString("en-US")} exceeds ${model.name}'s maxContext of ${model.maxContext.toLocaleString("en-US")}.`,
        modelMaxContext: model.maxContext,
        requestedContext: input.contextTokens,
        suggestion: `Clamp to the model's real ceiling. Advertising a larger window does not make it servable — KV cache grows linearly with context and would exhaust VRAM.`,
      };
    }
  }
  // promptTokens is a context too — a 2M-token prompt on an 8K model is
  // unservable regardless of VRAM. Guard it the same way as contextTokens.
  if (model && input.promptTokens > model.maxContext) {
    return {
      error: `promptTokens ${input.promptTokens.toLocaleString("en-US")} exceeds ${model.name}'s maxContext of ${model.maxContext.toLocaleString("en-US")}.`,
      modelMaxContext: model.maxContext,
      requestedContext: input.promptTokens,
      suggestion: `Split the input (retrieval, summarization, long-context model) — no serving configuration can run a single prompt past the model's trained window.`,
    };
  }

  // 3. Override KV cache and VRAM if explicit contextTokens provided.
  // calculate() uses promptTokens for both prefill and KV — for RAG/long-context
  // scenarios, callers know the full context length (prompt + retrieved docs).
  // We selectively recompute KV and VRAM without touching prefill-derived TTFT.
  if (input.contextTokens && input.contextTokens > input.promptTokens && model) {
    const kvAtContext = computeKVCacheGb(model, input.contextTokens, input.batchSize, kvBytesPerValue);
    const totalNeeded = result.modelSizeGb + kvAtContext;
    const availableGb = (gpu?.vramGb ?? 0) * input.gpuCount;
    (result as any).kvCacheTotalGb = kvAtContext;
    (result as any).totalVramNeededGb = totalNeeded;
    (result as any).vramFits = totalNeeded <= availableGb;
  } else if (input.kvQuantization !== "fp16") {
    // KV quantization without explicit context: shrink the prompt-sized KV.
    (result as any).kvCacheTotalGb = computeKVCacheGb(model!, input.promptTokens, input.batchSize, kvBytesPerValue);
    (result as any).totalVramNeededGb = result.modelSizeGb + (result as any).kvCacheTotalGb;
    (result as any).vramFits = (result as any).totalVramNeededGb <= (gpu?.vramGb ?? 0) * input.gpuCount;
  }

  return {
    summary: `${model?.name || input.model} on ${gpu?.name || input.gpu} (${input.quantization}): ${fmtTokens(result.decodeTokensPerSec)} tok/s decode, ${fmtMs(result.ttftMs)} TTFT, ${fmtBytes(result.totalVramNeededGb)} VRAM needed, ${fmtMoney(result.costPerMillionOutputTokens)}/M tokens`,
    feasibility: {
      modelFits: result.vramFits,
      fitsWithKvCache: result.vramFits,
      blockingReasons: result.vramFits ? [] : [`Needs ${fmtBytes(result.totalVramNeededGb)} but only ${(gpu?.vramGb ?? 0) * input.gpuCount} GB available`],
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
      activeParamsGb: (result as any).activeParamsGb != null ? +(result as any).activeParamsGb.toFixed(2) : null,
      isMoE: (result as any).isMoE ?? model?.isMoE ?? false,
      weightsNote: ((result as any).isMoE ?? model?.isMoE)
        ? "MoE: modelWeightsGb is ALL experts resident in VRAM. activeParamsGb is the per-token working set used for the bandwidth math. They differ by ~10x — use modelWeightsGb for capacity, activeParamsGb for speed."
        : "Dense model: modelWeightsGb equals activeParamsGb.",
      kvCacheGb: +result.kvCacheTotalGb.toFixed(2),
      kvQuantization: input.kvQuantization,
      totalRequiredGb: +result.totalVramNeededGb.toFixed(2),
      availableGb: gpu ? gpu.vramGb * input.gpuCount : 0,
      utilizationPct: +((result.totalVramNeededGb / ((gpu?.vramGb || 1) * input.gpuCount)) * 100).toFixed(1),
    },
    cost: {
      gpuHourlyUsd: result.costPerHour,
      gpuHourlyNote: input.gpuCount > 1
        ? `Rig total for ${input.gpuCount}× ${gpu?.name ?? input.gpu} (unit price $${gpu?.usdPerHour ?? 0}/hr each). Divide by ${input.gpuCount} for the per-GPU rate.`
        : `Per-GPU on-demand rate.`,
      costPerMillionTokens: result.costPerMillionOutputTokens,
      costPerRequest: result.costPerRequest,
    },
    ...(input.model && gpu ? { calibration: calibrationAnnotation(input.model, input.gpu, input.quantization) } : {}),
    confidence: {
      throughput: result.confidence.decodeTokensPerSec,
      latency: result.confidence.totalLatencyMs,
      memory: result.confidence.totalVramNeededGb,
    },
    assumptions: [
      `η_mem = 0.65 (typical real-world memory utilization)`,
      `η_compute = 0.50 (typical compute utilization)`,
      `KV cache in ${input.kvQuantization === "fp16" ? "FP16 (2 bytes per value)" : `${input.kvQuantization} (${kvBytesPerValue} bytes per value)`}`,
      `Engine: ${input.engine} (affects efficiency factors)`,
      `Continuous batching: ${input.continuousBatching ? `${input.continuousBatchingMultiplier}× multiplier` : "disabled"}`,
      ...(input.useSpeculative ? [`Speculative decoding: ${input.speculativeBoost ?? 2.0}× assumed acceptance boost`] : []),
      ...(input.cachePrefixTokens ? [`Prefix caching: ${input.cachePrefixTokens} tokens at ${Math.round((input.cacheHitRate ?? 0) * 100)}% hit rate`] : []),
      `These are planning estimates, not deployment guarantees`,
    ],
    catalogVersion: CATALOG_VERSION,
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

  // Filter: must have a known price AND meet memory requirement.
  // Uses FULL paramsB (all experts resident for MoE), not activeParamsB —
  // filtering on the active figure let tiny-GPU configs pass the pre-filter
  // and then report modelFits:false from the real fit check.
  const skippedNoPrice: string[] = [];
  const candidates = candidatePool.filter(g => {
    const fitsMemory = g.vramGb * input.gpuCount >= model.paramsB * quant.bytesPerParam;
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
        ? `Fastest: ${fastest?.gpuName} at ${fastest?.aggregateTokensPerSecond.toLocaleString("en-US")} aggregate tok/s (at $${fastest?.costPerMillionTokens.toLocaleString("en-US")}/M)`
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
    assumptions: [`Region: us-east-1 (default)`, `On-demand pricing`, `η_mem = 0.65`, `Catalog version: ${CATALOG_VERSION}`],
  };
}

function handleRecommendTopology(input: z.infer<typeof RecommendTopologySchema>) {
  const model = MODEL_MAP[input.model];
  const quant = QUANT_MAP[input.quantization as Quantization];
  if (!model || !quant) return { error: "Unknown model or quantization" };

  if (input.contextTokens > model.maxContext) {
    return {
      error: `contextTokens ${input.contextTokens.toLocaleString("en-US")} exceeds ${model.name}'s maxContext of ${model.maxContext.toLocaleString("en-US")}.`,
      modelMaxContext: model.maxContext,
      requestedContext: input.contextTokens,
      suggestion: `Clamp to the model's real ceiling. KV cache grows linearly with context and would exhaust VRAM.`,
    };
  }

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
      // The exact division behind maxConcurrentUsers, so the number can be
      // checked by hand instead of taken on faith. No hidden reserve is applied.
      // KV is shown to 4 decimals: at 2 decimals the printed floor could
      // disagree with maxConcurrentUsers (e.g. B200: floor(98.6/1.07)=92 vs
      // the true floor(98.6/1.0737)=91).
      maxConcurrentUsersMath: kvPerRequest > 0
        ? `floor((${g.vramGb} GB × ${rec.neededGpus} − ${residentWeightsGb.toFixed(1)} GB weights) ÷ ${kvPerRequest.toFixed(4)} GB/req) = ${maxConcurrent}`
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
      `Catalog version: ${CATALOG_VERSION}`,
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
      `Catalog version: ${CATALOG_VERSION}`,
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
    catalogVersion: CATALOG_VERSION,
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
    catalogVersion: CATALOG_VERSION,
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
    catalogVersion: CATALOG_VERSION,
  };
}

// ============================================================
// TOOL HANDLERS — new tools
// ============================================================

/** Inverse planner: SLO constraints → feasible configurations. */
function handleFindConfigForSlo(input: z.infer<typeof FindConfigForSloSchema>) {
  const model = MODEL_MAP[input.model];
  const quant = QUANT_MAP[input.quantization as Quantization];
  if (!model) return { error: `Unknown model: ${input.model}. Call list_models to see valid IDs.` };
  if (input.contextTokens > model.maxContext) {
    return {
      error: `contextTokens ${input.contextTokens.toLocaleString("en-US")} exceeds ${model.name}'s maxContext of ${model.maxContext.toLocaleString("en-US")}.`,
      modelMaxContext: model.maxContext,
      requestedContext: input.contextTokens,
    };
  }

  // Deterministic workload shape derived from contextTokens: prefill ≈ half
  // the context window, output = the other half. Stated explicitly so the
  // cost/throughput figures can be reproduced by hand.
  const prefill = Math.max(1, Math.floor(input.contextTokens * 0.5));
  const output = Math.max(1, Math.floor(input.contextTokens * 0.5));
  const tokensPerRequest = prefill + output;

  interface Cand {
    gpuId: string; gpuName: string; vendor: string; topology: string; gpuCount: number;
    vramGb: number; maxConcurrentUsers: number; aggregateTokensPerSecond: number;
    ttftMs: number; costPerMillionTokens: number; estimatedMonthlyUsd: number;
  }
  const candidates: Cand[] = [];
  const rejectedBy = { ttft: 0, cost: 0, throughput: 0, noPrice: 0 };

  for (const g of GPUS) {
    for (const tp of [1, 2, 4, 8]) {
      // Cheap pre-filters before paying for a full calculate().
      if (g.usdPerHour == null || g.usdPerHour <= 0) { rejectedBy.noPrice++; continue; }
      const weightsGb = model.paramsB * quant.bytesPerParam;
      const kvPerReq = computeKVCacheGb(model, input.contextTokens, 1);
      if (weightsGb + input.batchSize * kvPerReq > g.vramGb * tp) continue;

      const r = calculate({
        modelId: input.model,
        gpuId: g.id,
        quantization: input.quantization as Quantization,
        numGpus: tp,
        batchSize: input.batchSize,
        promptTokens: prefill,
        outputTokens: output,
      });
      if (!r.vramFits) continue;
      const tps = r.aggregateTokensPerSec;
      const ttft = r.ttftMs;
      const costPerM = r.costPerMillionOutputTokens;
      if (!Number.isFinite(costPerM)) continue;
      if (input.maxTtftMs !== undefined && ttft > input.maxTtftMs) { rejectedBy.ttft++; continue; }
      if (input.maxCostPerMillion !== undefined && costPerM > input.maxCostPerMillion) { rejectedBy.cost++; continue; }
      if (input.minTokensPerSecond !== undefined && tps < input.minTokensPerSecond) { rejectedBy.throughput++; continue; }

      const monthlyTokens = tokensPerRequest * input.requestsPerDay * 30;
      const monthlyUsd = (costPerM / 1e6) * monthlyTokens;
      candidates.push({
        gpuId: g.id,
        gpuName: g.name,
        vendor: g.vendor,
        topology: tp === 1 ? "Single GPU" : `Tensor Parallel ×${tp}`,
        gpuCount: tp,
        vramGb: g.vramGb,
        maxConcurrentUsers: computeMaxConcurrency(model, g, tp, input.contextTokens, quant.bytesPerParam),
        aggregateTokensPerSecond: Math.round(tps),
        ttftMs: Math.round(ttft),
        costPerMillionTokens: +costPerM.toFixed(2),
        estimatedMonthlyUsd: +monthlyUsd.toFixed(0),
      });
    }
  }

  if (candidates.length === 0) {
    return {
      error: `No feasible configuration for ${model.name} at ${fmtContext(input.contextTokens)} context (batch ${input.batchSize}) within the stated SLOs.`,
      rejectedBy: { exceededMaxTtft: rejectedBy.ttft, exceededMaxCost: rejectedBy.cost, belowMinThroughput: rejectedBy.throughput },
      suggestion: "Relax a constraint (raise maxTtftMs / maxCostPerMillion, lower minTokensPerSecond), lower batchSize, or move to a lower-bit quantization (fp8/int4) to shrink the memory footprint.",
      sloChecked: {
        model: input.model, quantization: input.quantization, contextTokens: input.contextTokens,
        batchSize: input.batchSize, requestsPerDay: input.requestsPerDay,
        maxTtftMs: input.maxTtftMs ?? null, maxCostPerMillion: input.maxCostPerMillion ?? null,
        minTokensPerSecond: input.minTokensPerSecond ?? null,
      },
    };
  }

  candidates.sort((a, b) => {
    if (input.sortBy === "lowest_cost") return a.estimatedMonthlyUsd - b.estimatedMonthlyUsd;
    if (input.sortBy === "highest_throughput") return b.aggregateTokensPerSecond - a.aggregateTokensPerSecond;
    return (b.aggregateTokensPerSecond / Math.max(b.costPerMillionTokens, 0.001)) - (a.aggregateTokensPerSecond / Math.max(a.costPerMillionTokens, 0.001));
  });

  const best = candidates[0];
  return {
    summary: `Found ${candidates.length} feasible config(s) for ${model.name} at ${fmtContext(input.contextTokens)} context (batch ${input.batchSize}). Recommended: ${best.topology} on ${best.gpuName} — ${best.aggregateTokensPerSecond.toLocaleString("en-US")} tok/s, ${fmtMs(best.ttftMs)} TTFT, $${best.costPerMillionTokens}/M, ≈$${best.estimatedMonthlyUsd.toLocaleString("en-US")}/mo at ${input.requestsPerDay.toLocaleString("en-US")} req/day.`,
    recommended: best,
    alternatives: candidates.slice(1, 6),
    totalFeasible: candidates.length,
    sloChecked: {
      model: input.model, quantization: input.quantization, contextTokens: input.contextTokens,
      batchSize: input.batchSize, requestsPerDay: input.requestsPerDay,
      maxTtftMs: input.maxTtftMs ?? null, maxCostPerMillion: input.maxCostPerMillion ?? null,
      minTokensPerSecond: input.minTokensPerSecond ?? null,
    },
    workloadShape: {
      note: "Deterministic stand-in workload: prefill = output = contextTokens ÷ 2 per request. Monthly cost scales linearly with requestsPerDay.",
      prefillTokens: prefill,
      outputTokens: output,
      tokensPerRequest,
    },
    formulaNotes: [
      "Feasibility = (paramsB × bytes/param) + batchSize × KV(context) fits in vramGb × gpuCount.",
      "Monthly = costPerMillion ÷ 1e6 × tokensPerRequest × requestsPerDay × 30.",
      "TTFT uses the core engine's compute-bound prefill model at the given TP degree.",
    ],
    assumptions: [`η_mem = 0.65`, `η_compute = 0.50`, `On-demand pricing, us-east-1 defaults`, `Catalog version: ${CATALOG_VERSION}`],
  };
}

/** One-call decision brief: memory + performance + build-vs-buy + risks. */
function handlePlanDeployment(input: z.infer<typeof PlanDeploymentSchema>) {
  const model = MODEL_MAP[input.model];
  const gpu = GPU_MAP[input.gpu];
  const quant = QUANT_MAP[input.quantization as Quantization];
  if (!model) return { error: `Unknown model: ${input.model}. Call list_models to see valid IDs.` };
  if (!gpu) return { error: `Unknown GPU: ${input.gpu}. Call list_gpus to see valid IDs.` };
  if (input.promptTokens > model.maxContext) {
    return {
      error: `promptTokens ${input.promptTokens.toLocaleString("en-US")} exceeds ${model.name}'s maxContext of ${model.maxContext.toLocaleString("en-US")}.`,
      modelMaxContext: model.maxContext,
      requestedContext: input.promptTokens,
    };
  }

  const calc = calculate({
    modelId: input.model,
    gpuId: input.gpu,
    quantization: input.quantization as Quantization,
    numGpus: input.gpuCount,
    batchSize: input.batchSize,
    promptTokens: input.promptTokens,
    outputTokens: input.outputTokens,
    engineId: input.engine,
    useContinuousBatching: input.continuousBatching,
    continuousBatchingMultiplier: input.continuousBatchingMultiplier,
    reasoningTokens: input.reasoningTokens,
  });

  const availableGb = gpu.vramGb * input.gpuCount;
  if (!calc.vramFits) {
    const topo = recommendTopology(model, gpu, input.promptTokens, input.batchSize, quant.bytesPerParam);
    return {
      error: `Infeasible: ${model.name} (${input.quantization}) + KV needs ${fmtBytes(calc.totalVramNeededGb)} but ${input.gpuCount}× ${gpu.name} provides ${fmtBytes(availableGb)}.`,
      recommendedTopology: {
        topology: topo.topology,
        neededGpus: topo.neededGpus,
        hasContextParallel: topo.hasContextParallel,
        reason: topo.reason,
      },
      nextSteps: topo.hasContextParallel
        ? [`Re-run with gpuCount=8 and note Context Parallel is required (RingAttention), or reduce contextTokens.`, `Or pick a smaller model / lower-bit quantization.`]
        : [`Re-run with gpuCount=${topo.neededGpus}.`, `Or pick a smaller model / lower-bit quantization (fp8, int4).`],
    };
  }

  // Build-vs-buy (blended API, same convention as estimate_api_vs_self_host)
  const apiCostPerRequest = (input.promptTokens / 1e6) * input.apiInputPrice + (input.outputTokens / 1e6) * input.apiOutputPrice;
  const apiMonthly = apiCostPerRequest * input.requestsPerDay * 30;
  const rigHourly = (gpu.usdPerHour ?? 0) * input.gpuCount;
  const selfMonthly = rigHourly * 730;
  const tokensPerRequest = input.promptTokens + input.outputTokens;
  const apiBlendedPerM = tokensPerRequest > 0 ? (apiCostPerRequest / tokensPerRequest) * 1e6 : input.apiOutputPrice;
  const breakEven = apiCostPerRequest > 0 ? selfMonthly / (30 * apiCostPerRequest) : Infinity;
  const meetsVolume = input.requestsPerDay >= breakEven;
  const verdict = meetsVolume
    ? `Self-host wins at ${input.requestsPerDay.toLocaleString("en-US")} req/day (break-even ${Math.round(breakEven).toLocaleString("en-US")}).`
    : `API wins at this volume — break-even is ${Math.round(breakEven).toLocaleString("en-US")} req/day, you plan ${input.requestsPerDay.toLocaleString("en-US")}.`;

  // Headroom: how many times the current KV load fits in free VRAM after weights
  const freeGb = availableGb - calc.modelSizeGb;
  const kvHeadroomMultiplier = calc.kvCacheTotalGb > 0 ? Math.max(0, Math.floor(freeGb / calc.kvCacheTotalGb)) : null;

  const risks: string[] = [];
  if (calc.longContextWarning) risks.push(calc.longContextWarning);
  if (kvHeadroomMultiplier !== null && kvHeadroomMultiplier <= 2) {
    risks.push(`Memory headroom is tight: free VRAM after weights covers only ≈${kvHeadroomMultiplier}× the current KV load. Batch growth or longer contexts will OOM.`);
  }
  if (input.gpuCount > 1 && gpu.nvlinkGbps === 0) {
    risks.push(`${gpu.name} has no NVLink — tensor-parallel traffic crosses PCIe, so real TP scaling will be worse than the ${Math.round(calc.multiGpuEfficiency * 100)}% efficiency assumed here.`);
  }
  if (gpu.flopsTflops === null) {
    risks.push(`${gpu.name} dense FP16 is not publicly reported — compute figures use a conservative fallback.`);
  }
  risks.push("All figures are planning projections; validate with a real serving run before committing spend.");

  return {
    summary: `${model.name} on ${input.gpuCount}× ${gpu.name} (${input.quantization}, ${input.engine}): ${Math.round(calc.aggregateTokensPerSec).toLocaleString("en-US")} aggregate tok/s, ${fmtMs(calc.ttftMs)} TTFT, ${fmtBytes(calc.totalVramNeededGb)} VRAM, ${fmtMoney(calc.costPerMillionOutputTokens)}/M output. ${verdict}`,
    config: {
      model: input.model, gpu: input.gpu, quantization: input.quantization,
      gpuCount: input.gpuCount, batchSize: input.batchSize,
      promptTokens: input.promptTokens, outputTokens: input.outputTokens,
      requestsPerDay: input.requestsPerDay, engine: input.engine,
      continuousBatching: input.continuousBatching ? `${input.continuousBatchingMultiplier}×` : "disabled",
      reasoningTokens: input.reasoningTokens,
    },
    memory: {
      weightsGb: +calc.modelSizeGb.toFixed(2),
      kvCacheGb: +calc.kvCacheTotalGb.toFixed(2),
      totalRequiredGb: +calc.totalVramNeededGb.toFixed(2),
      availableGb,
      utilizationPct: +((calc.totalVramNeededGb / availableGb) * 100).toFixed(1),
      kvHeadroomMultiplier,
      headroomNote: kvHeadroomMultiplier !== null
        ? `Free VRAM after weights fits ≈${kvHeadroomMultiplier}× the current KV load (batch ${input.batchSize} at ${input.promptTokens} tokens).`
        : "KV load is zero at this configuration.",
    },
    performance: {
      decodeTokensPerSecond: Math.round(calc.decodeTokensPerSec),
      aggregateTokensPerSecond: Math.round(calc.aggregateTokensPerSec),
      ttftMs: Math.round(calc.ttftMs),
      itlMs: Number.isFinite(calc.itlMs) ? Math.round(calc.itlMs) : null,
      billedOutputTokens: calc.billedOutputTokens,
    },
    cost: {
      rigHourlyUsd: +rigHourly.toFixed(2),
      rigHourlyNote: input.gpuCount > 1 ? `${input.gpuCount}× GPUs at $${gpu.usdPerHour}/hr each.` : "Single GPU on-demand rate.",
      costPerMillionOutputTokens: +calc.costPerMillionOutputTokens.toFixed(2),
      costPerRequest: +calc.costPerRequest.toFixed(4),
    },
    buildVsBuy: {
      selfHostMonthlyUsd: +selfMonthly.toFixed(0),
      apiMonthlyUsd: +apiMonthly.toFixed(0),
      apiBlendedCostPerMillion: +apiBlendedPerM.toFixed(2),
      apiModel: input.apiModel,
      breakEvenRequestsPerDay: Math.round(breakEven),
      verdict: meetsVolume ? "self-host" : "api",
    },
    risks,
    nextSteps: [
      `Cross-check with find_config_for_slo to see whether a different GPU beats ${gpu.name} on cost.`,
      "Fetch the real spec with fetch_model_spec to confirm layers / KV heads before ordering hardware.",
      "After the first serving run, record the measured tok/s with record_measured so future estimates self-correct.",
    ],
    assumptions: [
      `η_mem = 0.65, η_compute = 0.50`,
      `KV cache in FP16`,
      `730 hours/month; on-demand pricing`,
      `Break-even uses API cost only (no self-host ops overhead)`,
      `Catalog version: ${CATALOG_VERSION}`,
    ],
  };
}

// ---- fetch_model_spec ----

const HF_REPO_BY_MODEL: Record<string, string> = {
  "llama3-8b": "meta-llama/Meta-Llama-3-8B",
  "llama3-70b": "meta-llama/Meta-Llama-3-70B",
  "llama3-405b": "meta-llama/Meta-Llama-3.1-405B",
  "llama3-3-70b": "meta-llama/Llama-3.3-70B-Instruct",
  "llama2-7b": "meta-llama/Llama-2-7b-hf",
  "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
  "mistral-7b": "mistralai/Mistral-7B-v0.3",
  "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
  "mixtral-8x22b": "mistralai/Mixtral-8x22B-v0.1",
  "mistral-large-3": "mistralai/Mistral-Large-3",
  "pixtral-12b": "mistralai/Pixtral-12B-2409",
  "codestral-25": "mistralai/Codestral-25.08-v01",
  "qwen2-7b": "Qwen/Qwen2-7B",
  "qwen2-72b": "Qwen/Qwen2-72B",
  "qwen2-5-14b": "Qwen/Qwen2.5-14B",
  "qwen2-5-72b": "Qwen/Qwen2.5-72B",
  "qwen3-4b": "Qwen/Qwen3-4B",
  "qwen3-8b": "Qwen/Qwen3-8B",
  "qwen3-14b": "Qwen/Qwen3-14B",
  "qwen3-32b": "Qwen/Qwen3-32B",
  "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
  "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
  "qwen2-5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
  "qwen2-5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
  "deepseek-v3": "deepseek-ai/DeepSeek-V3",
  "deepseek-r1": "deepseek-ai/DeepSeek-R1",
  "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
  "gemma2-9b": "google/gemma-2-9b",
  "gemma2-27b": "google/gemma-2-27b",
  "phi3-7b": "microsoft/Phi-3-mini-4k-instruct",
  "phi4-14b": "microsoft/phi-4",
  "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "falcon3-10b": "tiiuae/Falcon3-10B-Instruct",
  "olmo2-13b": "allenai/OLMo-2-1124-13B",
  "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
  "bge-m3": "BAAI/bge-m3",
  "e5-mlarge": "intfloat/multilingual-e5-large",
  "gte-large": "thenlper/gte-large",
};

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

async function handleFetchModelSpec(input: z.infer<typeof FetchModelSpecSchema>) {
  const model = MODEL_MAP[input.modelId];
  if (!model) {
    return {
      error: `Unknown model: ${input.modelId}. Call list_models to see valid IDs.`,
      validModelIds: MODELS.slice(0, 12).map(m => m.id),
    };
  }
  const repo = input.hfRepo ?? HF_REPO_BY_MODEL[input.modelId] ?? null;
  if (!repo) {
    return {
      error: `No HuggingFace repo known for ${input.modelId}. Pass hfRepo explicitly (e.g. 'org/model-name').`,
      hint: "hfRepo is the repo path on huggingface.co whose config.json should be compared against the catalog.",
    };
  }

  const hfUrl = `https://huggingface.co/${repo}/resolve/main/config.json`;
  const catalog = {
    paramsB: model.paramsB,
    activeParamsB: model.activeParamsB,
    isMoE: model.isMoE,
    layers: model.layers,
    hiddenDim: model.hiddenDim,
    qHeads: model.qHeads,
    kvHeads: model.kvHeads,
    headDim: model.headDim,
    vocabSize: model.vocabSize,
    maxContext: model.maxContext,
  };

  // 1. Cache
  const cached = readSpecCache(input.modelId);
  if (cached && !input.forceRefresh && cached.hfRepo === repo) {
    const cfg = cached.config as Record<string, unknown>;
    const hfMapped = {
      layers: num(cfg["num_hidden_layers"]),
      hiddenDim: num(cfg["hidden_size"]),
      qHeads: num(cfg["num_attention_heads"]),
      kvHeads: num(cfg["num_key_value_heads"]) ?? num(cfg["num_attention_heads"]),
      headDim: num(cfg["head_dim"]) ?? (num(cfg["hidden_size"]) && num(cfg["num_attention_heads"]) ? num(cfg["hidden_size"])! / num(cfg["num_attention_heads"])! : null),
      vocabSize: num(cfg["vocab_size"]),
      maxPositionEmbeddings: num(cfg["max_position_embeddings"]),
      numExpertsRouted: num(cfg["num_experts"]),
      numExpertsPerTok: num(cfg["num_experts_per_tok"]),
    };
    return {
      source: "cache",
      cacheAgeHours: +((Date.now() - new Date(cached.fetchedAt).getTime()) / 3600000).toFixed(1),
      model: { id: model.id, name: model.name, family: model.family },
      hfRepo: repo,
      hfUrl,
      catalog,
      hfMapped,
      comparison: compareSpecs(catalog, hfMapped),
      note: "Served from the local 24h spec cache. Use forceRefresh=true to re-fetch.",
      catalogVersion: CATALOG_VERSION,
    };
  }

  // 2. Live fetch
  try {
    const response = await fetch(hfUrl, { signal: AbortSignal.timeout(10_000) });
    if (!response.ok) throw new Error(`HuggingFace returned ${response.status} for ${repo}/config.json`);
    const config = (await response.json()) as Record<string, unknown>;

    const hfMapped = {
      layers: num(config["num_hidden_layers"]),
      hiddenDim: num(config["hidden_size"]),
      qHeads: num(config["num_attention_heads"]),
      kvHeads: num(config["num_key_value_heads"]) ?? num(config["num_attention_heads"]),
      headDim: num(config["head_dim"]) ?? (num(config["hidden_size"]) && num(config["num_attention_heads"]) ? num(config["hidden_size"])! / num(config["num_attention_heads"])! : null),
      vocabSize: num(config["vocab_size"]),
      maxPositionEmbeddings: num(config["max_position_embeddings"]),
      numExpertsRouted: num(config["num_experts"]),
      numExpertsPerTok: num(config["num_experts_per_tok"]),
    };

    const written = writeSpecCache({ modelId: input.modelId, hfRepo: repo, fetchedAt: new Date().toISOString(), config });
    const persistent = written && storeIsPersistent();

    return {
      source: "huggingface",
      model: { id: model.id, name: model.name, family: model.family },
      hfRepo: repo,
      hfUrl,
      catalog,
      hfMapped,
      comparison: compareSpecs(catalog, hfMapped),
      cache: written
        ? { stored: true, persistent, ttlHours: 24, note: persistent ? "Cached for 24h in ~/.tokcalc/model-specs/." : "Cached to a temp dir (read-only home) — will not survive a restart." }
        : { stored: false, note: "Filesystem unavailable — fetched live, not cached." },
      note: "Verify catalog drift before sizing hardware: layers, kvHeads and headDim drive the KV-cache formula (2 × L × H_kv × D_h × 2 bytes).",
      catalogVersion: CATALOG_VERSION,
    };
  } catch (err) {
    // Stale fallback is only safe when the cached entry is for the SAME repo.
    // A 404 for repo X must not silently serve a spec cached from repo Y —
    // that presents old data as the answer to a request that factually failed.
    if (cached && cached.hfRepo === repo) {
      const ageHours = +((Date.now() - new Date(cached.fetchedAt).getTime()) / 3600000).toFixed(1);
      return {
        source: "cache-stale",
        warning: `HuggingFace fetch failed (${err instanceof Error ? err.message : String(err)}); serving a STALE cache entry from ${cached.fetchedAt}.`,
        cacheAgeHours: ageHours,
        model: { id: model.id, name: model.name, family: model.family },
        hfRepo: repo,
        hfUrl,
        catalog,
        hfMapped: null,
        comparison: null,
        catalogVersion: CATALOG_VERSION,
      };
    }
    return {
      error: `Failed to fetch ${hfUrl}: ${err instanceof Error ? err.message : String(err)}`,
      suggestion: "Check the repo id, network access, or pass hfRepo explicitly. Offline? The catalog remains fully usable via the other tools.",
      catalogVersion: CATALOG_VERSION,
    };
  }
}

function compareSpecs(catalog: Record<string, unknown>, hf: Record<string, unknown> | null) {
  if (!hf) return null;
  const rows: Array<{ field: string; catalog: unknown; huggingface: unknown; drift: boolean }> = [];
  const push = (field: string, cat: unknown, hfVal: unknown, compare = true) => {
    rows.push({
      field,
      catalog: cat ?? null,
      huggingface: hfVal ?? null,
      drift: compare && hfVal != null && cat != null && !Object.is(Number(cat), Number(hfVal)),
    });
  };
  push("layers", catalog.layers, hf.layers);
  push("kvHeads", catalog.kvHeads, hf.kvHeads);
  push("headDim", catalog.headDim, hf.headDim);
  push("hiddenDim", catalog.hiddenDim, hf.hiddenDim);
  push("qHeads", catalog.qHeads, hf.qHeads);
  push("vocabSize", catalog.vocabSize, hf.vocabSize);
  push("maxContext", catalog.maxContext, hf.maxPositionEmbeddings);
  push("numExpertsRouted", null, hf.numExpertsRouted, false);
  push("numExpertsPerTok", null, hf.numExpertsPerTok, false);
  const drifted = rows.filter(r => r.drift);
  return {
    driftedCount: drifted.length,
    driftedFields: drifted.map(r => r.field),
    verdict: drifted.length === 0 ? "Catalog matches HuggingFace on every comparable field." : `DRIFT: catalog differs from HuggingFace on ${drifted.length} field(s). KV-cache math uses layers × kvHeads × headDim — reconcile before trusting capacity numbers.`,
    fields: rows,
  };
}

// ---- record_measured ----

function handleRecordMeasured(input: z.infer<typeof RecordMeasuredSchema>) {
  const model = MODEL_MAP[input.model];
  const gpu = GPU_MAP[input.gpu];
  if (!model) return { error: `Unknown model: ${input.model}. Call list_models to see valid IDs.` };
  if (!gpu) return { error: `Unknown GPU: ${input.gpu}. Call list_gpus to see valid IDs.` };
  if (Object.keys(input.observed).length === 0) {
    return { error: "observed must contain at least one measurement (decodeTokensPerSecond, aggregateTokensPerSecond, ttftMs, or itlMs)." };
  }

  const predicted = calculate({
    modelId: input.model,
    gpuId: input.gpu,
    quantization: input.quantization as Quantization,
    numGpus: input.gpuCount,
    batchSize: input.batchSize,
    promptTokens: input.promptTokens ?? 500,
    outputTokens: input.outputTokens ?? 200,
    engineId: input.engine ?? "generic",
  });

  const ratios: Record<string, number> = {};
  if (input.observed.decodeTokensPerSecond != null && predicted.decodeTokensPerSec > 0) {
    ratios.decodeTps = +(input.observed.decodeTokensPerSecond / predicted.decodeTokensPerSec).toFixed(3);
  }
  if (input.observed.aggregateTokensPerSecond != null && predicted.aggregateTokensPerSec > 0) {
    ratios.aggregateTps = +(input.observed.aggregateTokensPerSecond / predicted.aggregateTokensPerSec).toFixed(3);
  }
  if (input.observed.ttftMs != null && predicted.ttftMs > 0) {
    ratios.ttftMs = +(predicted.ttftMs / input.observed.ttftMs).toFixed(3);
  }
  if (input.observed.itlMs != null && Number.isFinite(predicted.itlMs) && predicted.itlMs > 0) {
    ratios.itlMs = +(predicted.itlMs / input.observed.itlMs).toFixed(3);
  }
  // Ratios convention: throughput ratios are observed ÷ predicted (>1 = model
  // faster than predicted); latency ratios are predicted ÷ observed
  // (>1 = real system faster than predicted). Both directions: >1 means the
  // real world beats the model.

  const record = {
    model: input.model,
    gpu: input.gpu,
    quantization: input.quantization,
    gpuCount: input.gpuCount,
    batchSize: input.batchSize,
    observed: {
      decodeTps: input.observed.decodeTokensPerSecond,
      aggregateTps: input.observed.aggregateTokensPerSecond,
      ttftMs: input.observed.ttftMs,
      itlMs: input.observed.itlMs,
    },
    ratios,
    source: input.source,
    notes: input.notes,
    recordedAt: new Date().toISOString(),
  };

  const write = writeCalibrationRecord(record as any);
  const totalForCell = latestCalibrationFor(input.model, input.gpu, input.quantization).length;
  const primary = ratios.decodeTps ?? ratios.aggregateTps;

  return {
    summary: primary !== undefined
      ? `Recorded ${input.observed.decodeTokensPerSecond ?? input.observed.aggregateTokensPerSecond} tok/s on ${input.gpuCount}× ${gpu.name}. Predicted ${Math.round(ratios.decodeTps !== undefined ? predicted.decodeTokensPerSec : predicted.aggregateTokensPerSec)}; measured/predicted ratio ${primary}×.`
      : `Recorded latency-only measurement for ${model.name} on ${gpu.name}.`,
    predicted: {
      decodeTokensPerSecond: Math.round(predicted.decodeTokensPerSec),
      aggregateTokensPerSecond: Math.round(predicted.aggregateTokensPerSec),
      ttftMs: Number.isFinite(predicted.ttftMs) ? Math.round(predicted.ttftMs) : null,
      itlMs: Number.isFinite(predicted.itlMs) ? Math.round(predicted.itlMs) : null,
      contextOfPrediction: "Core engine output for the same model/GPU/quant/gpuCount/batchSize you reported.",
    },
    observed: input.observed,
    ratios,
    ratioConvention: ">1 = real system faster than predicted (throughput: observed ÷ predicted; latency: predicted ÷ observed).",
    storedTo: write.path,
    persistent: write.ok && storeIsPersistent(),
    recordsForThisConfig: totalForCell,
    note: write.ok
      ? (storeIsPersistent()
          ? "estimate_capacity and find_config_for_slo now annotate results for this model+GPU+quantization cell with your measured/predicted ratio."
          : "Stored to a temp dir (home dir unavailable) — this calibration will NOT survive a restart.")
      : "Could not store the measurement (filesystem unavailable); returning the comparison only — nothing was persisted.",
    assumptions: [`Ratios are per model+GPU+quantization cell; the most recent record wins.`, `Catalog version: ${CATALOG_VERSION}`],
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
  {
    name: "find_config_for_slo",
    description: "INVERSE planner: given model + context + traffic volume + optional TTFT/cost/throughput ceilings, search every GPU × topology (1/2/4/8 GPUs) and return feasible configurations ranked by cost, throughput, or value. Use when the user knows their SLOs but not the hardware.",
    inputSchema: formatInputSchema(FindConfigForSloSchema),
  },
  {
    name: "plan_deployment",
    description: "One-call deployment decision brief for a specific config: memory breakdown with KV headroom, performance, rig cost, blended build-vs-buy with break-even, concrete risks (no-NVLink TP, tight headroom, long context), and next steps. Chains what would otherwise take 4 separate tool calls.",
    inputSchema: formatInputSchema(PlanDeploymentSchema),
  },
  {
    name: "fetch_model_spec",
    description: "Fetch a model's real config.json from HuggingFace and diff it against tokcalc's catalog (layers, KV heads, head_dim, vocab, max context). Catches catalog drift before it corrupts KV-cache math. Cached 24h locally; use forceRefresh to re-fetch. Works offline via cache fallback.",
    inputSchema: formatInputSchema(FetchModelSpecSchema),
  },
  {
    name: "record_measured",
    description: "Record a real-world measurement (observed tok/s, TTFT, ITL) for a model+GPU+quantization cell. Compares against the model's prediction, stores a measured/predicted ratio, and future estimate_capacity / find_config_for_slo results for that cell are annotated with your calibration. The estimator improves as you use it.",
    inputSchema: formatInputSchema(RecordMeasuredSchema),
  },
];

/** The schema backing each tool, so we can echo valid names on a bad call. */
const TOOL_SCHEMAS: Record<string, z.ZodType> = {
  estimate_capacity: EstimateCapacitySchema,
  compare_gpus: CompareGpusSchema,
  recommend_topology: RecommendTopologySchema,
  estimate_api_vs_self_host: EstimateApiVsSelfHostSchema,
  list_models: ListModelsSchema,
  list_gpus: ListGpusSchema,
  get_mlperf_benchmarks: GetMlperfBenchmarksSchema,
  find_config_for_slo: FindConfigForSloSchema,
  plan_deployment: PlanDeploymentSchema,
  fetch_model_spec: FetchModelSpecSchema,
  record_measured: RecordMeasuredSchema,
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
      serverVersion: MCP_SERVER_VERSION,
      buildStamp: `tokcalc-mcp/${MCP_SERVER_VERSION}`,
    },
  };

  const issues =
    error && typeof error === "object" && "issues" in error
      ? ((error as { issues: unknown }).issues as Array<Record<string, unknown>>)
      : null;

  if (!issues || issues.length === 0) return base;

  const schema = TOOL_SCHEMAS[toolName];
  const validKeys = schema
    ? Object.keys((schema as unknown as { shape: Record<string, unknown> }).shape ?? {})
    : [];

  // Zod 4 puts the offending keys of an `unrecognized_keys` issue in
  // `issue.keys` (issue.path is the path of the OBJECT, usually empty) —
  // reading only `.path` is why the error used to say
  // "Unrecognized parameter(s): ." without naming the culprit.
  const keysOf = (issue: Record<string, unknown>): string[] => {
    if (Array.isArray(issue.keys)) {
      return issue.keys.filter((k): k is string => typeof k === "string");
    }
    if (Array.isArray(issue.path)) {
      return issue.path.filter((p): p is string => typeof p === "string");
    }
    return [];
  };

  const unrecognized = issues.filter((i) => i.code === "unrecognized_keys");
  const invalidValue = issues.filter((i) => i.code === "invalid_value");
  const wrongType = issues.filter((i) => i.code === "invalid_type");

  if (unrecognized.length > 0 && validKeys.length > 0) {
    const bad = Array.from(new Set(unrecognized.flatMap(keysOf)));

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
      error: `Unrecognized parameter(s): ${bad.join(", ") || "(unknown)"}. These schemas are strict — unknown keys are rejected rather than ignored, so nothing was silently dropped.`,
      unrecognizedParameters: bad,
      suggestions,
      validParameters: validKeys,
      hint: `Call tools/list (or read the inputSchema) to see every accepted parameter and its default.`,
    };
  }

  if (wrongType.length > 0 || invalidValue.length > 0) {
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
 * Create a fresh MCP Server instance with all 11 tools wired.
 *
 * Used by:
 *   - stdio entry point (index.ts) — one server per process
 *   - HTTP entry point (http.ts) — one server per request (stateless mode)
 *   - website /api/mcp route — one server per request (stateless mode)
 *
 * Each caller creates its own instance via this factory to avoid
 * shared transport state (per v0.2.0 research recommendation #5:
 * "Do not connect the same McpServer instance to multiple transports").
 */
export function createMcpServer(): Server {
  const server = new Server(
    { name: "tokcalc", version: MCP_SERVER_VERSION },
    {
      capabilities: {
        tools: {},
      },
      instructions:
        `tokcalc MCP server ${MCP_SERVER_VERSION} — LLM serving capacity planner (11 tools). ` +
        `Use list_models / list_gpus to discover canonical IDs before calling the estimators. ` +
        `Know your SLOs but not the hardware? Use find_config_for_slo. Want one call, one decision? ` +
        `Use plan_deployment. Suspect the catalog? fetch_model_spec diffs it against HuggingFace. ` +
        `Have real measurements? record_measured calibrates future estimates. All estimates are ` +
        `planning projections, not deployment guarantees. Every response carries provenance.serverVersion.`,
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
          result = await handleEstimateCapacity(input);
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
        case "find_config_for_slo": {
          const input = FindConfigForSloSchema.parse(args);
          result = handleFindConfigForSlo(input);
          break;
        }
        case "plan_deployment": {
          const input = PlanDeploymentSchema.parse(args);
          result = handlePlanDeployment(input);
          break;
        }
        case "fetch_model_spec": {
          const input = FetchModelSpecSchema.parse(args);
          result = await handleFetchModelSpec(input);
          break;
        }
        case "record_measured": {
          const input = RecordMeasuredSchema.parse(args);
          result = handleRecordMeasured(input);
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
      const failure = describeCallFailure(name, error);
      return {
        content: [
          { type: "text", text: JSON.stringify(failure, null, 2) },
        ],
        structuredContent: failure,
        isError: true,
      };
    }
  });

  return server;
}

/**
 * Test hooks — NOT part of the public MCP API.
 * Lets the golden-number test suite exercise the exact handlers behind the
 * tools (so our hand-verified oracle values fail CI, not just the math core).
 */
export const __test = {
  EstimateCapacitySchema,
  CompareGpusSchema,
  RecommendTopologySchema,
  EstimateApiVsSelfHostSchema,
  FindConfigForSloSchema,
  PlanDeploymentSchema,
  RecordMeasuredSchema,
  handleEstimateCapacity,
  handleCompareGpus,
  handleRecommendTopology,
  handleEstimateApiVsSelfHost,
  handleFindConfigForSlo,
  handlePlanDeployment,
  handleRecordMeasured,
  describeCallFailure,
};
