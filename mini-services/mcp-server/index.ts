/**
 * tokcalc MCP Server — local stdio transport.
 *
 * Exposes 6 read-only planning tools for AI agents (Cursor, Claude Desktop, Cline):
 *   1. estimate_capacity — VRAM/KV/throughput/latency/cost for one config
 *   2. compare_gpus — ranked GPU comparison for one workload
 *   3. recommend_topology — TP/CP topology recommendation
 *   4. estimate_api_vs_self_host — break-even analysis
 *   5. list_models — discover supported model IDs
 *   6. list_gpus — discover supported GPU IDs
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

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
} from "../../src/lib/token-calc.ts";

// Helper function to format Zod schema for MCP protocol compliance (Strips $schema meta-tag)
function formatInputSchema(schema: z.ZodTypeAny) {
  const json = zodToJsonSchema(schema, {
    target: "jsonSchema7",
    $refStrategy: "none",
  }) as Record<string, any>;

  delete json["$schema"];

  return {
    type: "object",
    properties: json.properties || {},
    required: json.required || [],
    ...json,
  };
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
});

const CompareGpusSchema = z.object({
  model: ModelIdSchema,
  quantization: QuantSchema.default("fp16"),
  gpuCount: z.number().int().min(1).max(8).default(1),
  batchSize: z.number().int().min(1).max(1000).default(8),
  promptTokens: z.number().int().min(1).max(2000000).default(500),
  outputTokens: z.number().int().min(1).max(200000).default(200),
  sortBy: z.enum(["lowest_cost","highest_throughput","best_value"]).default("best_value"),
  limit: z.number().int().min(1).max(30).default(10),
});

const RecommendTopologySchema = z.object({
  model: ModelIdSchema,
  quantization: QuantSchema.default("fp16"),
  contextTokens: z.number().int().min(1024).max(2000000).default(8192),
  batchSize: z.number().int().min(1).max(1000).default(1),
});

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
});

const ListModelsSchema = z.object({
  family: z.string().optional().describe("Filter by model family (e.g. 'Llama', 'Qwen')"),
  category: z.enum(["text","vlm","embedding","code","reasoning"]).optional(),
  isMoE: z.boolean().optional(),
});

const ListGpusSchema = z.object({
  vendor: z.string().optional().describe("Filter by vendor (e.g. 'NVIDIA', 'AMD')"),
  category: z.enum(["datacenter","workstation","consumer","mac","tpu","lpu","wse","legacy"]).optional(),
  minVramGb: z.number().optional(),
});

// ============================================================
// TOOL HANDLERS
// ============================================================

function handleEstimateCapacity(input: z.infer<typeof EstimateCapacitySchema>) {
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

  const candidates = GPUS.filter(g => g.vramGb * input.gpuCount >= model.activeParamsB * quant.bytesPerParam);

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

  return {
    summary: `Compared ${results.length} GPUs for ${model.name} (${quant.label}). Cheapest: ${sorted[0]?.gpuName} at $${sorted[0]?.costPerMillionTokens}/M tokens.`,
    comparisons: sorted,
    totalCandidates: results.length,
    sortBy: input.sortBy,
    assumptions: [`Region: us-east-1 (default)`, `On-demand pricing`, `η_mem = 0.65`, `Catalog version: 0.3.0`],
  };
}

function handleRecommendTopology(input: z.infer<typeof RecommendTopologySchema>) {
  const model = MODEL_MAP[input.model];
  const quant = QUANT_MAP[input.quantization as Quantization];
  if (!model || !quant) return { error: "Unknown model or quantization" };

  const results = GPUS.map(g => {
    const rec = recommendTopology(model, g, input.contextTokens, input.batchSize, quant.bytesPerParam);
    const maxConcurrent = computeMaxConcurrency(model, g, 1, input.contextTokens, quant.bytesPerParam);
    const kvPerRequest = computeKVCacheGb(model, input.contextTokens, 1);
    return {
      gpuId: g.id,
      gpuName: g.name,
      vramGb: g.vramGb,
      topology: rec.topology,
      neededGpus: rec.neededGpus,
      fits: rec.fits,
      maxConcurrentUsers: maxConcurrent,
      kvPerRequestGb: +kvPerRequest.toFixed(2),
      reason: rec.reason,
    };
  }).filter(r => r.fits).slice(0, 5);

  return {
    summary: `For ${model.name} at ${fmtContext(input.contextTokens)} context: ${results.length} feasible topology options across ${GPUS.length} GPUs.`,
    model: { name: model.name, paramsB: model.paramsB, activeParamsB: model.activeParamsB, isMoE: model.isMoE },
    context: { tokens: input.contextTokens, label: fmtContext(input.contextTokens) },
    recommendations: results,
    formula: `KV per request = 2 × ${model.layers} layers × ${model.kvHeads} KV heads × ${model.headDim} head_dim × 2 bytes × ${input.contextTokens} tokens = ${computeKVCacheGb(model, input.contextTokens, 1).toFixed(2)} GB`,
    assumptions: [`Single GPU unless TP needed`, `KV cache in FP16`, `Model weights + KV must fit in total VRAM`, `Catalog version: 0.3.0`],
  };
}

function handleEstimateApiVsSelfHost(input: z.infer<typeof EstimateApiVsSelfHostSchema>) {
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
  const effGpuPrice = (gpu?.usdPerHour ?? 0) * input.gpuCount;
  const effTokens = sh.aggregateTokensPerSec * (input.utilization / 100);
  const selfHostCostPerM = effTokens > 0 ? (effGpuPrice / 3600 / effTokens) * 1e6 : Infinity;
  const selfHostMonthly = effGpuPrice * 730;

  const apiCostPerRequest = (input.inputTokens / 1e6) * input.apiInputPrice + (input.outputTokens / 1e6) * input.apiOutputPrice;
  const apiMonthly = apiCostPerRequest * input.requestsPerDay * 30;
  const selfHostMonthlyTotal = selfHostMonthly;

  const breakEven = apiCostPerRequest > 0 ? selfHostMonthly / (30 * apiCostPerRequest) : Infinity;

  const cheaper = selfHostCostPerM < input.apiOutputPrice;
  const meetsVolume = input.requestsPerDay > breakEven;

  return {
    summary: cheaper && meetsVolume
      ? `Self-hosting is cheaper at ${input.requestsPerDay.toLocaleString()} req/day. $${selfHostCostPerM.toFixed(2)}/M vs $${input.apiOutputPrice}/M API.`
      : !meetsVolume
        ? `Not enough volume. Need ${Math.round(breakEven).toLocaleString()} req/day to break even (currently ${input.requestsPerDay.toLocaleString()}).`
        : `API is cheaper. Self-host $${selfHostCostPerM.toFixed(2)}/M vs API $${input.apiOutputPrice}/M.`,
    selfHost: {
      costPerMillionTokens: +selfHostCostPerM.toFixed(2),
      monthlyInfraUsd: +selfHostMonthly.toFixed(2),
      monthlyTotalUsd: +selfHostMonthlyTotal.toFixed(2),
      utilization: `${input.utilization}%`,
      throughput: `${fmtTokens(sh.aggregateTokensPerSec)} tok/s`,
    },
    api: {
      costPerMillionTokens: input.apiOutputPrice,
      monthlyUsd: +apiMonthly.toFixed(2),
      model: input.apiModel,
    },
    breakEven: {
      requestsPerDay: Math.round(breakEven),
      reached: meetsVolume,
      explanation: `At ${input.utilization}% utilization with ${input.gpuCount}× ${gpu?.name || input.gpu}, self-hosting breaks even at ${Math.round(breakEven).toLocaleString()} requests/day.`,
    },
    assumptions: [
      `Self-host throughput: ${fmtTokens(sh.aggregateTokensPerSec)} tok/s at ${input.utilization}% utilization`,
      `GPU price: $${effGpuPrice}/hr`,
      `API pricing: $${input.apiInputPrice}/M input, $${input.apiOutputPrice}/M output`,
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
  if (input.minVramGb) filtered = filtered.filter(g => g.vramGb >= input.minVramGb);

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
];

// ============================================================
// MCP SERVER SETUP
// ============================================================

const server = new Server(
  { name: "tokcalc", version: "0.1.2" },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Handle ListTools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOL_DEFINITIONS,
}));

// Handle CallTool
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
      default:
        return {
          content: [{ type: "text", text: `Unknown tool: ${name}` }],
          isError: true,
        };
    }

    return {
      content: [
        { type: "text", text: JSON.stringify(result, null, 2) },
      ],
      structuredContent: result,
    };
  } catch (error) {
    return {
      content: [
        { type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` },
      ],
      isError: true,
    };
  }
});

// ============================================================
// START SERVER (stdio transport)
// ============================================================

const transport = new StdioServerTransport();
await server.connect(transport);

console.error("tokcalc MCP server started (stdio transport) — 6 tools available");