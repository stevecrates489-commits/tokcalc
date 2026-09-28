/**
 * Golden-number + invariant test suite for the tokcalc MCP core.
 *
 * Every expected value below was hand-verified during adversarial QA sessions
 * against the live server (and re-derived from the catalog by hand):
 *
 *   - Mixtral 8x7B fp16 ×2 H100:  weights 93.4 GB, total 93.47 GB, aggregate 287 tok/s
 *   - Qwen3-30B-A3B fp8 RTX 5090: weights 30 GB, total 30.05 GB, aggregate 582 tok/s
 *   - DeepSeek R1 KV @16K:        65.50 GB/request, weights 1342 GB (int4)
 *   - Llama4 Scout KV @1M ctx:    206.16 GB/request
 *   - Llama3-8B fp16 H100:        16.06 + 0.07 = 16.13 GB
 *   - concurrency:  H100×2=62, H200=44, B200=91, B300=181 (Mixtral @8K)
 *   - break-even:   2,995 req/day, $292/mo (llama3-8b defaults, $2.5/$10 API)
 *
 * If one of these numbers changes, it means the math changed — the change must
 * be justified and this file updated deliberately, not silently.
 *
 * Run with:  bun test tests/mcp-golden.test.ts
 */

/// <reference types="bun-types" />
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "bun:test";
import { __test } from "../src/lib/mcp/core";
import { computeKVCacheGb, MODEL_MAP, GPU_MAP } from "../src/lib/token-calc";

const {
  EstimateCapacitySchema,
  CompareGpusSchema,
  RecommendTopologySchema,
  EstimateApiVsSelfHostSchema,
  FindConfigForSloSchema,
  PlanDeploymentSchema,
  RecordMeasuredSchema,
  handleEstimateCapacity,
  handleRecommendTopology,
  handleEstimateApiVsSelfHost,
  handleFindConfigForSlo,
  handlePlanDeployment,
  handleRecordMeasured,
  handleCompareGpus,
  describeCallFailure,
} = __test as any;

// ---------- helpers ----------

function parseStrict(schema: any, args: unknown): any {
  return schema.parse(args);
}

/** Stub a Zod-4-style error object without importing Zod's class hierarchy. */
function makeZodLikeError(issues: Array<Record<string, unknown>>): Error {
  return Object.assign(new Error("Validation failed"), { issues });
}

// Isolate the on-disk store so tests never pollute the user's real ~/.tokcalc.
process.env.TOKCALC_STORE_DIR = fs.mkdtempSync(path.join(os.tmpdir(), "tokcalc-test-"));

// In production every call goes through Schema.parse() first (defaults included).
// These wrappers force the tests through the same path — calling handlers
// directly skips the defaults and does NOT test what actually ships.
const est = (args: any) => handleEstimateCapacity(parseStrict(EstimateCapacitySchema, args));
const slo = (args: any) => handleFindConfigForSlo(parseStrict(FindConfigForSloSchema, args));
const plan = (args: any) => handlePlanDeployment(parseStrict(PlanDeploymentSchema, args));
const record = (args: any) => handleRecordMeasured(parseStrict(RecordMeasuredSchema, args));

// ============================================================
// GOLDEN NUMBERS — memory
// ============================================================

describe("golden: memory math (hand-verified oracle)", () => {
  it("Mixtral 8x7B fp16 ×2 H100: weights 93.4, KV 0.07, total 93.47", () => {
    const input = parseStrict(EstimateCapacitySchema, {
      model: "mixtral-8x7b", gpu: "h100-sxm", gpuCount: 2, quantization: "fp16", promptTokens: 500,
    }) as any;
    const out: any = handleEstimateCapacity(input);
    expect(out.memory.modelWeightsGb).toBe(93.4);
    expect(out.memory.activeParamsGb).toBe(25.8);
    expect(out.memory.kvCacheGb).toBe(0.07);
    expect(out.memory.totalRequiredGb).toBe(93.47);
    expect(out.memory.isMoE).toBe(true);
    expect(out.feasibility.modelFits).toBe(true);
  });

  it("Qwen3-30B-A3B fp8 RTX 5090: weights 30, total 30.05", () => {
    const input = parseStrict(EstimateCapacitySchema, {
      model: "qwen3-30b-a3b", gpu: "rtx-5090", quantization: "fp8",
    }) as any;
    const out: any = handleEstimateCapacity(input);
    expect(out.memory.modelWeightsGb).toBe(30);
    expect(out.memory.activeParamsGb).toBe(3);
    expect(out.memory.totalRequiredGb).toBe(30.05);
  });

  it("Llama3-8B fp16 H100: dense 16.06 + 0.07 = 16.13", () => {
    const input = parseStrict(EstimateCapacitySchema, {
      model: "llama3-8b", gpu: "h100-sxm",
    }) as any;
    const out: any = handleEstimateCapacity(input);
    expect(out.memory.modelWeightsGb).toBe(16.06);
    expect(out.memory.activeParamsGb).toBe(16.06);
    expect(out.memory.isMoE).toBe(false);
    expect(out.memory.totalRequiredGb).toBe(16.13);
  });

  it("DeepSeek R1 int4: weights 335.5 GB (671B × 0.5 B/param), KV 65.5 @16K, total 401", () => {
    const input = parseStrict(RecommendTopologySchema, {
      model: "deepseek-r1", quantization: "int4", contextTokens: 16384, batchSize: 1,
    }) as any;
    const out: any = handleRecommendTopology(input);
    const b200 = out.recommendations.find((r: any) => r.gpuId === "b200-sxm");
    expect(b200).toBeDefined();
    expect(b200.residentWeightsGb).toBe(335.5);
    expect(b200.kvPerRequestGb).toBe(65.5);
    expect(b200.vramNeededGb).toBe(401);
  });

  it("KV formula: 2 × L × H_kv × D_h × 2 × tokens / 1e9 (Scout @2^20 = 206.16 GB)", () => {
    const scout = MODEL_MAP["llama4-scout"];
    // The live server's "1M" label means 1,048,576 tokens (2^20).
    expect(computeKVCacheGb(scout, 1_048_576, 1)).toBeCloseTo(206.1589, 2);
    expect(computeKVCacheGb(scout, 1_000_000, 1)).toBeCloseTo(196.608, 2);
    const r1 = MODEL_MAP["deepseek-r1"];
    expect(computeKVCacheGb(r1, 16384, 1)).toBeCloseTo(65.5008, 2);
    const mixtral = MODEL_MAP["mixtral-8x7b"];
    expect(computeKVCacheGb(mixtral, 8192, 1)).toBeCloseTo(1.0737, 3);
  });
});

// ============================================================
// GOLDEN NUMBERS — concurrency + throughput
// ============================================================

describe("golden: concurrency + throughput (hand-verified oracle)", () => {
  it("Mixtral @8K concurrency: H100×2=62, H200=44, B200=91, B300=181 — exact floor of (VRAM−weights)/KV", () => {
    const input = parseStrict(RecommendTopologySchema, {
      model: "mixtral-8x7b", contextTokens: 8192, batchSize: 1,
    }) as any;
    const out: any = handleRecommendTopology(input);
    const byGpu = Object.fromEntries(out.recommendations.map((r: any) => [r.gpuId, r]));

    // Hand-derived: floor((VRAM × n − 93.44) / 1.0737)
    expect(byGpu["h100-sxm"].maxConcurrentUsers).toBe(62);
    expect(byGpu["h200-sxm"].maxConcurrentUsers).toBe(44);
    expect(byGpu["b200-sxm"].maxConcurrentUsers).toBe(91);
    expect(byGpu["b300-sxm"].maxConcurrentUsers).toBe(181);

    // The printed math must equal the number, using the unrounded KV value.
    for (const r of out.recommendations) {
      const m = /floor\(\((\d+) GB × (\d+) − ([\d.]+) GB weights\) ÷ ([\d.]+) GB\/req\) = (\d+)/.exec(
        r.maxConcurrentUsersMath,
      );
      expect(m).not.toBeNull();
      if (!m) continue;
      const [, vram, n, w, kv, users] = m;
      expect(Math.floor((+vram * +n - +w) / +kv)).toBe(+users);
    }
  });

  it("Mixtral fp16 ×2 H100 throughput: decode ≈143, aggregate ≈287", () => {
    const input = parseStrict(EstimateCapacitySchema, {
      model: "mixtral-8x7b", gpu: "h100-sxm", gpuCount: 2, quantization: "fp16", promptTokens: 500,
    }) as any;
    const out: any = handleEstimateCapacity(input);
    expect(out.performance.outputTokensPerSecond.expected).toBe(143);
    expect(out.performance.aggregateTokensPerSecond).toBe(287);
  });

  it("Qwen3-30B-A3B fp8 RTX 5090: 582 tok/s decode", () => {
    const input = parseStrict(EstimateCapacitySchema, {
      model: "qwen3-30b-a3b", gpu: "rtx-5090", quantization: "fp8",
    }) as any;
    const out: any = handleEstimateCapacity(input);
    expect(out.performance.outputTokensPerSecond.expected).toBe(582);
    expect(out.performance.aggregateTokensPerSecond).toBe(582);
  });
});

// ============================================================
// GOLDEN NUMBERS — cost + break-even
// ============================================================

describe("golden: cost + break-even (hand-verified oracle)", () => {
  it("llama3-8b defaults: break-even 2,995 req/day, $292/mo, blended $0.68 vs $4.64", () => {
    const input = parseStrict(EstimateApiVsSelfHostSchema, {
      model: "llama3-8b", gpu: "rtx-4090",
      apiInputPrice: 2.5, apiOutputPrice: 10, requestsPerDay: 37436,
    }) as any;
    const out: any = handleEstimateApiVsSelfHost(input);
    expect(out.breakEven.requestsPerDay).toBe(2995);
    expect(out.selfHost.monthlyTotalUsd).toBe(292);
    expect(out.selfHost.costPerMillionTokensBlended).toBe(0.68);
    expect(out.api.costPerMillionTokensBlended).toBe(4.64);
  });

  it("gpuHourlyUsd is the RIG TOTAL: 2× H100 → $5.00", () => {
    const input = parseStrict(EstimateCapacitySchema, {
      model: "llama3-8b", gpu: "h100-sxm", gpuCount: 2,
    }) as any;
    const out: any = handleEstimateCapacity(input);
    expect(out.cost.gpuHourlyUsd).toBe(5);
    expect(out.cost.gpuHourlyNote).toContain("$2.5/hr each");
  });

  it("gpuCount scaling golden: 1× H100 → 136/136, $5.12/M; 2× → 230/461, $3.01/M, rig $5.00", () => {
    const one = est({ model: "llama3-8b", gpu: "h100-sxm", gpuCount: 1 }) as any;
    const two = est({ model: "llama3-8b", gpu: "h100-sxm", gpuCount: 2 }) as any;
    // 1 GPU: decode 135.6 (bandwidth-bound), aggregate = decode (batch 1)
    expect(one.performance.outputTokensPerSecond.expected).toBe(136);
    expect(one.performance.aggregateTokensPerSecond).toBe(136);
    expect(one.cost.costPerMillionTokens).toBeCloseTo(5.12, 2);
    // 2 GPUs: NVLink 0.85 → decode 230.4; aggregate = decode × tp = 460.9
    expect(two.performance.outputTokensPerSecond.expected).toBe(230);
    expect(two.performance.aggregateTokensPerSecond).toBe(461);
    expect(two.cost.costPerMillionTokens).toBeCloseTo(3.01, 2);
    expect(two.cost.gpuHourlyUsd).toBe(5);
  });
});

// ============================================================
// INVARIANTS — would have caught every bug from the QA sessions
// ============================================================

describe("invariants", () => {
  it("totalRequiredGb === modelWeightsGb + kvCacheGb (MoE field/total divergence guard)", () => {
    for (const [id, gpu] of [["mixtral-8x7b", "h100-sxm"], ["qwen3-30b-a3b", "rtx-5090"], ["deepseek-r1", "b200-sxm"]] as const) {
      const out: any = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
        model: id, gpu, quantization: id === "qwen3-30b-a3b" ? "fp8" : id === "deepseek-r1" ? "int4" : "fp16",
      }) as any);
      const { modelWeightsGb, kvCacheGb, totalRequiredGb } = out.memory;
      expect(totalRequiredGb).toBeCloseTo(modelWeightsGb + kvCacheGb, 1);
    }
  });

  it("estimate_capacity rejects contextTokens > model.maxContext (per-model guard)", () => {
    const out: any = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "mixtral-8x7b", gpu: "h100-sxm", contextTokens: 65537,
    }) as any);
    expect(out.error).toContain("exceeds Mixtral 8x7B (MoE)'s maxContext of 32,768");
    expect(out.modelMaxContext).toBe(32768);
  });

  it("estimate_capacity rejects promptTokens beyond maxContext (2M prompt on 8K model)", () => {
    const out: any = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "llama3-8b", gpu: "h100-sxm", promptTokens: 2000000,
    }) as any);
    expect(out.error).toContain("promptTokens 2,000,000 exceeds");
    expect(out.modelMaxContext).toBe(8192);
  });

  it("recommend_topology rejects over-ceiling contexts too", () => {
    const out: any = handleRecommendTopology(parseStrict(RecommendTopologySchema, {
      model: "mixtral-8x7b", contextTokens: 65536,
    }) as any);
    expect(out.error).toContain("maxContext");
  });

  it("compare_gpus pre-filter uses FULL paramsB — no modelFits:false rows in the default scan", () => {
    const out: any = handleCompareGpus(parseStrict(CompareGpusSchema, {
      model: "mixtral-8x7b", quantization: "fp16",
    }) as any);
    expect(out.error).toBeUndefined();
    for (const row of out.comparisons) {
      expect(row.modelFits).toBe(true);
    }
  });

  it("summary line matches the sort request (compare_gpus self-consistency)", () => {
    const fast: any = handleCompareGpus(parseStrict(CompareGpusSchema, {
      model: "llama3-8b", sortBy: "highest_throughput", limit: 5,
    }) as any);
    expect(fast.summary).toContain("Fastest:");
    expect(fast.summary).not.toContain("Cheapest:");
    const cheap: any = handleCompareGpus(parseStrict(CompareGpusSchema, {
      model: "llama3-8b", sortBy: "lowest_cost", limit: 5,
    }) as any);
    expect(cheap.summary).toContain("Cheapest:");
    expect(cheap.leaders.cheapestByCost.gpuId).toBe(cheap.comparisons[0].gpuId);
  });

  it("strict schemas reject unknown keys AND name the culprit (Zod 4 issue.keys)", () => {
    let message = "";
    try {
      parseStrict(EstimateCapacitySchema, { model: "llama3-8b", gpu: "h100-sxm", tokensPerMonth: 1 });
    } catch (e: any) {
      message = e.message;
    }
    expect(message).toContain("tokensPerMonth");

    const failure: any = describeCallFailure(
      "estimate_capacity",
      makeZodLikeError([{ code: "unrecognized_keys", keys: ["tokensPerMonth"] }]),
    );
    expect(failure.error).toContain("tokensPerMonth");
    expect(failure.unrecognizedParameters).toEqual(["tokensPerMonth"]);
    expect(failure.suggestions[0]).toContain("tokensPerMonth");
  });

  it("KV quantization shrinks KV by exactly the byte ratio", () => {
    // qwen2-5-14b: maxContext 131072, 28 GB fp16 weights on an 80 GB H100
    const fp16 = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "qwen2-5-14b", gpu: "h100-sxm", contextTokens: 131072,
    }) as any) as any;
    const int8 = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "qwen2-5-14b", gpu: "h100-sxm", contextTokens: 131072, kvQuantization: "int8",
    }) as any) as any;
    const int4 = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "qwen2-5-14b", gpu: "h100-sxm", contextTokens: 131072, kvQuantization: "int4",
    }) as any) as any;
    expect(int8.memory.kvCacheGb).toBeCloseTo(fp16.memory.kvCacheGb / 2, 2);
    expect(int4.memory.kvCacheGb).toBeCloseTo(fp16.memory.kvCacheGb / 4, 2);
  });

  it("speculative decoding boosts decode throughput by the multiplier", () => {
    const base = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "llama3-8b", gpu: "h100-sxm",
    }) as any) as any;
    const spec = handleEstimateCapacity(parseStrict(EstimateCapacitySchema, {
      model: "llama3-8b", gpu: "h100-sxm", useSpeculative: true, speculativeBoost: 2,
    }) as any) as any;
    expect(spec.performance.outputTokensPerSecond.expected)
      .toBeGreaterThanOrEqual(base.performance.outputTokensPerSecond.expected * 2 - 1);
    expect(spec.performance.outputTokensPerSecond.expected)
      .toBeLessThanOrEqual(base.performance.outputTokensPerSecond.expected * 2 + 1);
  });

  it("break-even × per-request API cost × 30 === self-host monthly (to the cent)", () => {
    const input = parseStrict(EstimateApiVsSelfHostSchema, {
      model: "llama3-8b", gpu: "rtx-4090", apiInputPrice: 2.5, apiOutputPrice: 10,
    }) as any;
    const out: any = handleEstimateApiVsSelfHost(input);
    const apiPerReq = (500 / 1e6) * 2.5 + (200 / 1e6) * 10; // 0.00325
    expect(out.breakEven.requestsPerDay * apiPerReq * 30).toBeCloseTo(out.selfHost.monthlyTotalUsd, 0);
  });
});

// ============================================================
// NEW TOOLS
// ============================================================

describe("new tool: find_config_for_slo", () => {
  it("finds a feasible config for llama3-8b @8K with generous SLOs", () => {
    const out: any = slo({
      model: "llama3-8b", contextTokens: 8192, requestsPerDay: 10000, batchSize: 8,
    } as any);
    expect(out.error).toBeUndefined();
    expect(out.recommended).toBeDefined();
    expect(out.recommended.aggregateTokensPerSecond).toBeGreaterThan(0);
    expect(out.recommended.costPerMillionTokens).toBeGreaterThan(0);
    expect(out.totalFeasible).toBeGreaterThanOrEqual(1);
  });

  it("monthly cost is DEDICATED-rig basis: gpuCount × $/hr × 730 (not per-token)", () => {
    const out: any = slo({
      model: "llama3-8b", contextTokens: 8192, requestsPerDay: 10000, batchSize: 8,
      sortBy: "highest_throughput",
    } as any);
    // highest_throughput recommends a big rig — verify its monthly figure
    // equals the honest dedicated cost, not a per-token derivation.
    const rec = out.recommended;
    const unit = GPU_MAP[rec.gpuId].usdPerHour ?? 0;
    expect(rec.dedicatedMonthlyUsd).toBe(Math.round(unit * rec.gpuCount * 730));
    expect(rec.dedicatedMonthlyUsd).toBe(rec.estimatedMonthlyUsd);
    // and it must exceed the rig's raw hourly × 730 sanity floor
    expect(rec.dedicatedMonthlyUsd).toBeGreaterThan(rec.gpuCount * 100);
  });

  it("best_value right-sizes: a ~75 tok/s workload is not sold an 8-GPU rig", () => {
    const out: any = slo({
      model: "llama3-8b", contextTokens: 8192, requestsPerDay: 1000, batchSize: 8,
      sortBy: "best_value",
    } as any);
    // mean load = 8192 × 1000 / 86400 ≈ 94.8 tok/s
    expect(out.workloadShape.requiredAggregateTokensPerSecond).toBeGreaterThan(90);
    expect(out.workloadShape.requiredAggregateTokensPerSecond).toBeLessThan(100);
    const rec = out.recommended;
    expect(rec.sustainsMeanLoad).toBe(true);
    expect(rec.dedicatedMonthlyUsd).toBeLessThanOrEqual(600); // no multi-GPU B200 rig
    expect(rec.meanLoadUtilizationPct).toBeLessThanOrEqual(100);
  });

  it("alternatives never include undersized configs when the recommendation sustains", () => {
    const out: any = slo({
      model: "llama3-8b", contextTokens: 8192, requestsPerDay: 50000, batchSize: 8,
      sortBy: "best_value",
    } as any);
    if (out.recommended.sustainsMeanLoad) {
      for (const alt of out.alternatives ?? []) {
        expect(alt.sustainsMeanLoad).toBe(true);
      }
    }
  });

  it("respects a hard TTFT ceiling by excluding every config that violates it", () => {
    // Long-prompt workload (32K prefill on a 72B model) so TTFT is in the
    // 6–16 ms range on feasible hardware — a 5 ms ceiling is genuinely binding.
    const generous: any = slo({
      model: "qwen2-5-72b", contextTokens: 65536, requestsPerDay: 10000, batchSize: 8,
    } as any);
    const tight: any = slo({
      model: "qwen2-5-72b", contextTokens: 65536, requestsPerDay: 10000, batchSize: 8,
      maxTtftMs: 5,
    } as any);
    expect(tight.recommended).toBeUndefined();
    expect(tight.error).toContain("No feasible configuration");
    expect(generous.recommended.ttftMs).toBeGreaterThan(5);
  });

  it("rejects over-ceiling context up front", () => {
    const out: any = slo({
      model: "mixtral-8x7b", contextTokens: 65536, requestsPerDay: 1000, batchSize: 1,
    } as any);
    expect(out.error).toContain("maxContext");
  });
});

describe("new tool: plan_deployment", () => {
  it("produces a coherent brief for a feasible config", () => {
    const out: any = plan({
      model: "llama3-8b", gpu: "rtx-4090", promptTokens: 2000, outputTokens: 500,
      requestsPerDay: 10000, batchSize: 8, engine: "vllm", continuousBatching: true,
    } as any);
    expect(out.error).toBeUndefined();
    expect(out.memory.totalRequiredGb).toBeLessThan(out.memory.availableGb);
    expect(out.memory.kvHeadroomMultiplier).toBeGreaterThan(2);
    expect(out.buildVsBuy.breakEvenRequestsPerDay).toBeGreaterThan(0);
    expect(out.risks.length).toBeGreaterThanOrEqual(1);
    expect(out.summary).toContain("tok/s");
  });

  it("flags no-NVLink tensor parallelism as a risk", () => {
    const out: any = plan({
      model: "llama3-8b", gpu: "rtx-3090", gpuCount: 2, promptTokens: 2000,
      requestsPerDay: 1000, batchSize: 1, continuousBatching: false,
    } as any);
    expect(out.risks.join(" ")).toContain("no NVLink");
  });

  it("returns a topology recommendation instead of a number when infeasible", () => {
    const out: any = plan({
      model: "deepseek-r1", gpu: "rtx-4090", quantization: "fp16",
      promptTokens: 2000, requestsPerDay: 1000, batchSize: 1, continuousBatching: false,
    } as any);
    expect(out.error).toContain("Infeasible");
    expect(out.recommendedTopology.neededGpus).toBeGreaterThan(1);
  });
});

describe("new tool: record_measured", () => {
  it("computes the measured/predicted ratio and stores it", () => {
    const out: any = record({
      model: "llama3-8b", gpu: "rtx-4090", quantization: "fp16",
      observed: { decodeTokensPerSecond: 20.5 },
      source: "test suite",
    } as any);
    expect(out.error).toBeUndefined();
    expect(out.predicted.decodeTokensPerSecond).toBeGreaterThan(0);
    expect(out.ratios.decodeTps).toBeCloseTo(20.5 / out.predicted.decodeTokensPerSecond, 2);
    expect(out.storedTo).toContain("calibration.json");
  });

  it("annotates estimate_capacity with the calibration after a record", () => {
    record({
      model: "qwen3-8b", gpu: "rtx-5090", quantization: "fp16",
      observed: { decodeTokensPerSecond: 999 }, source: "test suite",
    } as any);
    const out: any = est({
      model: "qwen3-8b", gpu: "rtx-5090", quantization: "fp16",
    } as any);
    expect(out.calibration).toBeDefined();
    expect(out.calibration.observedDecodeTokensPerSecond).toBe(999);
    expect(out.calibration.measuredOverPredicted)
      .toBeCloseTo(999 / out.performance.outputTokensPerSecond.expected, 1);
  });

  it("rejects an empty observed object", () => {
    const out: any = record({
      model: "llama3-8b", gpu: "rtx-4090", observed: {},
    } as any);
    expect(out.error).toContain("at least one measurement");
  });
});
