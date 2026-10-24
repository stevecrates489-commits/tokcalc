/**
 * Catalog module tests.
 *
 * Three things are worth locking down here:
 *
 *   1. Every record in catalog/*.json satisfies its Zod schema. The CI job
 *      runs scripts/catalog-validate.ts, which does the same thing — this
 *      suite is the fast local feedback loop and the place where a schema
 *      change is caught by a failing assertion rather than a log line.
 *
 *   2. The architecture-aware KV engine is CORRECT, not just different from
 *      the old formula. Each attention kind gets a hand-computed expectation
 *      so a refactor cannot silently change the numbers.
 *
 *   3. Backward compatibility: a model with no declared attention layout
 *      must produce EXACTLY the numbers the old generic formula produced.
 *      Every pre-existing catalog entry depends on this.
 *
 * Run with: bun test tests/catalog.test.ts
 */

/// <reference types="bun-types" />
import { describe, expect, it } from "bun:test";

import {
  MODELS,
  MODEL_MAP,
  ACCELERATORS,
  QUANTIZATIONS,
  CATALOG_VERSION,
  getModel,
  activeModels,
  modelFamilies,
} from "../src/catalog/loader";
import {
  AcceleratorRecordSchema,
  ModelRecordSchema,
  QuantizationRecordSchema,
} from "../src/catalog/schemas";
import {
  calculate,
  computeKVBytesPerToken,
  computeStateBytes,
  computeKVCacheGb,
  FIRST_ANSWER_TOKENS,
  FIRST_ANSWER_METRIC_NOTE,
  type ModelSpec,
} from "../src/lib/token-calc";

// ---------------------------------------------------------------------------
// 1. Schema conformance
// ---------------------------------------------------------------------------

describe("catalog records", () => {
  it("every model record satisfies the Zod schema", () => {
    const failures: string[] = [];
    for (const m of MODELS) {
      const r = ModelRecordSchema.safeParse(m);
      if (!r.success) {
        failures.push(`${m.id}: ${r.error.issues.map((i) => i.message).join("; ")}`);
      }
    }
    expect(failures).toEqual([]);
  });

  it("every accelerator record satisfies the Zod schema", () => {
    const failures: string[] = [];
    for (const a of ACCELERATORS) {
      const r = AcceleratorRecordSchema.safeParse(a);
      if (!r.success) {
        failures.push(`${a.id}: ${r.error.issues.map((i) => i.message).join("; ")}`);
      }
    }
    expect(failures).toEqual([]);
  });

  it("every quantization record satisfies the Zod schema", () => {
    const failures: string[] = [];
    for (const q of QUANTIZATIONS) {
      const r = QuantizationRecordSchema.safeParse(q);
      if (!r.success) {
        failures.push(`${q.id}: ${r.error.issues.map((i) => i.message).join("; ")}`);
      }
    }
    expect(failures).toEqual([]);
  });

  it("model ids are unique and stable (kebab-case, no whitespace)", () => {
    const ids = MODELS.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) {
      expect(id).toMatch(/^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$/);
    }
  });

  it("every record names a provenance source, and every non-unspecified source has a URL", () => {
    for (const m of MODELS) {
      expect(typeof m.provenance.source).toBe("string");
      if (m.provenance.source !== "unspecified") {
        expect(m.provenance.sourceUrl).toBeTruthy();
      }
    }
  });

  it("a model with an attention layout declares how well the engine models it", () => {
    for (const m of MODELS) {
      if (m.attentionLayers) {
        expect(m.architectureStatus).toBeDefined();
        expect(m.attentionLayers.length).toBe(m.layers);
      }
    }
  });

  it("partial or lower-bound applicability always states at least one limitation", () => {
    // Provenance is source-of-information, not a confidence score. If a record
    // says "only partly holds" it must say WHICH part — otherwise the reader
    // cannot tell what to discount.
    for (const m of MODELS) {
      const p = m.provenance;
      if (p.applicability === "partial" || p.applicability === "lower-bound") {
        expect(p.limitations).toBeDefined();
        expect(p.limitations!.length).toBeGreaterThan(0);
        for (const limitation of p.limitations!) {
          expect(limitation.length).toBeGreaterThan(0);
        }
      }
    }
  });

  it("a memory-only architecture is reported as a lower bound, not as an exact figure", () => {
    // The memory number for a partially-modelled architecture UNDERSTATES the
    // real requirement. If applicability said "exact" the reader would treat a
    // floor as a measurement.
    for (const m of MODELS) {
      if (m.architectureStatus === "memory-only") {
        expect(m.provenance.applicability).toBe("lower-bound");
        expect(m.provenance.limitations?.length ?? 0).toBeGreaterThan(0);
      }
    }
  });

  it("applicability is only one of the three declared values", () => {
    for (const m of MODELS) {
      const a = m.provenance.applicability;
      if (a !== undefined) {
        expect(["exact", "partial", "lower-bound"]).toContain(a);
      }
    }
  });

  it("the catalog exposes a version and lookups", () => {
    expect(CATALOG_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
    expect(getModel("llama3-8b")?.name).toBe("Llama 3 8B");
    expect(getModel("does-not-exist")).toBeUndefined();
    expect(activeModels().length).toBeGreaterThan(0);
    expect(modelFamilies().length).toBeGreaterThan(3);
  });
});

// ---------------------------------------------------------------------------
// 2. The architecture-aware KV engine
// ---------------------------------------------------------------------------

describe("architecture-aware KV engine", () => {
  /** A model with no declared layout — the "ordinary full attention" case. */
  const plain: ModelSpec = {
    id: "plain",
    name: "Plain",
    family: "Test",
    category: "text",
    paramsB: 8,
    activeParamsB: 8,
    layers: 4,
    hiddenDim: 4096,
    qHeads: 32,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 32000,
    maxContext: 8192,
    isMoE: false,
  };

  it("a model with no layout uses the classic formula exactly", () => {
    // 2 (K&V) × 4 layers × 8 kvHeads × 128 headDim × 2 bytes = 16384 B/token
    expect(computeKVBytesPerToken(plain, 2)).toBe(16384);
  });

  it("a full-attention layer costs 2 × kvHeads × headDim × bytes", () => {
    const m: ModelSpec = {
      ...plain,
      attentionLayers: [{ kind: "full" }],
    };
    expect(computeKVBytesPerToken(m, 2)).toBe(2 * 8 * 128 * 2);
  });

  it("a sliding layer's cache saturates at its window", () => {
    const m: ModelSpec = {
      ...plain,
      attentionLayers: [{ kind: "sliding", windowTokens: 128 }],
    };
    const full = 2 * 8 * 128 * 2; // 4096 B/token
    // Below the window: linear.
    expect(computeKVBytesPerToken(m, 2, 64)).toBe(full);
    // At the window: the full rate.
    expect(computeKVBytesPerToken(m, 2, 128)).toBe(full);
    // Above the window: the effective rate collapses, because the cache
    // holds at most `window` entries however long the context gets.
    expect(computeKVBytesPerToken(m, 2, 12800)).toBeCloseTo(full * (128 / 12800), 6);
    // And the total cache size stops growing.
    expect(computeKVCacheGb(m, 128, 1)).toBeCloseTo(computeKVCacheGb(m, 12800, 1), 6);
  });

  it("an MLA layer stores a compressed latent, not a key/value pair", () => {
    const m: ModelSpec = {
      ...plain,
      attentionLayers: [{ kind: "mla", kvLoraRank: 512, mlaRopeHeadDim: 64 }],
    };
    // (512 latent + 64 rope) × 2 bytes = 1152 B/token — far below the
    // 4096 B/token a full GQA layer would cost for the same model.
    expect(computeKVBytesPerToken(m, 2)).toBe((512 + 64) * 2);
    expect(computeKVBytesPerToken(m, 2)).toBeLessThan(2 * 8 * 128 * 2);
  });

  it("a linear layer has no per-token cache and holds a fixed state", () => {
    const m: ModelSpec = {
      ...plain,
      attentionLayers: [{ kind: "linear", stateBytes: 4096 }],
    };
    expect(computeKVBytesPerToken(m, 2)).toBe(0);
    expect(computeStateBytes(m)).toBe(4096);
    // The state does not grow with context — only the batch multiplies it.
    expect(computeKVCacheGb(m, 1000, 1)).toBeCloseTo(4096 / 1e9, 12);
    expect(computeKVCacheGb(m, 1_000_000, 1)).toBeCloseTo(4096 / 1e9, 12);
    expect(computeKVCacheGb(m, 1000, 8)).toBeCloseTo((4096 * 8) / 1e9, 12);
  });

  it("counts every layer of a mixed layout", () => {
    const m: ModelSpec = {
      ...plain,
      attentionLayers: [
        { kind: "full" },
        { kind: "sliding", windowTokens: 128 },
        { kind: "mla", kvLoraRank: 512, mlaRopeHeadDim: 64 },
        { kind: "linear", stateBytes: 1024 },
      ],
    };
    expect(computeKVBytesPerToken(m, 2)).toBe(
      2 * 8 * 128 * 2 + // full
      2 * 8 * 128 * 2 + // sliding (below/at window)
      (512 + 64) * 2 + // mla
      0, // linear
    );
    expect(computeStateBytes(m)).toBe(1024);
  });
});

// ---------------------------------------------------------------------------
// 3. Backward compatibility — the contract for every pre-existing model
// ---------------------------------------------------------------------------

describe("backward compatibility", () => {
  it("a model with no attention layout produces the same KV numbers as before", () => {
    for (const m of MODELS) {
      if (m.attentionLayers) continue;
      const classic = 2 * m.layers * m.kvHeads * m.headDim * 2;
      expect(computeKVBytesPerToken(m, 2)).toBe(classic);
      expect(computeKVCacheGb(m, 8192, 1)).toBeCloseTo((classic * 8192) / 1e9, 9);
    }
  });

  it("the classic models still resolve through MODEL_MAP", () => {
    expect(MODEL_MAP["llama3-8b"].paramsB).toBe(8.03);
    expect(MODEL_MAP["mixtral-8x7b"].isMoE).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 4. The models added from real checkpoints
// ---------------------------------------------------------------------------

describe("checkpoint-derived models", () => {
  it("gpt-oss-20b carries its sliding/full split and a window", () => {
    const m = MODEL_MAP["gpt-oss-20b"];
    expect(m).toBeDefined();
    expect(m.attentionLayers?.length).toBe(24);
    const sliding = m.attentionLayers!.filter((l) => l.kind === "sliding");
    const full = m.attentionLayers!.filter((l) => l.kind === "full");
    expect(sliding.length).toBe(12);
    expect(full.length).toBe(12);
    expect(sliding[0].windowTokens).toBe(128);
    expect(m.nativeWeightFormat).toBe("mxfp4");
    expect(m.architectureStatus).toBe("supported");
  });

  it("Qwen3.8-27b mixes linear and full attention", () => {
    const m = MODEL_MAP["qwen3-8-27b"];
    expect(m).toBeDefined();
    const linear = m.attentionLayers!.filter((l) => l.kind === "linear");
    const full = m.attentionLayers!.filter((l) => l.kind === "full");
    expect(linear.length).toBe(48);
    expect(full.length).toBe(16);
    // Linear layers cost nothing per token.
    expect(computeKVBytesPerToken(m, 2)).toBe(16 * (2 * m.kvHeads * m.headDim * 2));
  });

  it("Kimi K3 mixes KDA and compressed-latent attention", () => {
    const m = MODEL_MAP["kimi-k3"];
    expect(m).toBeDefined();
    const kda = m.attentionLayers!.filter((l) => l.kind === "kda");
    const mla = m.attentionLayers!.filter((l) => l.kind === "mla");
    expect(kda.length).toBe(69);
    expect(mla.length).toBe(24);
    expect(mla[0].kvLoraRank).toBe(512);
    expect(computeStateBytes(m)).toBeGreaterThan(0);
  });

  it("GLM-5.3 is marked memory-only because its sparse-attention index cache is not modelled", () => {
    const m = MODEL_MAP["glm-5-3"];
    expect(m).toBeDefined();
    expect(m.architectureStatus).toBe("memory-only");
    expect(m.provenance.notes).toContain("index cache");
    // The MLA KV cache itself is still modelled.
    expect(computeKVBytesPerToken(m, 2)).toBe(78 * (512 + 64) * 2);
  });

  it("checkpoint-derived models carry their exact resident weight bytes", () => {
    expect(MODEL_MAP["qwen3-8-27b"].residentWeightBytes).toBe(55562855904);
    expect(MODEL_MAP["gpt-oss-20b"].residentWeightBytes).toBe(13761264768);
    expect(MODEL_MAP["gpt-oss-120b"].residentWeightBytes).toBe(65248815744);
    expect(MODEL_MAP["kimi-k3"].residentWeightBytes).toBe(1560860324864);
    expect(MODEL_MAP["glm-5-3"].residentWeightBytes).toBe(755617140416);
  });

  it("checkpoint-derived models are consistent with their parameter count", () => {
    // The safetensors index and the parameter count are two independent
    // measurements of the same checkpoint; they must agree.
    for (const id of ["qwen3-8-27b", "kimi-k3", "gpt-oss-20b", "gpt-oss-120b", "glm-5-3"]) {
      const m = MODEL_MAP[id];
      const bytesPerParam = m.residentWeightBytes! / (m.paramsB * 1e9);
      expect(bytesPerParam).toBeGreaterThan(0.3);
      expect(bytesPerParam).toBeLessThan(2.1);
    }
  });
});

// ---------------------------------------------------------------------------
// 5. The calculator uses the architecture-aware numbers
// ---------------------------------------------------------------------------

describe("calculate() uses the architecture-aware KV", () => {
  it("reports the recurrent state separately from the KV cache", () => {
    const res = calculate({
      modelId: "qwen3-8-27b",
      gpuId: "h100-sxm",
      quantization: "fp16",
      numGpus: 1,
      batchSize: 2,
      promptTokens: 4096,
      outputTokens: 128,
    });
    expect(res.stateBytesGb).toBeGreaterThan(0);
    // The reported per-token rate times the context times the batch, plus
    // the state, is the total conversation memory.
    const expected = (res.kvCachePerTokenKb * 1024 * 4096 * 2) / 1e9 + res.stateBytesGb;
    expect(res.kvCacheTotalGb).toBeCloseTo(expected, 6);
  });

  it("leaves ordinary models with a zero recurrent state", () => {
    const res = calculate({
      modelId: "llama3-8b",
      gpuId: "h100-sxm",
      quantization: "fp16",
      numGpus: 1,
      batchSize: 1,
      promptTokens: 4096,
      outputTokens: 128,
    });
    expect(res.stateBytesGb).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 6. Latency metric semantics
// ---------------------------------------------------------------------------

describe("latency metrics are described honestly", () => {
  const res = calculate({
    modelId: "llama3-8b",
    gpuId: "h100-sxm",
    quantization: "fp16",
    numGpus: 1,
    batchSize: 1,
    promptTokens: 4096,
    outputTokens: 128,
  });

  it("the first-token estimate is more than prefill alone", () => {
    // TTFT = prefill + the first decode step + per-request overhead. Reporting
    // prefill alone was the original bug.
    expect(res.ttftMs).toBeGreaterThan(res.prefillTimeMs);
  });

  it("the readable-output estimate is strictly later than the first token", () => {
    expect(res.firstAnswerLatencyMs).toBeGreaterThan(res.ttftMs);
  });

  it("the readable-output estimate is documented as a token-count proxy, not answer detection", () => {
    // The name "firstAnswerLatencyMs" is kept for API compatibility, but the
    // documented meaning must not imply semantic answer detection.
    expect(FIRST_ANSWER_TOKENS).toBe(12);
    expect(FIRST_ANSWER_METRIC_NOTE).toContain("12 generated tokens");
    expect(FIRST_ANSWER_METRIC_NOTE).toContain("does not identify the semantic start");
    expect(FIRST_ANSWER_METRIC_NOTE).toContain("hidden reasoning");
  });
});
