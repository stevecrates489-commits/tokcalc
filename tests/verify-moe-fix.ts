/**
 * Numeric check for the MoE modelWeightsGb fix.
 *
 * Bug: `modelSizeGb` reported activeParamsB × bytes, but VRAM is consumed by
 * paramsB × bytes (every MoE expert must be resident). The totals were right;
 * only the reported field was wrong, which silently broke any consumer that
 * divides by it.
 *
 * Mixtral-8x7B: paramsB 46.7, activeParamsB 12.9, fp16 (2 B/param)
 * Qwen3-30B-A3B: paramsB 30.5, activeParamsB 3.3, fp8 (1 B/param)
 */

import { calculate, MODELS, MODEL_MAP } from "../src/lib/token-calc.ts";

const cases = [
  { id: "mixtral-8x7b", gpu: "h100-sxm", gpuCount: 2, quant: "fp16", ctx: 8192, batch: 1 },
  { id: "qwen3-30b-a3b", gpu: "rtx-5090", gpuCount: 1, quant: "fp8", ctx: 8192, batch: 1 },
  { id: "llama3-70b", gpu: "h100-sxm", gpuCount: 2, quant: "fp8", ctx: 32768, batch: 8 },
];

let failures = 0;

for (const c of cases) {
  const model = MODEL_MAP[c.id];
  if (!model) {
    console.log(`  SKIP   ${c.id} (not in catalog)`);
    continue;
  }

  const r = calculate({
    modelId: c.id,
    gpuId: c.gpu,
    quantization: c.quant,
    numGpus: c.gpuCount,
    batchSize: c.batch,
    promptTokens: c.ctx,
    outputTokens: 128,
  });

  const bytes = c.quant === "fp16" ? 2 : c.quant === "fp8" ? 1 : 1;
  const expectResident = model.paramsB * bytes;
  const expectActive = model.activeParamsB * bytes;
  const expectTotal = expectResident + r.kvCacheTotalGb;

  const okWeights = Math.abs(r.modelSizeGb - expectResident) < 0.05;
  const okActive = Math.abs(r.activeParamsGb - expectActive) < 0.05;
  const okTotal = Math.abs(r.totalVramNeededGb - expectTotal) < 0.05;
  const okMoE = r.isMoE === model.isMoE;

  if (!(okWeights && okActive && okTotal && okMoE)) failures++;

  console.log(`  ${c.id} (${model.isMoE ? "MoE" : "dense"}) on ${c.gpuCount}× ${c.gpu} ${c.quant}`);
  console.log(`    paramsB=${model.paramsB}B  activeParamsB=${model.activeParamsB}B`);
  console.log(`    modelWeightsGb  = ${r.modelSizeGb.toFixed(2)}  (expect resident ${expectResident.toFixed(2)})  ${okWeights ? "OK" : "MISMATCH"}`);
  console.log(`    activeParamsGb  = ${r.activeParamsGb.toFixed(2)}  (expect active   ${expectActive.toFixed(2)})  ${okActive ? "OK" : "MISMATCH"}`);
  console.log(`    kvCacheGb       = ${r.kvCacheTotalGb.toFixed(2)}`);
  console.log(`    totalRequiredGb = ${r.totalVramNeededGb.toFixed(2)}  (expect ${expectTotal.toFixed(2)})  ${okTotal ? "OK" : "MISMATCH"}`);
  console.log(`    isMoE           = ${r.isMoE}  ${okMoE ? "OK" : "MISMATCH"}`);

  if (model.isMoE) {
    const ratio = r.modelSizeGb / r.activeParamsGb;
    console.log(`    resident/active = ${ratio.toFixed(2)}×  (capacity must use the former, speed the latter)`);
  }
  console.log("");
}

console.log(failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
