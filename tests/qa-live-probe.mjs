/**
 * Tiered live QA probe — runs the PUBLISHED npm tarball over stdio.
 *
 *   node tests/qa-live-probe.mjs            (tests @tokcalc/mcp-server@0.2.7)
 *   node tests/qa-live-probe.mjs 0.2.7      (explicit version)
 *
 * Tiers:
 *   1 easy        handshake, tool inventory, happy paths
 *   2 medium      golden numbers from hand-verified QA oracle + guards
 *   3 hard        the 4 new tools end-to-end (incl. live HuggingFace fetch)
 *   4 adversarial protocol abuse, boundary values, cross-tool consistency
 *
 * Exit code 0 iff every check passes.
 */

import { spawn } from "node:child_process";

const VERSION = process.argv[2] || "0.2.7";
const LOCAL = VERSION === "local"; // probe the freshly built dist instead of npm
// In local mode the built server reports its real semver; npm mode pins the exact tarball version.
const versionMatches = (v) => (LOCAL ? /^\d+\.\d+\.\d+$/.test(v) : v === VERSION);
let nextId = 1;
let pending = new Map();
let child = null;

function start() {
  const cmd = LOCAL
    ? "node mini-services/mcp-server/dist/index.js"
    : `npx -y @tokcalc/mcp-server@${VERSION}`;
  child = spawn(cmd, { shell: true });
  let buf = "";
  child.stdout.on("data", (d) => {
    buf += d.toString();
    let idx;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.id && pending.has(msg.id)) {
          pending.get(msg.id)(msg);
          pending.delete(msg.id);
        }
      } catch { /* non-JSON line — tolerate */ }
    }
  });
  child.stderr.on("data", () => {}); // server logs go to stderr; ignore
}

function send(method, params, timeoutMs = 45000) {
  const id = nextId++;
  const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`timeout after ${timeoutMs}ms: ${method}`));
    }, timeoutMs);
    pending.set(id, (msg) => { clearTimeout(t); resolve(msg); });
    child.stdin.write(payload);
  });
}

async function initialize() {
  await send("initialize", {
    protocolVersion: "2025-03-26",
    capabilities: {},
    clientInfo: { name: "tokcalc-qa", version: "1.0.0" },
  }, 90000);
  child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");
}

/** Call a tool; returns {raw, isError, content} where content is parsed JSON when possible. */
async function call(tool, args) {
  const res = await send("tools/call", { name: tool, arguments: args });
  const isError = res.result?.isError === true;
  let content = null;
  try { content = JSON.parse(res.result?.content?.[0]?.text ?? "null"); } catch { /* leave null */ }
  return { raw: res, isError, content };
}

// ---------- harness ----------
const results = [];
function check(tier, name, fn) {
  return (async () => {
    try {
      await fn();
      results.push({ tier, name, ok: true });
      console.log(`  PASS  [T${tier}] ${name}`);
    } catch (e) {
      results.push({ tier, name, ok: false, err: e.message });
      console.log(`  FAIL  [T${tier}] ${name}\n          ${e.message.split("\n")[0]}`);
    }
  })();
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function eq(a, b, msg) { assert(a === b, `${msg ?? "eq"}: expected ${b}, got ${a}`); }
function close(a, b, tol, msg) { assert(Math.abs(a - b) <= tol, `${msg ?? "close"}: expected ~${b} (±${tol}), got ${a}`); }

// ---------- tiers ----------

async function tier1() {
  console.log(`\n── Tier 1 · easy (handshake, inventory, happy paths) ──`);
  await check(1, `initialize reports a valid serverInfo.version${LOCAL ? " (local build)" : ` ${VERSION}`}`, async () => {
    const res = await send("initialize", { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "qa", version: "0" } }, 90000);
    assert(versionMatches(res.result?.serverInfo?.version), `serverInfo.version=${res.result?.serverInfo?.version}`);
  });
  await check(1, "tools/list exposes exactly the 11 expected tools", async () => {
    const res = await send("tools/list", {});
    const names = res.result.tools.map(t => t.name).sort();
    const expected = ["compare_gpus","estimate_api_vs_self_host","estimate_capacity","fetch_model_spec","find_config_for_slo","get_mlperf_benchmarks","list_gpus","list_models","plan_deployment","record_measured","recommend_topology"].sort();
    eq(JSON.stringify(names), JSON.stringify(expected), "tool names");
  });
  await check(1, "estimate_capacity happy path returns full structure", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm" });
    eq(isError, false, "isError");
    assert(content.memory && content.performance && content.cost && content.provenance, "missing sections");
    assert(versionMatches(content.provenance.serverVersion), `provenance stamp=${content.provenance.serverVersion}`);
  });
  await check(1, "list_models / list_gpus / get_mlperf_benchmarks regressions", async () => {
    const m = await call("list_models", { isMoE: true });
    eq(m.content.count, 10, "MoE model count");
    const g = await call("list_gpus", { category: "consumer" });
    eq(g.content.count, 5, "consumer GPU count");
    const b = await call("get_mlperf_benchmarks", { gpuModel: "H100" });
    eq(b.content.count, 3, "H100 MLPerf rows");
  });
}

async function tier2() {
  console.log(`\n── Tier 2 · medium (golden numbers + guards, live tarball) ──`);
  await check(2, "Mixtral fp16 ×2 H100: 93.4 / 25.8 / 93.47 / 287", async () => {
    const { content } = await call("estimate_capacity", { model: "mixtral-8x7b", gpu: "h100-sxm", gpuCount: 2, quantization: "fp16", promptTokens: 500 });
    eq(content.memory.modelWeightsGb, 93.4, "weights");
    eq(content.memory.activeParamsGb, 25.8, "active");
    eq(content.memory.totalRequiredGb, 93.47, "total");
    eq(content.performance.aggregateTokensPerSecond, 287, "aggregate");
  });
  await check(2, "DeepSeek R1 int4 topology: 335.5 + 65.5 = 401", async () => {
    const { content } = await call("recommend_topology", { model: "deepseek-r1", quantization: "int4", contextTokens: 16384, batchSize: 1 });
    const b200 = content.recommendations.find(r => r.gpuId === "b200-sxm");
    assert(b200, "b200 row missing");
    eq(b200.residentWeightsGb, 335.5, "weights");
    eq(b200.kvPerRequestGb, 65.5, "kv/req");
    eq(b200.vramNeededGb, 401, "total");
  });
  await check(2, "Concurrency oracle 62/44/91/181 + printed math reconciles", async () => {
    const { content } = await call("recommend_topology", { model: "mixtral-8x7b", contextTokens: 8192, batchSize: 1 });
    const byGpu = Object.fromEntries(content.recommendations.map(r => [r.gpuId, r]));
    eq(byGpu["h100-sxm"].maxConcurrentUsers, 62, "h100×2");
    eq(byGpu["h200-sxm"].maxConcurrentUsers, 44, "h200");
    eq(byGpu["b200-sxm"].maxConcurrentUsers, 91, "b200");
    eq(byGpu["b300-sxm"].maxConcurrentUsers, 181, "b300");
    for (const r of content.recommendations) {
      const m = /floor\(\((\d+) GB × (\d+) − ([\d.]+) GB weights\) ÷ ([\d.]+) GB\/req\) = (\d+)/.exec(r.maxConcurrentUsersMath);
      assert(m, `math unparsable for ${r.gpuId}`);
      eq(Math.floor((+m[1] * +m[2] - +m[3]) / +m[4]), +m[5], `math consistency ${r.gpuId}`);
    }
  });
  await check(2, "Break-even oracle: 2995 req/day, $292/mo, blended 0.68 vs 4.64", async () => {
    const { content } = await call("estimate_api_vs_self_host", { model: "llama3-8b", gpu: "rtx-4090", apiInputPrice: 2.5, apiOutputPrice: 10, requestsPerDay: 37436 });
    eq(content.breakEven.requestsPerDay, 2995, "break-even");
    eq(content.selfHost.monthlyTotalUsd, 292, "monthly");
    eq(content.selfHost.costPerMillionTokensBlended, 0.68, "self-host blended");
    eq(content.api.costPerMillionTokensBlended, 4.64, "api blended");
  });
  await check(2, "gpuHourlyUsd is the rig total + note explains it", async () => {
    const { content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", gpuCount: 2 });
    eq(content.cost.gpuHourlyUsd, 5, "rig hourly");
    assert(content.cost.gpuHourlyNote.includes("$2.5/hr each"), "note missing unit price");
  });
  await check(2, "maxContext guard: mixtral @65537 rejected with modelMaxContext", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "mixtral-8x7b", gpu: "h100-sxm", contextTokens: 65537 });
    eq(isError, true, "isError");
    eq(content.modelMaxContext, 32768, "modelMaxContext");
  });
  await check(2, "Strict schema: bogus key named in error + suggestion", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", tokensPerMonth: 5 });
    eq(isError, true, "isError");
    assert((content.unrecognizedParameters || []).includes("tokensPerMonth"), `culprit not named: ${JSON.stringify(content.error)}`);
    assert((content.validParameters || []).includes("promptTokens"), "valid list missing");
  });
  await check(2, "Invalid enum + unknown model rejected cleanly", async () => {
    const a = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", quantization: "int3" });
    eq(a.isError, true, "enum not rejected");
    const b = await call("estimate_capacity", { model: "gpt-99-turbo", gpu: "h100-sxm" });
    eq(b.isError, true, "unknown model not rejected");
    assert(b.content.error.includes("Unknown model"), "wrong unknown-model message");
  });
  await check(2, "KV quantization: int4 KV = fp16 KV ÷ 4", async () => {
    const fp16 = await call("estimate_capacity", { model: "qwen2-5-14b", gpu: "h100-sxm", contextTokens: 32768 });
    const q4 = await call("estimate_capacity", { model: "qwen2-5-14b", gpu: "h100-sxm", contextTokens: 32768, kvQuantization: "int4" });
    close(q4.content.memory.kvCacheGb, fp16.content.memory.kvCacheGb / 4, 0.01, "int4/fp16 ratio");
  });
  await check(2, "Infeasible config: zeroed throughput, blocking reason, null cost (mixtral fp16 on one 80GB H100)", async () => {
    const { content } = await call("estimate_capacity", { model: "mixtral-8x7b", gpu: "h100-sxm", promptTokens: 500 });
    eq(content.feasibility.modelFits, false, "modelFits");
    eq(content.performance.outputTokensPerSecond.expected, 0, "throughput zeroed");
    eq(content.cost.costPerMillionTokens, null, "cost null");
  });
}

async function tier3() {
  console.log(`\n── Tier 3 · hard (4 new tools end-to-end) ──`);
  await check(3, "find_config_for_slo: feasible set + recommendation fields", async () => {
    const { isError, content } = await call("find_config_for_slo", { model: "llama3-8b", contextTokens: 8192, requestsPerDay: 10000, batchSize: 8 });
    eq(isError, false, "isError");
    assert(content.totalFeasible >= 1, `only ${content.totalFeasible} feasible`);
    assert(content.recommended.costPerMillionTokens > 0 && content.recommended.aggregateTokensPerSecond > 0, "bad recommendation");
    assert(content.workloadShape.tokensPerRequest === content.workloadShape.prefillTokens + content.workloadShape.outputTokens, "workload shape inconsistent");
  });
  await check(3, "find_config_for_slo: impossible ceiling → clean no-fit with rejectedBy", async () => {
    const { isError, content } = await call("find_config_for_slo", { model: "llama3-8b", contextTokens: 8192, requestsPerDay: 10000, batchSize: 8, maxCostPerMillion: 0.01 });
    eq(isError, true, "isError");
    assert(content.error.includes("No feasible configuration"), "wrong error");
    assert(content.rejectedBy && typeof content.rejectedBy.exceededMaxCost === "number", "rejectedBy missing");
  });
  await check(3, "plan_deployment: coherent brief + hand-checkable break-even", async () => {
    const { content } = await call("plan_deployment", { model: "llama3-8b", gpu: "rtx-4090", promptTokens: 2000, outputTokens: 500, requestsPerDay: 10000, batchSize: 8, engine: "vllm", continuousBatching: true });
    assert(content.memory.kvHeadroomMultiplier >= 1, "headroom missing");
    const perReq = (2000 / 1e6) * 0.15 + (500 / 1e6) * 0.6;
    const rigMonthly = 0.4 * 730;
    eq(content.buildVsBuy.breakEvenRequestsPerDay, Math.round(rigMonthly / (30 * perReq)), "break-even math");
    assert(content.risks.length >= 1 && content.nextSteps.length >= 3, "brief incomplete");
  });
  await check(3, "plan_deployment: infeasible → topology recommendation, not a number", async () => {
    const { isError, content } = await call("plan_deployment", { model: "deepseek-r1", gpu: "rtx-4090", promptTokens: 2000, requestsPerDay: 1000 });
    eq(isError, true, "isError");
    assert(content.recommendedTopology.neededGpus > 1, "topology missing");
  });
  await check(3, "fetch_model_spec: LIVE HuggingFace fetch + zero drift on qwen3-32b", async () => {
    const { isError, content } = await call("fetch_model_spec", { modelId: "qwen3-32b", forceRefresh: true });
    eq(isError, false, "isError");
    assert(content.source === "huggingface" || content.source === "cache", `source=${content.source}`);
    eq(content.hfMapped.layers, 64, "layers from HF (Qwen3-32B is genuinely 64-layer)");
    assert(content.comparison.driftedCount === 0, `unexpected drift: ${JSON.stringify(content.comparison.driftedFields)}`);
  });
  await check(3, "fetch_model_spec: bad hfRepo → clean error, no crash", async () => {
    const { isError, content } = await call("fetch_model_spec", { modelId: "qwen3-32b", hfRepo: "definitely/not-a-real-repo-xyz", forceRefresh: true });
    eq(isError, true, "isError");
    assert(content.error.includes("404") || content.error.includes("Failed"), `unexpected: ${content.error}`);
  });
  await check(3, "record_measured → estimate_capacity shows calibration annotation", async () => {
    const rec = await call("record_measured", { model: "gemma2-9b", gpu: "rtx-3090", observed: { decodeTokensPerSecond: 17.5 }, source: "qa probe" });
    eq(rec.isError, false, "record failed");
    assert(rec.content.ratios.decodeTps > 0 && rec.content.ratios.decodeTps !== 1, "ratio missing");
    const est = await call("estimate_capacity", { model: "gemma2-9b", gpu: "rtx-3090" });
    assert(est.content.calibration && est.content.calibration.measuredOverPredicted > 0, "annotation missing");
    close(est.content.calibration.measuredOverPredicted, rec.content.ratios.decodeTps, 0.01, "ratio mismatch");
  });
  await check(3, "record_measured: empty observed rejected", async () => {
    const { isError, content } = await call("record_measured", { model: "llama3-8b", gpu: "h100-sxm", observed: {} });
    eq(isError, true, "isError");
    assert(content.error.includes("at least one"), "wrong message");
  });
}

async function tier4() {
  console.log(`\n── Tier 4 · adversarial (boundaries, abuse, cross-tool) ──`);
  await check(4, "Unknown tool name → clean isError", async () => {
    const { isError } = await call("definitely_not_a_tool", {});
    eq(isError, true, "isError");
  });
  await check(4, "arguments:null → schema error, no crash", async () => {
    const res = await send("tools/call", { name: "list_models", arguments: null });
    assert(res.result !== undefined || res.error !== undefined, "no response");
  });
  await check(4, "Malformed JSON line does not kill the server", async () => {
    child.stdin.write("THIS IS NOT JSON AT ALL\n");
    await new Promise(r => setTimeout(r, 300));
    const res = await send("tools/list", {});
    assert(res.result.tools.length === 11, "server broken after garbage input");
  });
  await check(4, "Pipelined batch of 3 requests → 3 ordered responses", async () => {
    const ids = [nextId++, nextId++, nextId++];
    const proms = ids.map(id => new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("pipeline timeout")), 20000);
      pending.set(id, (msg) => { clearTimeout(t); resolve(msg); });
    }));
    // write AFTER arming the listeners (the previous version awaited first — deadlock)
    child.stdin.write(ids.map(id => JSON.stringify({ jsonrpc: "2.0", id, method: "tools/list", params: {} })).join("\n") + "\n");
    const msgs = await Promise.all(proms);
    msgs.forEach((m, i) => eq(m.id, ids[i], `pipelined ${i}`));
  });
  await check(4, "Boundary: gpuCount=256 accepted (80GB×256 rig)", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", gpuCount: 256 });
    eq(isError, false, "isError");
    eq(content.memory.availableGb, 20480, "available");
  });
  await check(4, "Boundary: batchSize=10000 on 8B model → graceful response", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", batchSize: 10000 });
    assert(content && content.provenance, "no structured response");
    assert(typeof content.performance.aggregateTokensPerSecond === "number", "aggregate not a number");
  });
  await check(4, "promptTokens beyond maxContext rejected (2M prompt on 8K model)", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", promptTokens: 2000000 });
    eq(isError, true, "isError");
    eq(content.modelMaxContext, 8192, "modelMaxContext");
  });
  await check(4, "recommend_topology batchSize=1000 on 671B → graceful zero options", async () => {
    const { content } = await call("recommend_topology", { model: "deepseek-v3", contextTokens: 8192, batchSize: 1000 });
    assert(Array.isArray(content.recommendations), "recommendations not an array");
    eq(content.recommendations.length, 0, "expected zero feasible");
  });
  await check(4, "Cross-tool: topology's vramNeeded matches estimate's totalRequired (mixtral 8K b1)", async () => {
    const est = await call("estimate_capacity", { model: "mixtral-8x7b", gpu: "h100-sxm", contextTokens: 8192 });
    const topo = await call("recommend_topology", { model: "mixtral-8x7b", contextTokens: 8192, batchSize: 1 });
    const h200 = topo.content.recommendations.find(r => r.gpuId === "h200-sxm");
    close(h200.vramNeededGb, est.content.memory.totalRequiredGb, 0.2, "cross-tool total");
  });
  await check(4, "Cross-tool: compare_gpus row equals estimate for the same GPU (llama3-8b/mi300x)", async () => {
    const est = await call("estimate_capacity", { model: "llama3-8b", gpu: "mi300x", batchSize: 8 });
    const cmp = await call("compare_gpus", { model: "llama3-8b", batchSize: 8, gpus: ["mi300x"], sortBy: "highest_throughput" });
    const row = cmp.content.comparisons[0];
    eq(row.outputTokensPerSecond, est.content.performance.outputTokensPerSecond.expected, "decode mismatch");
    close(row.costPerMillionTokens, +est.content.cost.costPerMillionTokens.toFixed(2), 0.01, "cost mismatch");
  });
  await check(4, "Prefix caching params accepted and reflected in assumptions", async () => {
    const { isError, content } = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", promptTokens: 1000, cachePrefixTokens: 5000, cacheHitRate: 0.8 });
    eq(isError, false, "isError");
    assert(JSON.stringify(content.assumptions).includes("Prefix caching"), "assumption missing");
  });
  await check(4, "Speculative decoding multiplies decode tok/s", async () => {
    const base = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm" });
    const spec = await call("estimate_capacity", { model: "llama3-8b", gpu: "h100-sxm", useSpeculative: true, speculativeBoost: 2 });
    close(spec.content.performance.outputTokensPerSecond.expected, base.content.performance.outputTokensPerSecond.expected * 2, 1.5, "spec boost");
  });
  await check(4, "fetch_model_spec offline-id: unknown model → helpful error + valid IDs", async () => {
    const { isError, content } = await call("fetch_model_spec", { modelId: "no-such-model" });
    eq(isError, true, "isError");
    assert(Array.isArray(content.validModelIds) && content.validModelIds.length > 0, "valid list missing");
  });
}

// ---------- main ----------
const overallStart = Date.now();
start();
try {
  await initialize();
  await tier1();
  await tier2();
  await tier3();
  await tier4();
} catch (e) {
  console.error(`\nFATAL: ${e.message}`);
  process.exitCode = 2;
} finally {
  child?.kill();
}

const passed = results.filter(r => r.ok).length;
const failed = results.length - passed;
console.log(`\n${"═".repeat(62)}`);
console.log(`RESULT: ${passed}/${results.length} checks passed  (${failed} failed)  in ${((Date.now() - overallStart) / 1000).toFixed(1)}s`);
for (const t of [1, 2, 3, 4]) {
  const rs = results.filter(r => r.tier === t);
  if (rs.length) console.log(`  Tier ${t}: ${rs.filter(r => r.ok).length}/${rs.length}`);
}
process.exitCode = failed > 0 ? 1 : 0;
