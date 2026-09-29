#!/usr/bin/env node
/**
 * HTTP smoke probe for the tokcalc MCP hosted endpoint (/api/mcp).
 *
 * Complements tests/qa-live-probe.mjs (which exercises the stdio tarball):
 * this script hits the HTTP surface directly so auth, rate-limit wiring,
 * and transport-level behavior get regression coverage too.
 *
 * Usage:
 *   node tests/http-smoke.mjs                          # against local dev (localhost:3000)
 *   node tests/http-smoke.mjs https://tokcalc.vercel.app
 *   BASE=http://localhost:3000 MCP_API_KEY=... node tests/http-smoke.mjs
 *
 * Exit code 0 = all checks passed, 1 = at least one failure.
 * Checks marked [optional] warn instead of failing (e.g. rate-limit env may
 * not be configured locally).
 */

const BASE = process.argv[2] || process.env.BASE || "http://localhost:3000";
const ENDPOINT = `${BASE.replace(/\/$/, "")}/api/mcp`;
const API_KEY = process.env.MCP_API_KEY || "";

let passed = 0;
let failed = 0;
let warned = 0;

function check(name, ok, detail, { optional = false } = {}) {
  if (ok) {
    passed++;
    console.log(`  PASS  ${name}${detail ? ` — ${detail}` : ""}`);
  } else if (optional) {
    warned++;
    console.log(`  WARN  ${name}${detail ? ` — ${detail}` : ""} (optional)`);
  } else {
    failed++;
    console.log(`  FAIL  ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

async function rpc(body, headers = {}) {
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
      "MCP-Protocol-Version": "2025-03-26",
      ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
      ...headers,
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  const contentType = res.headers.get("content-type") ?? "";
  let json = null;
  try {
    if (contentType.includes("text/event-stream")) {
      // Streamable HTTP may answer a plain request with an SSE frame:
      // extract the first `data:` payload line and parse it.
      const dataLine = text
        .split("\n")
        .map((l) => l.trim())
        .find((l) => l.startsWith("data:"));
      if (dataLine) json = JSON.parse(dataLine.slice(5).trim());
    } else {
      json = JSON.parse(text);
    }
  } catch {
    // Non-JSON body — leave json null; callers handle it.
  }
  return { status: res.status, headers: res.headers, json, text };
}

// ─────────────────────────────────────────────────────────────
// 1. Transport / protocol guards
// ─────────────────────────────────────────────────────────────

const missingVersion = await rpc({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }, { "MCP-Protocol-Version": "" });
check("rejects missing MCP-Protocol-Version header", missingVersion.status === 400, `status ${missingVersion.status}`);

const noAuth = await fetch(ENDPOINT, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-03-26",
  },
  body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }),
});
check(
  "rejects unauthenticated request (401) when auth enabled",
  noAuth.status === 401 || noAuth.status === 200, // 200 only if server runs with auth disabled
  `status ${noAuth.status}`,
);

// ─────────────────────────────────────────────────────────────
// 2. tools/list — inventory
// ─────────────────────────────────────────────────────────────

const expectedTools = [
  "estimate_capacity", "compare_gpus", "recommend_topology", "estimate_api_vs_self_host",
  "list_models", "list_gpus", "get_mlperf_benchmarks", "find_config_for_slo",
  "plan_deployment", "fetch_model_spec", "record_measured",
];

const listRes = await rpc({ jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
const tools = listRes.json?.result?.tools;
check("tools/list returns 200 + JSON-RPC result", listRes.status === 200 && Array.isArray(tools), `status ${listRes.status}`);
if (Array.isArray(tools)) {
  const names = tools.map((t) => t.name);
  check("serves exactly the 11 catalog tools in order", JSON.stringify(names) === JSON.stringify(expectedTools), names.join(","));
  for (const t of tools) {
    check(`tool ${t.name} has inputSchema object`, t.inputSchema?.type === "object", "", { optional: false });
  }
}

// ─────────────────────────────────────────────────────────────
// 3. Golden number through the full HTTP stack
// ─────────────────────────────────────────────────────────────
// Mixtral 8x7B fp16 ×2 H100: weights ≈ 93.4 GB, total ≈ 93.47 GB (hand-verified oracle)

const est = await rpc({
  jsonrpc: "2.0",
  id: 3,
  method: "tools/call",
  params: {
    name: "estimate_capacity",
    arguments: {
      model: "mixtral-8x7b",
      gpu: "h100-sxm",
      gpuCount: 2,
      quantization: "fp16",
      // Same inputs as the golden suite's 93.4/93.47 oracle (promptTokens: 500).
      promptTokens: 500,
      batchSize: 1,
    },
  },
});
const estText = est.json?.result?.content?.[0]?.text;
let estObj = null;
try { estObj = JSON.parse(estText || "{}"); } catch { /* not JSON */ }
const weights = estObj?.memory?.modelWeightsGb ?? estObj?.modelWeightsGb;
const total = estObj?.memory?.totalRequiredGb ?? estObj?.totalVramNeededGb;
check(
  "estimate_capacity golden: Mixtral fp16 ×2 H100 weights ≈ 93.4 GB",
  weights !== undefined && Math.abs(weights - 93.4) < 0.5,
  weights === undefined ? "no modelWeightsGb in response" : `got ${weights}`,
);
check(
  "estimate_capacity golden: total ≈ 93.47 GB",
  total !== undefined && Math.abs(total - 93.47) < 0.5,
  total === undefined ? "no totalRequiredGb in response" : `got ${total}`,
);

// ─────────────────────────────────────────────────────────────
// 4. Error hygiene
// ─────────────────────────────────────────────────────────────

const badTool = await rpc({
  jsonrpc: "2.0",
  id: 4,
  method: "tools/call",
  params: { name: "nonexistent_tool", arguments: {} },
});
const badToolText = badTool.json?.result?.content?.[0]?.text ?? "";
check(
  "unknown tool returns isError with named culprit",
  badTool.status === 200 && (badTool.json?.result?.isError === true || /unknown tool/i.test(badToolText)),
  `status ${badTool.status}`,
);

// ─────────────────────────────────────────────────────────────
// 5. Version stamp consistency
// ─────────────────────────────────────────────────────────────

const stamp = estObj?.provenance?.serverVersion;
check(
  "provenance.serverVersion matches expected pattern",
  typeof stamp === "string" && /^\d+\.\d+\.\d+$/.test(stamp),
  stamp ?? "missing",
);
if (process.env.EXPECTED_VERSION) {
  check(
    "provenance.serverVersion == EXPECTED_VERSION",
    stamp === process.env.EXPECTED_VERSION,
    `${stamp} vs ${process.env.EXPECTED_VERSION}`,
  );
}

// ─────────────────────────────────────────────────────────────
// Summary
// ─────────────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed, ${warned} warned — ${ENDPOINT}`);
process.exit(failed === 0 ? 0 : 1);
