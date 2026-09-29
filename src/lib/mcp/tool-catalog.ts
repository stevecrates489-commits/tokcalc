/**
 * Client-safe tool catalog — the /mcp docs page (a client component) imports
 * this instead of core.ts, because core.ts transitively uses node:fs
 * (spec cache) and cannot be bundled for the browser.
 *
 * PARITY CONTRACT (enforced by tests/mcp-surface-parity.test.ts):
 *   1. names + order here MUST match TOOL_DEFINITIONS in src/lib/mcp/core.ts
 *      (the wire format served by stdio, self-hosted HTTP, and /api/mcp)
 *   2. every name MUST be handled by the CallToolRequest switch in core.ts
 *   3. every name MUST appear in mini-services/mcp-server/package.json's
 *      description (the npm listing advertises the same tool set)
 *
 * When adding tool #12: update the schema + handler + switch in core.ts,
 * then mirror it here — the parity test will fail until all agree.
 */
export const TOOL_CATALOG: ReadonlyArray<{ name: string; description: string }> = [
  { name: "estimate_capacity", description: "VRAM/KV/throughput/latency for one config. Returns feasibility, performance ranges, cost, confidence." },
  { name: "compare_gpus", description: "Ranked GPU comparison for one workload. Sort by lowest cost, highest throughput, or best value." },
  { name: "recommend_topology", description: "TP/CP topology recommendation. Returns feasible GPU + topology designs for a context length." },
  { name: "estimate_api_vs_self_host", description: "Break-even analysis. Compares monthly API costs vs self-hosted GPU infrastructure." },
  { name: "list_models", description: "Discover supported model IDs. Filter by family, category, MoE." },
  { name: "list_gpus", description: "Discover supported GPU IDs. Filter by vendor, category, min VRAM." },
  { name: "get_mlperf_benchmarks", description: "Curated MLPerf Inference v4.1 audited configs. Cross-validate estimates against reference systems." },
  { name: "find_config_for_slo", description: "Inverse planner. Given model + context + traffic + SLO ceilings (TTFT, $/M, tok/s), searches every GPU × topology and returns feasible configs ranked by cost, throughput, or value." },
  { name: "plan_deployment", description: "One-call decision brief: memory + KV headroom, performance, rig cost, blended build-vs-buy with break-even, concrete risks, and next steps." },
  { name: "fetch_model_spec", description: "Fetches the model's real config.json from HuggingFace and diffs it against tokcalc's catalog (layers, KV heads, head_dim). Catches catalog drift before it skews KV math. 24h cache." },
  { name: "record_measured", description: "Calibration loop. Record real tok/s / TTFT from your serving run; future estimates for that model+GPU+quantization are annotated with your measured/predicted ratio." },
];
