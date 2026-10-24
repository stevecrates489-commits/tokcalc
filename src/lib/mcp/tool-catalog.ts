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
 *   4. the `annotations` block here MUST match the wire annotations
 *      name-for-name — in particular record_measured must be marked
 *      readOnlyHint: false, because it persists a measurement.
 *
 * When adding tool #12: update the schema + handler + switch in core.ts,
 * then mirror it here — the parity test will fail until all agree.
 */

export interface ToolAnnotations {
  /** Human-readable title for the tool. */
  title: string;
  /** The tool does not modify its environment. */
  readOnlyHint: boolean;
  /** The tool may perform destructive updates. */
  destructiveHint: boolean;
  /** Repeating the call with the same arguments has no additional effect. */
  idempotentHint: boolean;
  /** The tool may interact with external entities (network, other systems). */
  openWorldHint: boolean;
}

/** A tool that only reads the catalog and computes — no side effects. */
const READ_ONLY = (title: string): ToolAnnotations => ({
  title,
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
});

export const TOOL_CATALOG: ReadonlyArray<{
  name: string;
  description: string;
  annotations: ToolAnnotations;
}> = [
  { name: "estimate_capacity", description: "VRAM/KV/throughput/latency for one config. Returns feasibility, performance ranges, cost, confidence.", annotations: READ_ONLY("Estimate serving capacity") },
  { name: "compare_gpus", description: "Ranked GPU comparison for one workload. Sort by lowest cost, highest throughput, or best value.", annotations: READ_ONLY("Compare accelerators") },
  { name: "recommend_topology", description: "TP/CP topology recommendation. Returns feasible GPU + topology designs for a context length.", annotations: READ_ONLY("Recommend GPU topology") },
  { name: "estimate_api_vs_self_host", description: "Break-even analysis. Compares monthly API costs vs self-hosted GPU infrastructure.", annotations: READ_ONLY("Compare API vs self-host") },
  { name: "list_models", description: "Discover supported model IDs. Filter by family, category, MoE.", annotations: READ_ONLY("List supported models") },
  { name: "list_gpus", description: "Discover supported GPU IDs. Filter by vendor, category, min VRAM.", annotations: READ_ONLY("List supported accelerators") },
  { name: "get_mlperf_benchmarks", description: "Curated MLPerf Inference v4.1 audited configs. Cross-validate estimates against reference systems.", annotations: READ_ONLY("Get MLPerf reference results") },
  { name: "find_config_for_slo", description: "Inverse planner. Given model + context + traffic + SLO ceilings (TTFT, $/M, tok/s), searches every GPU × topology and returns feasible configs ranked by cost, throughput, or value.", annotations: READ_ONLY("Find a config for an SLO") },
  { name: "plan_deployment", description: "One-call decision brief: memory + KV headroom, performance, rig cost, blended build-vs-buy with break-even, concrete risks, and next steps.", annotations: READ_ONLY("Plan a deployment") },
  { name: "fetch_model_spec", description: "Fetches the model's real config.json from HuggingFace and diffs it against tokcalc's catalog (layers, KV heads, head_dim). Catches catalog drift before it skews KV math. 24h cache.", annotations: { ...READ_ONLY("Fetch upstream model spec"), openWorldHint: true } },
  {
    name: "record_measured",
    description: "Calibration loop. Record real tok/s / TTFT from your serving run; future estimates for that model+GPU+quantization are annotated with your measured/predicted ratio.",
    // The only writing tool in the server. It persists a calibration record
    // and changes the numbers later estimate_capacity calls return for that
    // cell — so it must not advertise itself as read-only.
    annotations: {
      title: "Record a measured benchmark",
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
  },
];
