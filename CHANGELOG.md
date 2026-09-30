# Changelog

All notable changes to tokcalc (website + `@tokcalc/mcp-server`) are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — 2026-09

### Added

- **`tokcalc plan` CLI** (`npx @tokcalc/mcp-server plan`) — a non-interactive
  capacity planner for CI and bots that cannot speak MCP. Reads `.tokcalc.json`
  or flags, emits Markdown (default) or JSON. Exit code is `0` when the plan
  fits, `2` when it does not, `1` on usage error, so it can gate a build
  directly. It calls the same `src/lib/token-calc.ts` the site and MCP tools
  use, rather than reimplementing the math.
- **GitHub Action** (root `action.yml`) — comments a capacity + cost plan on
  every pull request that touches the plan, so reviewers can sanity-check the
  model/GPU/context choice without leaving the diff. Runs tokcalc's own
  TypeScript sources via `setup-node`, so it cannot drift from the calculator.
  Opt into a hard failure with `fail-on-infeasible: true`.
- **MCP Registry publication** — `server.json` is now published to the official
  registry by `.github/workflows/publish-mcp.yml` on every `v*` tag, using
  GitHub OIDC (no stored PAT). The workflow refuses to publish when the tag,
  `MCP_SERVER_VERSION`, and `server.json` disagree. The registry listing had
  been pinned at `0.2.0` while npm was at `0.2.9`, so registry installs were
  serving a build from before the single-core refactor.

### Fixed

- Catalog counts in the docs said 35 models / 30 GPUs; the catalog actually
  holds 39 / 31.
- stdio entry logged "7 tools available"; there have been 11 since 0.2.7.

## [0.2.9] — 2026-09

### Fixed

- **`find_config_for_slo` cost model** — dedicated-config monthly cost was computed
  per-token (an 8×B200 rig quoted ≈ $233/mo instead of the real ≈ $35,040/mo).
  Now `dedicatedMonthlyUsd = gpuCount × $/hr × 730`, consistent with every other
  tool. `costPerMillionTokens` is labeled as marginal cost.
- Ranking now tiered: sustaining configs rank above undersized ones.
  Adds `requiredTps`, `meanLoadUtilizationPct`, `sustainsMeanLoad`;
  `best_value` ranking credit-caps throughput at 2× mean load;
  `alternatives` filtered to the sustaining tier.
- `workloadShape` + `formulaNotes` updated to describe the new basis.

## [0.2.8] — 2026-09

### Added

- Tiered black-box QA battery (`tests/qa-live-probe.mjs`): 35 checks against the
  **published** npm tarball — handshake/inventory, golden numbers, new-tool
  behavior incl. live HuggingFace fetch, and adversarial inputs
  (garbage JSON, pipelining, gpuCount=256, cross-tool consistency).

### Fixed

- `estimate_capacity`: `promptTokens` now guarded by the model's `maxContext`
  (consistent with `maxContextTokens`).
- Strict-schema errors name the culprit key via Zod 4 `issue.keys`
  (Zod 4 removed `.path` on some issues).
- KV-cache math shows 4 decimals so hidden rounding headroom is visible.

## [0.2.7] — 2026-09

### Added

- 4 new tools (11 total): `find_config_for_slo` (inverse planner),
  `plan_deployment` (one-call brief), `fetch_model_spec` (HuggingFace
  config.json diff, 24h cache), `record_measured` (calibration loop).
- `estimate_capacity`: `kvQuantization` (fp16/int8/int4), `cachePrefixTokens` /
  `cacheHitRate`, `useSpeculative` / `speculativeBoost`.
- Unified core: `src/lib/mcp/core.ts` is the single tool implementation;
  `mini-services/mcp-server/server.ts` and `src/lib/mcp/server.ts` are thin
  shims re-exporting it (kills the two-copies drift).

### Changed

- Golden-number test suite (`tests/mcp-golden.test.ts`, bun test) locking
  hand-verified oracle values.

## [0.2.6] — 2026-09

### Fixed

- MoE resident-weights: capacity uses all resident params
  (`paramsB`), throughput uses active params (`activeParamsB`).
- Strict input schemas (`additionalProperties: false`) with culprits named.
- Rebuilt all three dist entry points; prepublish staleness guard added.

## [0.2.0] — 2026

### Added

- Public hosted endpoint `https://tokcalc.vercel.app/api/mcp` (stateless
  Streamable HTTP) with bearer API-key auth + Upstash Redis rate limiting.
- Self-serve API keys at [/mcp](https://tokcalc.vercel.app/mcp).

## [0.1.4] — 2026

### Added

- Initial public MCP server: stdio transport, 7 planning tools
  (estimate_capacity, compare_gpus, recommend_topology,
  estimate_api_vs_self_host, list_models, list_gpus, get_mlperf_benchmarks).

[Unreleased]: consolidation — GitHub Actions CI, surface-parity tests,
HTTP smoke probe (`tests/http-smoke.mjs`), shared tool catalog
(`src/lib/mcp/tool-catalog.ts`), repo-wide typecheck restored to zero errors.
