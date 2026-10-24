# Changelog

All notable changes to tokcalc (website + `@tokcalc/mcp-server`) are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Catalog module** (`catalog/` + `src/catalog/`) — the model, accelerator and
  quantization data has moved out of `src/lib/token-calc.ts` into structured
  JSON records under `catalog/`, loaded through one shared loader
  (`src/catalog/loader.ts`) that the website, the MCP server and the CLI all
  use. Every record carries a stable id, a lifecycle status and a
  `provenance` block naming the source it came from. `src/catalog/schemas.ts`
  holds the Zod schemas; `scripts/catalog-validate.ts` runs them in CI, so an
  invalid record fails the build instead of reaching a user. The loader
  deliberately does not import zod — it is bundled into the browser, and CI
  has already checked the data by then.
- **Catalog tooling** — `scripts/catalog-import.mjs` derives a model record
  from a real HuggingFace checkpoint (config.json → per-layer attention
  layout, safetensors index → exact resident bytes) instead of having
  somebody transcribe a config by hand. `scripts/hf-params.mjs` reads the
  safetensors headers to count total and active parameters.
  `scripts/catalog-provenance.mjs` backfills provenance only for sources
  that actually answer — a URL that 404s is worse than no URL.
- **Architecture-aware KV engine** — the KV-cache maths now understands
  sliding-window attention (whose cache stops growing at its window),
  linear/recurrent attention (which keeps a fixed-size state instead of a
  cache), and Multi-head Latent Attention (which stores one compressed
  latent, not a key/value pair). A model with no declared layout produces
  exactly the numbers it always did, so every pre-existing catalog entry is
  unaffected.
- **Five model families from real checkpoints** — Qwen3.8-27B, Kimi K3,
  GPT-OSS 20B, GPT-OSS 120B and GLM-5.3, ingested with their upstream
  repository, pinned revision, licence, native weight format, exact resident
  checkpoint size and per-layer attention layout. 44 models in total.
- **MCP tool annotations** — every tool now declares `readOnlyHint`,
  `destructiveHint`, `idempotentHint` and `openWorldHint`, so a client can
  tell a read from a write before calling it.
- **First-answer latency** — a separate metric from TTFT. TTFT says when the
  first character appears; first-answer latency says when the user has
  something readable, which is usually several times longer.

### Fixed

- **The KV-cache rate was displayed 1024× too large.** The per-token figure
  is in kilobytes, but it was being formatted by a helper that takes
  gigabytes, so a 131.2 KB/token cache was shown as "131.2 MB/token" next to
  a total that said 67.1 MB. There is now a formatter that takes kilobytes.
- **TTFT was not prefill time.** The glossary claimed they were the same
  thing. TTFT is prefill plus the first decode step plus per-request
  overhead, and the engine now computes it that way.
- **Provenance labels conflated three different kinds of evidence.**
  "Measured" covered vendor spec sheets, published prices and real benchmark
  runs alike. The confidence tiers are now `official_spec`,
  `published_price`, `observed_run`, `modeled`, `heuristic` and
  `user_supplied`.
- **`record_measured` was advertised as read-only.** It persists a
  calibration measurement and changes the numbers later `estimate_capacity`
  calls return for that cell. It is now declared as a write, the npm listing
  no longer calls all eleven tools read-only, and the surface-parity test
  enforces it.
- **The per-model pages used a hardcoded KV formula** that assumed every
  layer is plain full attention. They now use the engine, so a hybrid model
  shows the cache it actually has.
- **The catalog called non-GPU accelerators "GPUs".** The reference tab, the
  GPU pickers and the footer now say "Accelerators & systems" — the catalog
  has always included TPUs, LPUs, wafer-scale systems and Apple Silicon.
- **Programmatic content engine** — `src/lib/content/` derives every published
  number from `src/lib/token-calc.ts`, the same module the calculator, MCP
  server, and CLI use. No spec on these pages is hand-typed, so a catalog
  change (a new price, a corrected VRAM figure) propagates automatically
  instead of leaving stale prose behind.
- **`/models/[model]`** — 39 per-model pages: VRAM requirement by
  quantization, which GPUs fit and how many are needed, throughput and cost
  per million tokens, concurrency vs context length, architecture facts, and
  a FAQ rendered as `FAQPage` JSON-LD.
- **`/models/[model]/on/[gpu]`** — 710 model-on-GPU pages answering the
  highest-intent query ("can I run Llama 3.3 70B on an H100?"), with a
  1×/2×/4×/8× tensor-parallel ladder, a context-length table, and cheaper or
  faster alternatives.
- **`/gpus/[gpu]`** — 31 per-accelerator pages: hardware specs, the models
  that fit on one card, and a cross-accelerator comparison on a fixed
  reference workload.
- **`/models` and `/gpus`** — hub pages listing the whole catalog.
- **Dynamic sitemap** (`src/app/sitemap.ts`) generated from the catalog,
  replacing the hand-maintained `public/sitemap.xml` that listed 4 URLs while
  the site served dozens. A model or GPU added to the catalog is now
  discoverable on the next build with no manual XML editing.

### Fixed

- Hardware with no published rental price (`usdPerHour: null` — Groq,
  Cerebras) and purchased hardware (`usdPerHour: 0` — Apple Silicon) no
  longer ranks as "the cheapest configuration", because the cost model
  resolves both to $0/hr. They are excluded from every price ranking and
  labelled "quote required" instead.
- Wafer-scale systems and LPUs are no longer reported as "fits on one card" —
  the whole system is the unit of deployment, so a per-card fit is a category
  error.

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
