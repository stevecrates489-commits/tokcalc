#!/usr/bin/env node
/**
 * tokcalc plan — non-interactive capacity planner CLI.
 *
 * This is the same engine the web calculator and the MCP tools run on
 * (src/lib/token-calc.ts), exposed for CI and bots that cannot speak MCP.
 * The GitHub Action at the repo root drives this.
 *
 * It talks to the calculator directly rather than through the MCP protocol
 * layer on purpose: a CLI should not pay for JSON-RPC framing to get a number.
 *
 * Usage:
 *   tokcalc-plan plan --model llama3-70b --gpu h100-sxm --quant fp8
 *   tokcalc-plan plan --config .tokcalc.json --format json
 *   tokcalc-plan models | gpus | quants
 */

import { readFileSync, existsSync } from "node:fs";
import {
  calculate,
  recommendTopology,
  fmtTokens,
  fmtBytes,
  fmtMs,
  fmtMoney,
  GPUS,
  MODELS,
  QUANTIZATIONS,
  GPU_MAP,
  MODEL_MAP,
} from "../../src/lib/token-calc.ts";
import { MCP_SERVER_VERSION, CATALOG_VERSION } from "../../src/lib/mcp-version.ts";

const DEFAULT_CONFIG_PATH = ".tokcalc.json";

/** Provenance stamp, surfaced in every output format.
 *  The npm prepublish guard requires each `bin` artifact to carry this. */
const buildStamp = `tokcalc-plan/${MCP_SERVER_VERSION}`;

interface PlanConfig {
  model?: string;
  gpu?: string;
  quantization?: string;
  numGpus?: number;
  contextTokens?: number;
  concurrency?: number;
  promptTokens?: number;
  outputTokens?: number;
  useContinuousBatching?: boolean;
  continuousBatchingMultiplier?: number;
  reasoningTokens?: number;
  cachePrefixTokens?: number;
  cacheHitRate?: number;
  engineId?: string;
}

/** Resolve an id case-insensitively; on a miss, show near matches. */
function resolve<T extends { id: string }>(
  map: Record<string, T>,
  table: readonly T[],
  requested: string,
  label: string,
): T {
  const direct = map[requested];
  if (direct) return direct;

  const lower = requested.toLowerCase();
  const byId = table.find((x) => x.id.toLowerCase() === lower);
  if (byId) return byId;

  const contains = table.filter((x) => x.id.toLowerCase().includes(lower)).slice(0, 8);
  const hint = contains.length
    ? `\n  did you mean: ${contains.map((c) => c.id).join(", ")}`
    : `\n  available: ${table.slice(0, 12).map((c) => c.id).join(", ")}...`;
  throw new Error(`Unknown ${label} "${requested}".${hint}`);
}

function parseArgs(argv: string[]): { cmd: string; flags: Record<string, string | boolean> } {
  const cmd = argv[0] && !argv[0].startsWith("-") ? argv[0] : "plan";
  const rest = argv[0] && !argv[0].startsWith("-") ? argv.slice(1) : argv;
  const flags: Record<string, string | boolean> = {};
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = rest[i + 1];
    if (next === undefined || next.startsWith("--")) {
      flags[key] = true;
    } else {
      flags[key] = next;
      i++;
    }
  }
  return { cmd, flags };
}

function loadConfig(flags: Record<string, string | boolean>): PlanConfig {
  const path = typeof flags.config === "string" ? flags.config : DEFAULT_CONFIG_PATH;
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8")) as PlanConfig;
  } catch (e) {
    throw new Error(`Could not parse ${path}: ${(e as Error).message}`);
  }
}

function num(v: unknown): number | undefined {
  if (v === undefined || v === null || v === "") return undefined;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function buildMarkdown(
  cfg: Required<Pick<PlanConfig, "model" | "gpu" | "quantization" | "numGpus" | "contextTokens" | "concurrency" | "promptTokens" | "outputTokens">>,
  res: ReturnType<typeof calculate>,
  topo: ReturnType<typeof recommendTopology> | null,
): string {
  const verdict = res.vramFits
    ? `✅ **Fits** — ${fmtBytes(res.totalVramNeededGb)} of VRAM required`
    : `❌ **Does not fit** — ${fmtBytes(res.totalVramNeededGb)} required, ${fmtBytes(
        (GPU_MAP[cfg.gpu]?.vramGb ?? 0) * cfg.numGpus,
      )} available`;

  const moeLine = res.isMoE
    ? `\n> **MoE model.** Resident weights (all experts) **${fmtBytes(res.modelSizeGb)}**; only **${fmtBytes(
        res.activeParamsGb,
      )}** is touched per token. Sizing on the active set is the most common MoE planning error — you must buy VRAM for the resident set.`
    : "";

  const rows: string[] = [
    "| Metric | Value |",
    "| --- | --- |",
    `| Weights | ${fmtBytes(res.modelSizeGb)} |`,
    `| KV cache @ ${fmtTokens(cfg.contextTokens)} ctx × ${cfg.concurrency} | ${fmtBytes(res.kvCacheTotalGb)} |`,
    `| **Total VRAM** | **${fmtBytes(res.totalVramNeededGb)}** |`,
    `| Decode (1 stream) | ${res.decodeTokensPerSec.toFixed(1)} tok/s |`,
    `| Aggregate throughput | ${res.aggregateTokensPerSec.toFixed(1)} tok/s |`,
    `| TTFT (prefill) | ${fmtMs(res.ttftMs)} |`,
    `| ITL (inter-token) | ${fmtMs(res.itlMs)} |`,
    `| End-to-end latency | ${fmtMs(res.totalLatencyMs)} |`,
    `| Cost | ${fmtMoney(res.costPerHour)}/hr · ${fmtMoney(res.costPerMillionOutputTokens)}/M out-tokens |`,
  ];

  const topoBlock = topo
    ? `\n**Suggested topology:** ${topo.topology} — ${topo.reason}`
    : "";

  return [
    `## tokcalc plan`,
    ``,
    verdict + moeLine,
    ``,
    ...rows,
    topoBlock,
    ``,
    `<sub>${buildStamp} · catalog ${CATALOG_VERSION} · \`${cfg.model}\` on ${cfg.numGpus}× ${cfg.gpu} @ ${cfg.quantization}</sub>`,
  ].join("\n");
}

function main(): number {
  const { cmd, flags } = parseArgs(process.argv.slice(2));

  if (flags.version || flags.v) {
    console.log(`tokcalc-plan ${MCP_SERVER_VERSION} (catalog ${CATALOG_VERSION})`);
    return 0;
  }

  if (cmd === "models" || cmd === "gpus" || cmd === "quants") {
    if (cmd === "models") MODELS.forEach((m) => console.log(m.id));
    if (cmd === "gpus") GPUS.forEach((g) => console.log(g.id));
    if (cmd === "quants") QUANTIZATIONS.forEach((q) => console.log(q.id));
    return 0;
  }

  if (cmd === "help" || flags.help || flags.h) {
    console.log(
      [
        "tokcalc plan — LLM serving capacity planner",
        "",
        "Usage:",
        "  tokcalc-plan plan [options]",
        "",
        "Options:",
        "  --config <path>        JSON config (default: .tokcalc.json)",
        "  --model <id>           model id (see: tokcalc-plan models)",
        "  --gpu <id>             GPU id (see: tokcalc-plan gpus)",
        "  --quant <format>       quantization (see: tokcalc-plan quants)",
        "  --gpus <n>             tensor-parallel degree",
        "  --context <tokens>     context length per request",
        "  --concurrency <n>      concurrent requests",
        "  --prompt <tokens>      prompt tokens per request",
        "  --output <tokens>      generated tokens per request",
        "  --format <md|json>     output format (default: md)",
        "  --version              print version",
        "",
        "Flags override the config file.",
      ].join("\n"),
    );
    return 0;
  }

  const fileCfg = loadConfig(flags);

  const modelId = (flags.model as string) ?? fileCfg.model;
  const gpuId = (flags.gpu as string) ?? fileCfg.gpu;
  const quantization = (flags.quant as string) ?? fileCfg.quantization ?? "bf16";

  if (!modelId) throw new Error("Missing --model (or \"model\" in the config file).");
  if (!gpuId) throw new Error("Missing --gpu (or \"gpu\" in the config file).");

  const model = resolve(MODEL_MAP, MODELS, modelId, "model");
  const gpu = resolve(GPU_MAP, GPUS, gpuId, "GPU");

  const quant = QUANTIZATIONS.find(
    (q) => q.id === quantization || q.id.toLowerCase() === quantization.toLowerCase(),
  );
  if (!quant) {
    throw new Error(
      `Unknown quantization "${quantization}".\n  available: ${QUANTIZATIONS.map((q) => q.id).join(", ")}`,
    );
  }

  const numGpus = num(flags.gpus) ?? num(fileCfg.numGpus) ?? 1;
  const contextTokens = num(flags.context) ?? num(fileCfg.contextTokens) ?? 8192;
  const concurrency = num(flags.concurrency) ?? num(fileCfg.concurrency) ?? 8;
  const promptTokens = num(flags.prompt) ?? num(fileCfg.promptTokens) ?? 2048;
  const outputTokens = num(flags.output) ?? num(fileCfg.outputTokens) ?? 512;

  const res = calculate({
    modelId: model.id,
    gpuId: gpu.id,
    quantization: quant.id,
    numGpus,
    batchSize: concurrency,
    promptTokens,
    outputTokens,
    useContinuousBatching: fileCfg.useContinuousBatching,
    continuousBatchingMultiplier: fileCfg.continuousBatchingMultiplier,
    reasoningTokens: fileCfg.reasoningTokens,
    cachePrefixTokens: fileCfg.cachePrefixTokens,
    cacheHitRate: fileCfg.cacheHitRate,
    engineId: fileCfg.engineId,
  });

  let topo: ReturnType<typeof recommendTopology> | null = null;
  try {
    topo = recommendTopology(model, gpu, contextTokens, concurrency, quant.bytesPerParam);
  } catch {
    topo = null;
  }

  const cfg = { model: model.id, gpu: gpu.id, quantization: quant.id, numGpus, contextTokens, concurrency, promptTokens, outputTokens };

  if (flags.format === "json") {
    console.log(
      JSON.stringify(
        { buildStamp, catalog: CATALOG_VERSION, config: cfg, result: res, topology: topo },
        null,
        2,
      ),
    );
  } else {
    console.log(buildMarkdown(cfg, res, topo));
  }

  // Non-zero exit when the config does not fit, so CI can gate on it.
  return res.vramFits ? 0 : 2;
}

try {
  process.exitCode = main();
} catch (e) {
  console.error(`tokcalc-plan: ${(e as Error).message}`);
  process.exitCode = 1;
}
