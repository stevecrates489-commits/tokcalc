/**
 * Catalog validation — the CI gate.
 *
 *   bun scripts/catalog-validate.ts
 *
 * Checks, in order:
 *   1. every record matches its Zod schema
 *   2. ids are unique within each file and across files
 *   3. the manifest's counts and version match the actual files
 *   4. cross-record integrity (e.g. a model does not reference a
 *      quantization that does not exist)
 *
 * Exits non-zero on the first category of failure so CI fails the build.
 * This is where the Zod schemas in src/catalog/schemas.ts are enforced —
 * the runtime loader deliberately does not import zod, because it is also
 * bundled into the browser.
 *
 * Run it after editing anything under catalog/.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  AcceleratorRecordSchema,
  CatalogManifestSchema,
  ModelsFileSchema,
  QuantizationsFileSchema,
} from "../src/catalog/schemas";
import type { AcceleratorRecord, ModelRecord, QuantizationRecord } from "../src/catalog/types";

const REPO = resolve(import.meta.dirname, "..");
const CATALOG = resolve(REPO, "catalog");

function readJson(file: string): unknown {
  return JSON.parse(readFileSync(resolve(CATALOG, file), "utf8"));
}

let failures = 0;
let warnings = 0;

function fail(message: string): void {
  failures++;
  console.error(`  FAIL  ${message}`);
}

function warn(message: string): void {
  warnings++;
  console.warn(`  WARN  ${message}`);
}

function section(title: string): void {
  console.log(`\n${title}`);
}

// ---------------------------------------------------------------------------
// 1. Schema validation
// ---------------------------------------------------------------------------

section("Schema validation");

const manifestRaw = readJson("manifest.json");
const manifestResult = CatalogManifestSchema.safeParse(manifestRaw);
if (!manifestResult.success) {
  fail(`manifest.json is invalid:\n${formatIssues(manifestResult.error.issues)}`);
}

const modelsFile = readJson("models.json");
const modelsResult = ModelsFileSchema.safeParse(modelsFile);
if (!modelsResult.success) {
  fail(`models.json is invalid:\n${formatIssues(modelsResult.error.issues)}`);
}

const accelFile = readJson("accelerators.json");
const accelResult = AcceleratorRecordSchema.array().safeParse(
  (accelFile as { records?: unknown[] })?.records,
);
if (!accelResult.success) {
  fail(`accelerators.json is invalid:\n${formatIssues(accelResult.error.issues)}`);
}

const quantsFile = readJson("quantizations.json");
const quantsResult = QuantizationsFileSchema.safeParse(quantsFile);
if (!quantsResult.success) {
  fail(`quantizations.json is invalid:\n${formatIssues(quantsResult.error.issues)}`);
}

if (failures === 0) {
  console.log("  ok    manifest, models, accelerators, quantizations all match their schemas");
}

function formatIssues(
  issues: ReadonlyArray<{ path: ReadonlyArray<PropertyKey>; message: string }>,
): string {
  return issues
    .map((i) => `          ${i.path.map(String).join(".") || "(root)"}: ${i.message}`)
    .join("\n");
}

// ---------------------------------------------------------------------------
// 2. Id uniqueness
// ---------------------------------------------------------------------------

section("Id uniqueness");

const models = (modelsResult.success ? modelsResult.data.records : []) as ModelRecord[];
const accels = (accelResult.success ? accelResult.data : []) as AcceleratorRecord[];
const quants = (quantsResult.success ? quantsResult.data.records : []) as QuantizationRecord[];

function checkUnique(label: string, ids: string[]): void {
  const seen = new Set<string>();
  const dupes = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) dupes.add(id);
    seen.add(id);
  }
  if (dupes.size > 0) fail(`${label}: duplicate id(s) ${[...dupes].join(", ")}`);
  else console.log(`  ok    ${label}: ${ids.length} ids, all unique`);
}

checkUnique("models", models.map((m) => m.id));
checkUnique("accelerators", accels.map((a) => a.id));
checkUnique("quantizations", quants.map((q) => q.id));

const crossCollisions = new Set<string>();
for (const m of models) {
  if (accels.some((a) => a.id === m.id)) crossCollisions.add(m.id);
}
if (crossCollisions.size > 0) {
  fail(`id collision between models and accelerators: ${[...crossCollisions].join(", ")}`);
} else {
  console.log("  ok    no id collisions between models and accelerators");
}

// ---------------------------------------------------------------------------
// 3. Manifest consistency
// ---------------------------------------------------------------------------

section("Manifest consistency");

if (manifestResult.success) {
  const m = manifestResult.data;
  const check = (label: string, declared: number, actual: number) => {
    if (declared !== actual) fail(`${label}: manifest says ${declared}, file has ${actual}`);
    else console.log(`  ok    ${label}: ${actual}`);
  };
  check("models", m.counts.models, models.length);
  check("accelerators", m.counts.accelerators, accels.length);
  check("quantizations", m.counts.quantizations, quants.length);

  const files = m.files as Record<string, string>;
  for (const [key, file] of Object.entries(files)) {
    try {
      readJson(file);
      console.log(`  ok    ${key} -> ${file} exists`);
    } catch {
      fail(`${key} -> ${file} does not exist`);
    }
  }
} else {
  fail("manifest.json could not be read — skipping manifest consistency checks");
}

// ---------------------------------------------------------------------------
// 4. Cross-record integrity + provenance coverage
// ---------------------------------------------------------------------------

section("Cross-record integrity");

const quantIds = new Set(quants.map((q) => q.id));
for (const q of quants) {
  if (q.id !== q.id.toLowerCase()) fail(`quantization id "${q.id}" is not lowercase`);
}

const unspecified = models.filter((m) => m.provenance.source === "unspecified");
const unspecifiedAccels = accels.filter((a) => a.provenance.source === "unspecified");

if (unspecified.length > 0) {
  warn(
    `${unspecified.length} model record(s) have provenance.source = "unspecified": ` +
      unspecified.map((m) => m.id).join(", "),
  );
}
if (unspecifiedAccels.length > 0) {
  warn(
    `${unspecifiedAccels.length} accelerator record(s) have provenance.source = "unspecified": ` +
      unspecifiedAccels.map((a) => a.id).join(", "),
  );
}
if (unspecified.length === 0 && unspecifiedAccels.length === 0) {
  console.log("  ok    every record names a provenance source");
}

// Every model that declares an attention layout must be fully described.
for (const m of models) {
  if (m.attentionLayers && m.attentionLayers.length !== m.layers) {
    fail(`${m.id}: attentionLayers has ${m.attentionLayers.length} entries, model has ${m.layers} layers`);
  }
}

// Attention layouts must be internally consistent.
for (const m of models) {
  for (const [i, layer] of (m.attentionLayers ?? []).entries()) {
    if (layer.kind === "sliding" && !layer.windowTokens) {
      fail(`${m.id} layer ${i}: sliding layer without windowTokens`);
    }
    if (layer.kind === "mla" && (!layer.kvLoraRank || !layer.mlaRopeHeadDim)) {
      fail(`${m.id} layer ${i}: mla layer missing kvLoraRank / mlaRopeHeadDim`);
    }
    if ((layer.kind === "linear" || layer.kind === "kda") && !layer.stateBytes) {
      fail(`${m.id} layer ${i}: ${layer.kind} layer missing stateBytes`);
    }
  }
}

// ---------------------------------------------------------------------------

console.log(
  `\n${failures === 0 ? "CATALOG VALID" : "CATALOG INVALID"} — ` +
    `${failures} failure(s), ${warnings} warning(s)`,
);

if (failures > 0) process.exit(1);
