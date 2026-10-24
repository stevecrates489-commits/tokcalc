/**
 * The ONE catalog loader.
 *
 * The website, the MCP server and the CLI all read the catalog through this
 * module. There is no second copy of the model list anywhere in the repo.
 *
 * Data flow:
 *   catalog/*.json   (editable source of truth, reviewed in Git)
 *        ↓  import  (bundled — no runtime file read, works in the browser)
 *   this loader      (typed records + integrity checks)
 *        ↓
 *   src/lib/token-calc.ts  (the calculation engine)
 *
 * Validation: the authoritative Zod schemas live in `./schemas` and run in
 * CI via `scripts/catalog-validate.ts`, so an invalid record fails the build
 * rather than reaching a user. The checks below are the cheap runtime
 * invariants that must hold no matter what — they cost nothing and catch the
 * mistakes that actually happen (a duplicated id, a stale manifest count).
 *
 * This module deliberately does NOT import zod: it is imported by client
 * components, and pulling the validator into the browser bundle to re-check
 * data that CI has already checked would be a bad trade.
 */

import modelsFile from "../../catalog/models.json";
import acceleratorsFile from "../../catalog/accelerators.json";
import quantizationsFile from "../../catalog/quantizations.json";
import manifestFile from "../../catalog/manifest.json";

import type {
  AcceleratorRecord,
  ModelRecord,
  QuantizationRecord,
} from "./types";

/** Catalog manifest — version and file index. */
export const CATALOG_MANIFEST = manifestFile as unknown as {
  catalogVersion: string;
  description: string;
  updatedAt: string;
  files: { models: string; accelerators: string; quantizations: string };
  counts: { models: number; accelerators: number; quantizations: number };
};

/**
 * Version of the catalog data. Bumped whenever a record changes in a way that
 * moves a published number. Surfaced in MCP output and on the site.
 */
export const CATALOG_VERSION = CATALOG_MANIFEST.catalogVersion;

function takeRecords<T>(file: unknown, label: string): T[] {
  if (!file || typeof file !== "object" || !Array.isArray((file as any).records)) {
    throw new Error(
      `catalog: ${label}.json is malformed — expected an object with a "records" array`,
    );
  }
  return (file as { records: T[] }).records;
}

/**
 * Reject duplicate ids. Ids are stable public identifiers — they appear in
 * URLs, in MCP tool arguments and in saved measurement records — so a
 * duplicate silently shadows one of the two records.
 */
function assertUniqueIds<T extends { id: string }>(records: T[], label: string): void {
  const seen = new Set<string>();
  const dupes = new Set<string>();
  for (const r of records) {
    if (seen.has(r.id)) dupes.add(r.id);
    seen.add(r.id);
  }
  if (dupes.size > 0) {
    throw new Error(
      `catalog: duplicate ${label} id(s): ${[...dupes].join(", ")} — ids must be unique`,
    );
  }
}

export const MODELS: ModelRecord[] = takeRecords<ModelRecord>(modelsFile, "models");
export const ACCELERATORS: AcceleratorRecord[] =
  takeRecords<AcceleratorRecord>(acceleratorsFile, "accelerators");
export const QUANTIZATIONS: QuantizationRecord[] =
  takeRecords<QuantizationRecord>(quantizationsFile, "quantizations");

assertUniqueIds(MODELS, "model");
assertUniqueIds(ACCELERATORS, "accelerator");
assertUniqueIds(QUANTIZATIONS, "quantization");

// The manifest's counts are a human-readable summary in catalog/manifest.json.
// If they drift from the actual files, the summary is lying — catch it here.
{
  const drift: string[] = [];
  const check = (label: string, declared: number, actual: number) => {
    if (declared !== actual) drift.push(`${label}: manifest says ${declared}, file has ${actual}`);
  };
  check("models", CATALOG_MANIFEST.counts.models, MODELS.length);
  check("accelerators", CATALOG_MANIFEST.counts.accelerators, ACCELERATORS.length);
  check("quantizations", CATALOG_MANIFEST.counts.quantizations, QUANTIZATIONS.length);
  if (drift.length > 0) {
    throw new Error(
      `catalog: manifest counts are stale — ${drift.join("; ")}. Update catalog/manifest.json.`,
    );
  }
}

export const MODEL_MAP: Record<string, ModelRecord> = Object.fromEntries(
  MODELS.map((m) => [m.id, m]),
);

export const ACCELERATOR_MAP: Record<string, AcceleratorRecord> = Object.fromEntries(
  ACCELERATORS.map((g) => [g.id, g]),
);

export const QUANT_MAP: Record<string, QuantizationRecord> = Object.fromEntries(
  QUANTIZATIONS.map((q) => [q.id, q]),
);

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

/** Look a model up by its stable id. Returns undefined if unknown. */
export function getModel(id: string): ModelRecord | undefined {
  return MODEL_MAP[id];
}

/** Look an accelerator up by its stable id. Returns undefined if unknown. */
export function getAccelerator(id: string): AcceleratorRecord | undefined {
  return ACCELERATOR_MAP[id];
}

/** Look a quantization up by its stable id. Returns undefined if unknown. */
export function getQuantization(id: string): QuantizationRecord | undefined {
  return QUANT_MAP[id];
}

/**
 * Records that are still current.
 *
 * Deprecated records stay in the catalog so that old URLs, saved
 * measurement records and older MCP sessions keep resolving — but they are
 * not offered as new recommendations.
 */
export function activeModels(): ModelRecord[] {
  return MODELS.filter((m) => m.lifecycle === "active");
}

export function activeAccelerators(): AcceleratorRecord[] {
  return ACCELERATORS.filter((a) => a.lifecycle === "active");
}

/** Every distinct model family, in first-appearance order. */
export function modelFamilies(): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of MODELS) {
    if (!seen.has(m.family)) {
      seen.add(m.family);
      out.push(m.family);
    }
  }
  return out;
}

/** Every distinct accelerator vendor, in first-appearance order. */
export function acceleratorVendors(): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const a of ACCELERATORS) {
    if (!seen.has(a.vendor)) {
      seen.add(a.vendor);
      out.push(a.vendor);
    }
  }
  return out;
}
