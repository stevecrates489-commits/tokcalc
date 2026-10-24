/**
 * README count-drift gate.
 *
 * The README quotes hard numbers — how many models, accelerators and
 * quantization formats exist, how many tests each suite contains. Those
 * numbers drift every time the catalog or a test file changes, and a stale
 * README number is worse than no number: it tells a reader the catalog is
 * smaller than it is.
 *
 * This script recomputes every quoted number from the source of truth (the
 * catalog JSON, the test files) and fails if the README disagrees. So the
 * README can no longer go stale silently — a change to the catalog that
 * forgets the README fails CI instead of shipping a lie.
 *
 * Run: bun scripts/readme-counts-check.ts
 */

import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(import.meta.dir, "..");
const README = path.join(ROOT, "README.md");

const failures: string[] = [];

function readJson(rel: string): any {
  return JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8"));
}

function readText(rel: string): string {
  return fs.readFileSync(path.join(ROOT, rel), "utf8");
}

// ---------------------------------------------------------------------------
// 1. Catalog counts — from catalog/*.json, the source of truth
// ---------------------------------------------------------------------------

const modelCount = (readJson("catalog/models.json").records as unknown[]).length;
const acceleratorCount = (readJson("catalog/accelerators.json").records as unknown[]).length;
const quantizationCount = (readJson("catalog/quantizations.json").records as unknown[]).length;

// ---------------------------------------------------------------------------
// 2. Test counts — counted from the test files themselves
// ---------------------------------------------------------------------------

/**
 * Count `it(...)` blocks in a test file.
 *
 * Deliberately counts the source rather than a hardcoded number: a test that
 * is added but not counted in the README would otherwise pass unnoticed.
 */
function countTests(rel: string): number {
  const src = readText(rel);
  const matches = src.match(/^\s*it\(/gm);
  return matches ? matches.length : 0;
}

const goldenTests = countTests("tests/mcp-golden.test.ts");
const parityTests = countTests("tests/mcp-surface-parity.test.ts");
const catalogTests = countTests("tests/catalog.test.ts");

// ---------------------------------------------------------------------------
// 3. Compare against what the README says
// ---------------------------------------------------------------------------

const readme = readText("README.md");

/**
 * Every place the README states a catalog count — checked against the real
 * count.
 *
 * The patterns are deliberately specific ("44 models," / "44 models ×" /
 * "Models (44 entries:") rather than a loose "N models". The README also
 * contains unrelated numbers that happen to be counts of things — "8 model
 * calls" in an example prompt, "13 models with input/cached/output" in the
 * API-pricing row — and a loose pattern would flag those as drift when they
 * are not.
 *
 * EVERY match must carry the right number: a README that says "44 models" in
 * one place and "39 models" in another is exactly the drift this gate exists
 * to catch.
 */
function expectCount(description: string, expected: number, patterns: RegExp[]) {
  const matches: RegExpMatchArray[] = [];
  for (const pattern of patterns) {
    matches.push(...readme.matchAll(pattern));
  }
  if (matches.length === 0) {
    failures.push(`${description}: the README never states this count (expected ${expected})`);
    return;
  }
  const wrong = matches.filter(m => Number(m[1]) !== expected);
  if (wrong.length > 0) {
    const found = [...new Set(matches.map(m => m[0].trim()))].join(" | ");
    failures.push(
      `${description}: expected ${expected}, but the README says "${found}"`,
    );
  }
}

expectCount("model count", modelCount, [
  /\b(\d+) models,/g,
  /\b(\d+) models ×/g,
  /\*\*Models\*\* \((\d+) entries:/g,
]);
expectCount("accelerator count", acceleratorCount, [
  /\b(\d+) accelerators\b/g,
  /\*\*Accelerators\*\* \((\d+) entries:/g,
]);
expectCount("quantization count", quantizationCount, [
  /\b(\d+) quantization formats\b/g,
  /\*\*Quantization\*\* \((\d+) formats:/g,
]);
expectCount(
  "golden test count",
  goldenTests,
  [/Golden-number suite \((\d+) tests\)/g],
);
expectCount(
  "surface-parity test count",
  parityTests,
  [/Surface-parity suite \((\d+) tests\)/g],
);
expectCount(
  "catalog test count",
  catalogTests,
  [/Catalog suite \((\d+) tests\)/g],
);

// ---------------------------------------------------------------------------
// 4. Report
// ---------------------------------------------------------------------------

console.log(
  `catalog: ${modelCount} models, ${acceleratorCount} accelerators, ${quantizationCount} quantization formats`,
);
console.log(
  `tests:   ${goldenTests} golden, ${parityTests} parity, ${catalogTests} catalog`,
);

if (failures.length > 0) {
  console.error(`\nREADME COUNT DRIFT — ${failures.length} problem(s):\n`);
  for (const f of failures) console.error(`  • ${f}`);
  console.error(
    "\nThe README quotes catalog and test counts; they must match the source of truth.\n",
  );
  process.exit(1);
}

console.log("\nREADME COUNTS OK — every quoted number matches the source of truth.");
