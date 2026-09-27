#!/usr/bin/env node
/**
 * prepublish guard — refuse to publish a stale artifact.
 *
 * Why this exists: on 2026-09-27 we published 0.2.2 / 0.2.3 / 0.2.4 in quick
 * succession. Each time the source was fixed but the hand-rolled build command
 * only rebuilt ONE of the three entry points, so the bundle npm actually
 * shipped (dist/index.js, which every `bin` alias points at) kept serving
 * pre-fix code. Three releases shipped with a correct package.json version
 * wrapped around a stale bundle, and it was only caught because an external
 * agent diffed observed behaviour against the source.
 *
 * This makes that class of mistake impossible to ship, and cheap to catch:
 * it fails the publish instead of silently releasing something wrong.
 *
 * Checks, in order:
 *   1. every artifact named in package.json `bin` + `main` exists
 *   2. each artifact is newer than every .ts source file it bundles
 *   3. each artifact actually contains the build fingerprint constant
 *
 * Escape hatch for deliberate local testing: SKIP_DIST_CHECK=1 npm publish
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const pkgRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const pkg = JSON.parse(fs.readFileSync(path.join(pkgRoot, "package.json"), "utf8"));

function fail(msg) {
  console.error(`\n[prepublish] BLOCKED: ${msg}\n`);
  console.error("  Run `npm run build` (rebuilds every bin entry point), then retry.\n");
  console.error("  To bypass deliberately: SKIP_DIST_CHECK=1 npm publish\n");
  process.exit(1);
}

if (process.env.SKIP_DIST_CHECK === "1") {
  console.warn("[prepublish] SKIP_DIST_CHECK=1 — publishing without verification.");
  process.exit(0);
}

// ---- 1. collect the artifacts npm will actually ship ----------------------
const artifacts = new Set();
for (const target of Object.values(pkg.bin || {})) artifacts.add(target);
if (pkg.main) artifacts.add(pkg.main);

if (artifacts.size === 0) fail("no bin/main entries found in package.json");

// ---- 2. sources they are built from ---------------------------------------
function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "node_modules" || entry.name === "dist") continue;
      walk(full, acc);
    } else if (entry.name.endsWith(".ts")) {
      acc.push(full);
    }
  }
  return acc;
}
const sources = walk(pkgRoot);
if (sources.length === 0) fail("no .ts sources found — unexpected layout");

const newestSource = sources.reduce(
  (acc, f) => Math.max(acc, fs.statSync(f).mtimeMs),
  0,
);
const newestSourceName = sources
  .map((f) => [f, fs.statSync(f).mtimeMs])
  .sort((a, b) => b[1] - a[1])[0][0];

// ---- 3. verify each artifact ---------------------------------------------
const problems = [];

for (const rel of artifacts) {
  const abs = path.join(pkgRoot, rel);

  if (!fs.existsSync(abs)) {
    problems.push(`${rel} — missing. npm would ship a broken bin entry.`);
    continue;
  }
  if (fs.statSync(abs).mtimeMs < newestSource) {
    problems.push(
      `${rel} — STALE. Built before ${path.relative(pkgRoot, newestSourceName)} was modified.`,
    );
  }
  const body = fs.readFileSync(abs, "utf8");
  if (!body.includes("buildStamp")) {
    problems.push(
      `${rel} — no build fingerprint. Rebuild with \`npm run build\`; a bundle without ` +
        `the stamp is a pre-0.2.4 artifact and predates the provenance feature.`,
    );
  }
}

if (problems.length > 0) {
  console.error("\n[prepublish] BLOCKED — dist/ is out of sync with source:\n");
  for (const p of problems) console.error(`  • ${p}`);
  console.error("\n  Run `npm run build` then retry. Bypass: SKIP_DIST_CHECK=1 npm publish\n");
  process.exit(1);
}

console.log(
  `[prepublish] OK — ${artifacts.size} artifact(s) verified against ${sources.length} source file(s).`,
);
