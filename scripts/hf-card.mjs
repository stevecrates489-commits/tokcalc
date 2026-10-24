/**
 * Pull the parameter-count / license / context claims out of a HuggingFace
 * model card so the catalog records can cite a real source instead of a
 * remembered number.
 *
 *   node scripts/hf-card.mjs <repo>
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const repo = process.argv[2];
if (!repo) {
  console.error("usage: node scripts/hf-card.mjs <repo>");
  process.exit(1);
}

const dir = join(process.env.TEMP, "hfcache");
mkdirSync(dir, { recursive: true });
const slug = repo.replace(/\//g, "__");
const path = join(dir, `${slug}.README.md`);

if (!existsSync(path)) {
  const url = `https://huggingface.co/${repo}/resolve/main/README.md`;
  console.log(`  fetching ${url}`);
  const res = await fetch(url);
  if (!res.ok) {
    console.error(`${res.status} fetching ${url}`);
    process.exit(1);
  }
  writeFileSync(path, Buffer.from(await res.arrayBuffer()));
}

let text;
try {
  text = readFileSync(path, "utf8");
} catch {
  console.error(`no cached README for ${repo}`);
  process.exit(2);
}

console.log(`=== ${repo} README (${text.length} chars) ===`);

const PATTERNS = [
  /\b\d+(?:\.\d+)?\s*[BM]\b(?=[^\n]{0,40}(?:param|total))/gi,
  /\b(?:total|active)\s+parameters?[^.\n]{0,60}/gi,
  /\b\d+(?:\.\d+)?\s*[BM]\s*(?:total\s+)?parameters?/gi,
  /\bactive\s+parameters?[^.\n]{0,40}/gi,
  /\b\d+(?:\.\d+)?\s*[BM]\s*(?:total\s+)?params\b/gi,
  /^license:.*$/gim,
  /^\s*license\s*[:=]\s*\S+.*$/gim,
  /\bcontext (?:length|window)[^.\n]{0,50}/gi,
  /\b\d+K\s*context\b/gi,
  /\bcontext_length\s*[:=]\s*[^\n]{0,20}/gi,
  /\b(?:MIT|Apache-2\.0|Apache 2\.0|LICENSE|MODEL LICENSE)[^\n]{0,60}/g,
];

const seen = new Set();
for (const re of PATTERNS) {
  const matches = text.match(re);
  if (!matches) continue;
  for (const m of matches.slice(0, 12)) {
    const line = m.replace(/\s+/g, " ").trim();
    if (line.length < 3) continue;
    if (seen.has(line)) continue;
    seen.add(line);
    console.log(`  ${line}`);
  }
}

// Also grab the YAML frontmatter, which usually carries license + base_model.
const fm = text.match(/^---\n([\s\S]*?)\n---/);
if (fm) {
  console.log("  --- frontmatter ---");
  for (const line of fm[1].split("\n")) {
    if (/license|base_model|pipeline_tag|tags/i.test(line)) {
      console.log(`    ${line.trim()}`);
    }
  }
}
