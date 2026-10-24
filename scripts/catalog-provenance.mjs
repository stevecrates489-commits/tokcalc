/**
 * Backfill provenance on legacy catalog records — but only for sources that
 * actually resolve.
 *
 *   node scripts/catalog-provenance.mjs [--dry-run]
 *
 * The catalog predates the provenance requirement, so many records carry
 * `provenance.source = "unspecified"`. Rather than hand-writing URLs and
 * hoping, this script checks every candidate URL with a real request and
 * only writes a record when the source answers.
 *
 * A source that 404s is worse than no source: it looks citable and is not.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const REPO = resolve(import.meta.dirname, "..");
const dryRun = process.argv.includes("--dry-run");

/** Candidate source URLs for the model records, keyed by catalog id. */
const MODEL_SOURCES = {
  "llama3-8b": "https://huggingface.co/meta-llama/Meta-Llama-3-8B",
  "llama3-70b": "https://huggingface.co/meta-llama/Meta-Llama-3-70B",
  "llama3-405b": "https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct",
  "llama3-3-70b": "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
  "llama2-7b": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
  "llama4-scout": "https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "llama4-maverick": "https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
  "mistral-7b": "https://huggingface.co/mistralai/Mistral-7B-v0.3",
  "mixtral-8x7b": "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1",
  "mixtral-8x22b": "https://huggingface.co/mistralai/Mixtral-8x22B-v0.1",
  "mistral-large-3": "https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512",
  "pixtral-12b": "https://huggingface.co/mistralai/Pixtral-12B-2409",
  "codestral-25": "https://huggingface.co/mistralai/Codestral-2508",
  "qwen2-7b": "https://huggingface.co/Qwen/Qwen2-7B-Instruct",
  "qwen2-72b": "https://huggingface.co/Qwen/Qwen2-72B-Instruct",
  "qwen2-5-14b": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
  "qwen2-5-72b": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
  "qwen3-4b": "https://huggingface.co/Qwen/Qwen3-4B",
  "qwen3-8b": "https://huggingface.co/Qwen/Qwen3-8B",
  "qwen3-14b": "https://huggingface.co/Qwen/Qwen3-14B",
  "qwen3-32b": "https://huggingface.co/Qwen/Qwen3-32B",
  "qwen3-30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B",
  "qwen3-235b-a22b": "https://huggingface.co/Qwen/Qwen3-235B-A22B",
  "qwen2-5-vl-7b": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
  "qwen2-5-vl-72b": "https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct",
  "deepseek-v3": "https://huggingface.co/deepseek-ai/DeepSeek-V3",
  "deepseek-r1": "https://huggingface.co/deepseek-ai/DeepSeek-R1",
  "deepseek-coder-v2": "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct",
  "gemma2-9b": "https://huggingface.co/google/gemma-2-9b",
  "gemma2-27b": "https://huggingface.co/google/gemma-2-27b",
  "phi3-7b": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
  "phi4-14b": "https://huggingface.co/microsoft/phi-4",
  "smollm2-1.7b": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct",
  "falcon3-10b": "https://huggingface.co/tiiuae/falcon3-10b-instruct",
  "olmo2-13b": "https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct",
  "gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b",
  "bge-m3": "https://huggingface.co/BAAI/bge-m3",
  "e5-mlarge": "https://huggingface.co/intfloat/multilingual-e5-large",
  "gte-large": "https://huggingface.co/thenlper/gte-large",
};

/**
 * Candidate vendor pages for the accelerator records. The first one that
 * answers is the one recorded.
 */
const ACCELERATOR_SOURCES = {
  "a10g": [
    "https://www.nvidia.com/en-us/data-center/a10g",
    "https://www.nvidia.com/en-us/data-center/products/a10g",
    // NVIDIA retired the standalone A10G product page; the A10 family
    // datasheet is the live vendor document.
    "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf",
  ],
  "t4": ["https://www.nvidia.com/en-us/data-center/tesla-t4"],
  "v100": ["https://www.nvidia.com/en-us/data-center/v100"],
  "l4": ["https://www.nvidia.com/en-us/data-center/l4"],
  "rtx-6000a": ["https://www.nvidia.com/en-us/design-visualization/rtx-6000/"],
  "a4000": ["https://www.nvidia.com/en-us/design-visualization/rtx-a4000/"],
  "rtx-4090": ["https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/"],
  "rtx-3090": ["https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/"],
  "rtx-4080": ["https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080/"],
  "rtx-3080": ["https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080/"],
  "m2u-800": [
    "https://www.apple.com/newsroom/2023/06/apple-unveils-m2-ultra/",
    "https://www.apple.com/mac-studio/specs/",
  ],
  "m2u-192": [
    "https://www.apple.com/newsroom/2023/06/apple-unveils-m2-ultra/",
    "https://www.apple.com/mac-studio/specs/",
  ],
  "m3max-64": ["https://www.apple.com/macbook-pro/specs/"],
  "m4max-128": ["https://www.apple.com/macbook-pro/specs/"],
};

/** Does the URL actually answer? */
async function resolves(url) {
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { Range: "bytes=0-1023", "User-Agent": "tokcalc-catalog-provenance" },
      redirect: "follow",
    });
    return { ok: res.ok, status: res.status, final: res.url };
  } catch (err) {
    return { ok: false, status: 0, error: err.message };
  }
}

/**
 * A HuggingFace model page answers 401 for a GATED repo — the repo exists,
 * it just needs a licence click. That is still a citable source, so treat
 * 401 as "exists". The API endpoint is the reliable existence check.
 */
async function resolvesModel(url) {
  const repo = url.replace("https://huggingface.co/", "");
  const api = await resolves(`https://huggingface.co/api/models/${repo}`);
  if (api.ok) return { ok: true, status: api.status, final: url };
  const page = await resolves(url);
  if (page.status === 401) return { ok: true, status: 401, final: url, gated: true };
  return { ok: false, status: page.status || api.status, final: url };
}

/** Try each candidate URL in order; the first that answers wins. */
async function resolvesAny(candidates) {
  let last = { ok: false, status: 0 };
  for (const url of candidates) {
    const r = await resolves(url);
    if (r.ok) return { ...r, final: r.final || url };
    last = r;
  }
  return last;
}

async function checkAll(candidates, check) {
  const entries = Object.entries(candidates);
  const results = new Map();
  let cursor = 0;
  async function worker() {
    while (cursor < entries.length) {
      const [id, value] = entries[cursor++];
      const r = await check(value);
      results.set(id, r);
      const flag = r.ok ? "ok  " : "MISS";
      const extra = r.gated ? " (gated)" : "";
      console.log(`  ${flag} ${id.padEnd(20)} ${r.status} ${extra}`);
    }
  }
  await Promise.all(Array.from({ length: 6 }, worker));
  return results;
}

const today = new Date().toISOString().slice(0, 10);

console.log("Checking model sources…");
const modelResults = await checkAll(MODEL_SOURCES, resolvesModel);
console.log("\nChecking accelerator sources…");
const accelResults = await checkAll(ACCELERATOR_SOURCES, resolvesAny);

const modelsPath = join(REPO, "catalog", "models.json");
const accelPath = join(REPO, "catalog", "accelerators.json");

const models = JSON.parse(readFileSync(modelsPath, "utf8"));
const accels = JSON.parse(readFileSync(accelPath, "utf8"));

let updated = 0;
for (const m of models.records) {
  if (m.provenance.source !== "unspecified") continue;
  const r = modelResults.get(m.id);
  if (!r?.ok) continue;
  m.provenance = {
    source: "model-card",
    sourceUrl: r.final || MODEL_SOURCES[m.id],
    retrievedAt: today,
    notes: r.gated
      ? "The upstream repository is licence-gated: the model card is readable after accepting the licence."
      : undefined,
  };
  if (!m.provenance.notes) delete m.provenance.notes;
  updated++;
}

let updatedAccels = 0;
for (const a of accels.records) {
  if (a.provenance.source !== "unspecified") continue;
  const r = accelResults.get(a.id);
  if (!r?.ok) continue;
  a.provenance = {
    source: "vendor-spec",
    sourceUrl: r.final || ACCELERATOR_SOURCES[a.id],
    retrievedAt: today,
  };
  if (!a.sourceUrl) a.sourceUrl = a.provenance.sourceUrl;
  updatedAccels++;
}

console.log(`\nmodels: ${updated} backfilled, accelerators: ${updatedAccels} backfilled`);

const stillUnspecified = [
  ...models.records.filter((m) => m.provenance.source === "unspecified").map((m) => m.id),
  ...accels.records.filter((a) => a.provenance.source === "unspecified").map((a) => a.id),
];
if (stillUnspecified.length > 0) {
  console.log(`still unspecified (no verified source): ${stillUnspecified.join(", ")}`);
}

if (!dryRun) {
  writeFileSync(modelsPath, JSON.stringify(models, null, 2) + "\n", "utf8");
  writeFileSync(accelPath, JSON.stringify(accels, null, 2) + "\n", "utf8");
  console.log("written");
} else {
  console.log("(dry run — nothing written)");
}
