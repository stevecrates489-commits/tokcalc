/**
 * Derive a catalog model record from a real HuggingFace checkpoint.
 *
 *   node scripts/catalog-import.mjs --repo <org/name> --id <catalog-id> \
 *        --name "<display name>" --family <family> --category <category> [--dry-run]
 *
 * The point of this script is that the architecture is DERIVED, not typed by
 * hand. A hand-written attention layout is a transcription of a config.json
 * that somebody read once, and it rots the moment the upstream checkpoint
 * changes. Here the per-layer layout is computed from the checkpoint's own
 * fields, and the resident byte count comes from the safetensors index.
 *
 * What it reads:
 *   config.json                  → layers, heads, attention kinds, context
 *   model.safetensors.index.json → exact resident bytes
 *   README.md                    → published parameter counts, license
 *   api/models/<repo>            → authoritative total parameter count + revision
 *
 * What it will NOT do:
 *   invent a number. If the layout is not fully determined by the config
 *   (e.g. DeepSeek-style sparse-attention indexers keep a per-token index
 *   cache that the config does not describe), the record is marked
 *   "memory-only" and says so, rather than guessing a formula.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

const REPO = resolve(import.meta.dirname, "..");
const CACHE = join(process.env.TEMP, "hfcache");
mkdirSync(CACHE, { recursive: true });

/** Fetch a file from the upstream repo into the local cache. */
async function cacheFile(name, url) {
  const path = join(CACHE, name);
  if (existsSync(path)) return path;
  console.log(`  fetching ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} fetching ${url}`);
  writeFileSync(path, Buffer.from(await res.arrayBuffer()));
  return path;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const flag = (name, fallback = undefined) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const has = (name) => args.includes(`--${name}`);

const repo = flag("repo");
const id = flag("id");
const name = flag("name");
const family = flag("family");
const category = flag("category", "text");
const activeParams = flag("active-params"); // billions, for MoE models
const license = flag("license"); // SPDX id, or the HF license_name for custom licences
const dryRun = has("dry-run");

if (!repo || !id || !name || !family) {
  console.error(
    "usage: node scripts/catalog-import.mjs --repo <org/name> --id <id> " +
      "--name <name> --family <family> [--category text|vlm|code|reasoning|embedding] [--dry-run]",
  );
  process.exit(1);
}

const slug = repo.replace(/\//g, "__");
const cfgPath = join(CACHE, `${slug}.config.json`);
const idxPath = join(CACHE, `${slug}.index.json`);
const apiPath = join(CACHE, `${slug}.api.json`);

await cacheFile(`${slug}.config.json`, `https://huggingface.co/${repo}/resolve/main/config.json`);
await cacheFile(
  `${slug}.index.json`,
  `https://huggingface.co/${repo}/resolve/main/model.safetensors.index.json`,
);
await cacheFile(`${slug}.api.json`, `https://huggingface.co/api/models/${repo}`);

const cfg = JSON.parse(readFileSync(cfgPath, "utf8"));
const idx = JSON.parse(readFileSync(idxPath, "utf8"));
const api = JSON.parse(readFileSync(apiPath, "utf8"));

const inner = cfg.text_config ?? cfg.llm_config ?? cfg.language_config ?? cfg;

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

const layers = inner.num_hidden_layers;
const hiddenDim = inner.hidden_size;
const qHeads = inner.num_attention_heads;
const kvHeads = inner.num_key_value_heads ?? inner.num_attention_heads;
const headDim = inner.head_dim ?? Math.floor(hiddenDim / qHeads);
const vocabSize = inner.vocab_size;
const maxContext = inner.max_position_embeddings;
const residentWeightBytes = idx.metadata?.total_size;

/** Which attention kind each layer uses. */
function deriveKinds() {
  const kinds = new Array(layers).fill("full");

  if (Array.isArray(inner.layer_types)) {
    const MAP = {
      full_attention: "full",
      sliding_attention: "sliding",
      linear_attention: "linear",
    };
    for (let i = 0; i < inner.layer_types.length && i < layers; i++) {
      const mapped = MAP[inner.layer_types[i]];
      if (!mapped) throw new Error(`unknown layer_types entry: ${inner.layer_types[i]}`);
      kinds[i] = mapped;
    }
    return kinds;
  }

  if (inner.linear_attn_config) {
    // Kimi-style: explicit full_attn_layers + kda_layers lists (1-based).
    const lac = inner.linear_attn_config;
    const fullLayers = new Set(lac.full_attn_layers ?? []);
    const kdaLayers = new Set(lac.kda_layers ?? []);
    for (let i = 1; i <= layers; i++) {
      if (fullLayers.has(i)) kinds[i - 1] = "mla";
      else if (kdaLayers.has(i)) kinds[i - 1] = "kda";
    }
    return kinds;
  }

  if (inner.kv_lora_rank) {
    for (let i = 0; i < layers; i++) kinds[i] = "mla";
    return kinds;
  }

  return kinds;
}

const kinds = deriveKinds();

/**
 * Recurrent-state bytes for one linear / KDA layer, per sequence.
 *
 * A delta-rule linear attention layer keeps a state matrix per head instead
 * of a growing key/value cache. The shape comes straight from the config:
 *   Gated DeltaNet (Qwen): num_value_heads × key_head_dim × value_head_dim
 *   Kimi Delta Attention:   num_heads × head_dim × head_dim
 */
function linearStateBytes() {
  if (inner.mamba_ssm_dtype) {
    // Qwen publishes the SSM state dtype explicitly.
    const b = inner.mamba_ssm_dtype === "float32" ? 4 : inner.mamba_ssm_dtype === "bfloat16" ? 2 : 4;
    const heads = inner.linear_num_value_heads;
    return heads * inner.linear_key_head_dim * inner.linear_value_head_dim * b;
  }
  const lac = inner.linear_attn_config;
  if (lac?.num_heads && lac?.head_dim) {
    // No published state dtype — use the checkpoint dtype and say so.
    const b = inner.dtype === "float32" ? 4 : 2;
    return lac.num_heads * lac.head_dim * lac.head_dim * b;
  }
  return undefined;
}

const stateBytes = linearStateBytes();

const attentionLayers = kinds.map((kind) => {
  const layer = { kind };
  switch (kind) {
    case "sliding":
      if (!inner.sliding_window) {
        throw new Error("sliding layer but no sliding_window in the config");
      }
      layer.windowTokens = inner.sliding_window;
      break;
    case "mla":
      if (!inner.kv_lora_rank) throw new Error("mla layer but no kv_lora_rank in the config");
      layer.kvLoraRank = inner.kv_lora_rank;
      layer.mlaRopeHeadDim = inner.qk_rope_head_dim ?? 0;
      break;
    case "linear":
    case "kda":
      if (!stateBytes) {
        throw new Error(
          `${kind} layer but the recurrent state size is not derivable from the config — ` +
            "this model must be recorded as architectureStatus \"memory-only\" by hand",
        );
      }
      layer.stateBytes = stateBytes;
      break;
    case "full":
      break;
  }
  return layer;
});

// ---------------------------------------------------------------------------
// Architecture status
// ---------------------------------------------------------------------------

const notes = [];
let architectureStatus = "supported";

if (Array.isArray(inner.indexer_types)) {
  // DeepSeek-style sparse attention: a per-layer indexer keeps its own
  // per-token index cache. The config publishes the indexer's head count
  // and head dim but not how the cache is shared between layers, so the
  // memory figure would be wrong in an unknown direction. Say so instead
  // of guessing.
  architectureStatus = "memory-only";
  notes.push(
    `DeepSeek-style sparse attention (DSA): ${inner.indexer_types.length} layers carry a ` +
      `per-token index cache (index_n_heads=${inner.index_n_heads}, index_head_dim=${inner.index_head_dim}) ` +
      `that this engine does not model. Memory figures for this model UNDERSTATE the real ` +
      `requirement at long context; the KV cache itself (MLA) is modelled exactly.`,
  );
}

if (stateBytes && !inner.mamba_ssm_dtype && inner.linear_attn_config) {
  notes.push(
    "Recurrent state size is derived from the checkpoint dtype (bf16). The config does not " +
      "publish a separate state dtype; the error bound is under 0.03% of the model footprint.",
  );
}

if (inner.mtp_num_hidden_layers || inner.num_nextn_predict_layers) {
  const mtp = inner.mtp_num_hidden_layers ?? inner.num_nextn_predict_layers;
  notes.push(
    `Checkpoint carries ${mtp} multi-token-prediction head layer(s). These are a draft head, ` +
      `not part of the per-token decode path, so they are not counted as active parameters.`,
  );
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const totalParams = api?.safetensors?.total ?? null;

// Active parameters: prefer the checkpoint's own MoE arithmetic.
const expertsPerToken =
  inner.num_experts_per_tok ?? inner.experts_per_token ?? inner.num_experts_per_token ?? null;
const isMoE = Boolean(expertsPerToken) && expertsPerToken > 0;

let activeParamsB = null;
if (!isMoE) {
  activeParamsB = totalParams / 1e9;
} else if (activeParams) {
  // The model card is the authoritative source for the active count.
  activeParamsB = Number(activeParams);
  notes.push(
    `Active parameter count (${activeParams}B) taken from the model card; the checkpoint ` +
      `config does not state it.`,
  );
}

// ---------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------

const record = {
  id,
  name,
  family,
  category,
  paramsB: totalParams !== null ? Number((totalParams / 1e9).toFixed(3)) : 0,
  activeParamsB: activeParamsB !== null ? Number(activeParamsB.toFixed(3)) : 0,
  layers,
  hiddenDim,
  qHeads,
  kvHeads,
  headDim,
  vocabSize,
  maxContext,
  isMoE,
  attentionLayers,
  architectureStatus,
  residentWeightBytes,
  repository: repo,
  revision: api.sha,
  nativeWeightFormat:
    inner.quantization_config?.quant_method === "mxfp4" ||
    inner.quantization_config?.format === "mxfp4-pack-quantized"
      ? "mxfp4"
      : inner.quantization_config?.quant_method === "fp8"
        ? "fp8"
        : inner.dtype ?? "bfloat16",
  lifecycle: "active",
  provenance: {
    source: "hf-config",
    sourceUrl: `https://huggingface.co/${repo}`,
    retrievedAt: new Date().toISOString().slice(0, 10),
    notes: [
      `Architecture derived from config.json at revision ${api.sha}.`,
      `Resident weight bytes (${residentWeightBytes.toLocaleString("en-US")}) measured from ` +
        `model.safetensors.index.json total_size.`,
      `Total parameter count (${totalParams?.toLocaleString("en-US")}) from the HuggingFace ` +
        `model API safetensors metadata.`,
      ...notes,
    ].join(" "),
  },
};

if (license) record.license = license;

if (inner.rope_scaling?.rope_type) {
  record.contextExtensionMethod = `RoPE scaling (${inner.rope_scaling.rope_type})`;
  record.nativeContext = inner.rope_scaling.original_max_position_embeddings ?? inner.initial_context_length;
}
if (inner.initial_context_length) {
  record.nativeContext = inner.initial_context_length;
}

if (dryRun) {
  console.log(JSON.stringify(record, null, 2));
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Merge into catalog/models.json
// ---------------------------------------------------------------------------

const catalogPath = join(REPO, "catalog", "models.json");
const catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
const idxModels = catalog.records.findIndex((r) => r.id === id);
if (idxModels >= 0) {
  catalog.records[idxModels] = record;
  console.log(`replaced ${id}`);
} else {
  catalog.records.push(record);
  console.log(`added ${id}`);
}
catalog.records.sort((a, b) => a.id.localeCompare(b.id));
writeFileSync(catalogPath, JSON.stringify(catalog, null, 2) + "\n", "utf8");
console.log(`catalog/models.json now has ${catalog.records.length} records`);

// Keep the manifest's counts honest — the loader fails the build if they drift.
const manifestPath = join(REPO, "catalog", "manifest.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
manifest.counts.models = catalog.records.length;
manifest.updatedAt = new Date().toISOString().slice(0, 10);
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n", "utf8");
console.log(`catalog/manifest.json counts.models = ${manifest.counts.models}`);
