/**
 * Inspect a HuggingFace config.json + safetensors index and print the
 * architecture fields the catalog cares about.
 *
 *   node scripts/hf-inspect.mjs <repo>
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const repo = process.argv[2];
if (!repo) {
  console.error("usage: node scripts/hf-inspect.mjs <repo>");
  process.exit(1);
}

const dir = join(process.env.TEMP, "hfcache");
mkdirSync(dir, { recursive: true });
const slug = repo.replace(/\//g, "__");

/** Fetch a checkpoint file into the local cache if it is not there yet. */
async function cache(name, url) {
  const path = join(dir, name);
  if (existsSync(path)) return path;
  console.log(`  fetching ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} fetching ${url}`);
  writeFileSync(path, Buffer.from(await res.arrayBuffer()));
  return path;
}

await cache(`${slug}.config.json`, `https://huggingface.co/${repo}/resolve/main/config.json`);
await cache(
  `${slug}.index.json`,
  `https://huggingface.co/${repo}/resolve/main/model.safetensors.index.json`,
);

const cfg = JSON.parse(readFileSync(join(dir, `${slug}.config.json`), "utf8"));
const idx = JSON.parse(readFileSync(join(dir, `${slug}.index.json`), "utf8"));

const text = JSON.stringify(cfg, null, 2);
console.log(`=== ${repo} ===`);
console.log(`model_type: ${cfg.model_type}`);
console.log(`architectures: ${JSON.stringify(cfg.architectures)}`);

// Show the sub-config that actually holds the interesting numbers.
const inner =
  cfg.text_config ??
  cfg.llm_config ??
  cfg.language_config ??
  cfg.thinker_config ??
  cfg;

const KEYS = [
  "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
  "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
  "sliding_window", "layer_types", "full_attention_interval",
  "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "qk_head_dim", "v_head_dim",
  "mla_use_nope", "index_head_dim", "index_n_heads", "indexer_types", "index_topk",
  "linear_attn_config", "linear_num_key_heads", "linear_num_value_heads",
  "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
  "mamba_ssm_dtype", "mtp_num_hidden_layers", "num_nextn_predict_layers",
  "num_experts", "num_experts_per_token", "num_shared_experts", "n_routed_experts",
  "num_experts_per_tok", "n_shared_experts", "moe_intermediate_size",
  "routed_expert_hidden_size", "first_k_dense_replace", "quantization_config",
  "rope_scaling", "torch_dtype", "dtype",
];

for (const k of KEYS) {
  if (inner[k] !== undefined) {
    const v = inner[k];
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    console.log(`  ${k}: ${s.length > 220 ? s.slice(0, 220) + " …" : s}`);
  }
}

// Layer-type histograms for hybrid models.
if (Array.isArray(inner.layer_types)) {
  const counts = {};
  for (const t of inner.layer_types) counts[t] = (counts[t] ?? 0) + 1;
  console.log(`  layer_types histogram: ${JSON.stringify(counts)}`);
  console.log(`  layer_types first 12: ${JSON.stringify(inner.layer_types.slice(0, 12))}`);
}
if (Array.isArray(inner.indexer_types)) {
  const counts = {};
  for (const t of inner.indexer_types) counts[t] = (counts[t] ?? 0) + 1;
  console.log(`  indexer_types histogram: ${JSON.stringify(counts)}`);
}

if (inner.linear_attn_config) {
  console.log("  linear_attn_config:");
  for (const [k, v] of Object.entries(inner.linear_attn_config)) {
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    console.log(`    ${k}: ${s.length > 400 ? s.slice(0, 400) + " …" : s}`);
  }
}

console.log(`  total_size (bytes): ${idx.metadata?.total_size ?? "n/a"}`);
console.log(`  tensor count: ${Object.keys(idx.weight_map ?? {}).length}`);
console.log(`  config bytes: ${text.length}`);
