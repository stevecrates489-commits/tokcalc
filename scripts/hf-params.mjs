/**
 * Authoritative parameter accounting straight from the safetensors headers.
 *
 * The safetensors format is:
 *   [8 bytes little-endian u64: header length][header JSON][tensor data]
 *
 * A Range request for the first (8 + header_len) bytes of each shard gives
 * every tensor's name, dtype and shape. That is enough to count total
 * parameters AND — for MoE models — the parameters that are actually ACTIVE
 * per token, which the model card almost never publishes and which the
 * decode-speed formula needs.
 *
 *   node scripts/hf-params.mjs <repo> [shardLimit]
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

const repo = process.argv[2];
if (!repo) {
  console.error("usage: node scripts/hf-params.mjs <repo> [shardLimit]");
  process.exit(1);
}
const shardLimit = Number(process.argv[3] ?? 0); // 0 = all shards

const dir = join(process.env.TEMP, "hfcache");
const slug = repo.replace(/\//g, "__");
const idx = JSON.parse(readFileSync(join(dir, `${slug}.index.json`), "utf8"));
const cfg = JSON.parse(readFileSync(join(dir, `${slug}.config.json`), "utf8"));

const allShards = [...new Set(Object.values(idx.weight_map ?? {}))];
const shards = shardLimit > 0 ? allShards.slice(0, shardLimit) : allShards;

console.log(`=== ${repo} ===`);
console.log(`  shards: ${allShards.length} (reading ${shards.length}), tensors: ${Object.keys(idx.weight_map).length}`);

async function head(url, len) {
  const res = await fetch(url, { headers: { Range: `bytes=0-${len - 1}` } });
  if (!res.ok) throw new Error(`${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

/** Read one shard's header, returning { name: {dtype, size} }. */
async function readShard(shard) {
  const url = `https://huggingface.co/${repo}/resolve/main/${shard}`;
  const eight = await head(url, 8);
  const headerLen = Number(eight.readBigUInt64LE(0));
  const buf = await head(url, 8 + headerLen);
  const len = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + len).toString("utf8"));
  const out = new Map();
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const size = Array.isArray(meta.shape) ? meta.shape.reduce((a, b) => a * b, 1) : 1;
    out.set(name, { dtype: meta.dtype, size });
  }
  return out;
}

// Concurrency-limited parallel read.
const shapes = new Map();
let cursor = 0;
const CONCURRENCY = 8;
async function worker() {
  while (cursor < shards.length) {
    const i = cursor++;
    const shard = shards[i];
    try {
      const s = await readShard(shard);
      for (const [k, v] of s) shapes.set(k, v);
    } catch (err) {
      console.log(`  (skipping ${shard}: ${err.message})`);
    }
    if ((i + 1) % 20 === 0) process.stdout.write(`  ...${i + 1}/${shards.length}\r`);
  }
}
await Promise.all(Array.from({ length: CONCURRENCY }, worker));
process.stdout.write(" ".repeat(40) + "\r");

// --- Classification ---------------------------------------------------------

const CFG = cfg.text_config ?? cfg;
const EXPERTS_PER_TOKEN =
  CFG.num_experts_per_tok ?? CFG.experts_per_token ?? CFG.num_experts_per_token ?? 1;

const buckets = {
  embed: 0,
  lmHead: 0,
  attention: 0,
  indexer: 0,
  sharedExpert: 0,
  routedExpert: 0,
  router: 0,
  denseMlp: 0,
  other: 0,
};
const perExpertLayer = new Map(); // layer -> summed params of ALL its experts
const expertIds = new Set(); // distinct expert ids
let fusedExpertParams = 0; // params held in fused "experts.gate_up_proj" tensors
const otherNames = [];

const add = (k, n) => { buckets[k] += n; };

for (const [name, m] of shapes) {
  const s = m.size;
  if (/embed_tokens|\.wte\./.test(name)) add("embed", s);
  else if (/lm_head/.test(name)) add("lmHead", s);
  else if (/indexer|\bindex_/.test(name)) add("indexer", s);
  else if (/self_attn|attn/.test(name)) add("attention", s);
  else if (/shared_expert|shared_mlp/.test(name)) add("sharedExpert", s);
  else if (/experts\.\d+\./.test(name)) {
    add("routedExpert", s);
    const m2 = name.match(/experts\.(\d+)\./);
    if (m2) expertIds.add(m2[1]);
    const layer = name.match(/layers\.(\d+)\./)?.[1] ?? "?";
    if (!perExpertLayer.has(layer)) perExpertLayer.set(layer, 0);
    perExpertLayer.set(layer, perExpertLayer.get(layer) + s);
  } else if (/experts\.(gate_up_proj|down_proj|gate_proj|up_proj|w[123])/.test(name)) {
    // Fused expert tensor: holds every expert of the layer in one tensor.
    add("routedExpert", s);
    fusedExpertParams += s;
  } else if (/\.gate\.|router/.test(name)) add("router", s);
  else if (/(down_proj|up_proj|gate_proj|gate_up_proj)/.test(name)) add("denseMlp", s);
  else { add("other", s); if (otherNames.length < 8) otherNames.push(name); }
}

const total = Object.values(buckets).reduce((a, b) => a + b, 0);

// One expert's worth of params.
//  - per-expert naming ("experts.12.*"): the same expert index appears in
//    every layer, so divide by (distinct indices × layers that have experts).
//  - fused naming ("experts.gate_up_proj"): one tensor holds every expert, so
//    divide by the number of local experts.
const numLocalExperts =
  CFG.num_local_experts ?? CFG.n_routed_experts ?? CFG.num_experts ?? expertIds.size;
const oneExpert = expertIds.size > 0
  ? buckets.routedExpert / (expertIds.size * Math.max(1, perExpertLayer.size))
  : fusedExpertParams > 0
    ? fusedExpertParams / numLocalExperts
    : 0;

const active =
  buckets.embed +
  buckets.lmHead +
  buckets.attention +
  buckets.indexer +
  buckets.sharedExpert +
  buckets.router +
  buckets.denseMlp +
  oneExpert * EXPERTS_PER_TOKEN;

const fmt = (n) => (n / 1e9).toFixed(3) + " B";
console.log(`  tensors read: ${shapes.size}`);
console.log(`  TOTAL params:      ${total.toLocaleString("en-US")}  (${fmt(total)})`);
console.log(`  ACTIVE params/tok: ${Math.round(active).toLocaleString("en-US")}  (${fmt(active)})`);
console.log(`  experts/token: ${EXPERTS_PER_TOKEN}, one expert = ${fmt(oneExpert)}`);
console.log("  breakdown (B):");
for (const [k, v] of Object.entries(buckets)) {
  if (v > 0) console.log(`    ${k.padEnd(14)} ${fmt(v)}`);
}
if (otherNames.length) console.log(`  other sample names: ${otherNames.join(", ")}`);
