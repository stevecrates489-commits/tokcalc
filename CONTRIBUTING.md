# Contributing to tokcalc

First of all — **thank you** for considering a contribution. tokcalc is a community-maintained capacity planner and every contribution makes the LLM serving ecosystem more transparent.

This document explains how to add models, GPUs, quantization formats, and benchmark data. The bar for acceptance is **source-cited, defensible data** — not vibes.

## Code of conduct

Be kind. Be precise. Disclose provenance. Don't market. Don't fabricate.

## Quick ways to contribute

| Contribution | Difficulty | Where to look |
|---|---|---|
| Add a missing model | Easy | `src/lib/token-calc.ts` → `MODELS` array |
| Add a missing GPU | Easy | `src/lib/token-calc.ts` → `GPUS` array |
| Add a quantization format | Easy | `src/lib/token-calc.ts` → `QUANTIZATIONS` array |
| Fix a typo / UI bug | Trivial | Anywhere in `src/app/page.tsx` |
| Improve a formula | Hard | `calculate()` in `src/lib/token-calc.ts` |
| Submit a benchmark | Medium | Open an issue with the schema below |

## How to add a model

Add a row to the `MODELS` array in [`src/lib/token-calc.ts`](src/lib/token-calc.ts). The schema:

```typescript
{
  id: "llama3-15b",                    // unique, lowercase, no spaces
  name: "Llama 3.2 15B",               // human-readable, includes variant
  family: "Llama",                     // for dropdown grouping
  category: "text",                    // "text" | "vlm" | "embedding" | "code" | "reasoning"
  paramsB: 15.0,                       // TOTAL params in billions (e.g., 671 for DeepSeek V3 MoE)
  activeParamsB: 15.0,                // ACTIVE params per token (for MoE; = paramsB for dense)
  layers: 64,                          // num_hidden_layers from config.json
  hiddenDim: 5120,                     // hidden_size from config.json
  qHeads: 40,                          // num_attention_heads
  kvHeads: 8,                          // num_key_value_heads (for GQA)
  headDim: 128,                        // head_dim (or hidden_size / num_attention_heads)
  vocabSize: 128256,                   // vocab_size
  maxContext: 131072,                  // max_position_embeddings
  isMoE: false,                        // true if Mixture-of-Experts
}
```

### Source requirements (mandatory)
- **Architecture fields** (`layers`, `hiddenDim`, `qHeads`, `kvHeads`, `headDim`, `vocabSize`, `maxContext`) MUST come from the model's `config.json` on HuggingFace — not from blog posts or third-party summaries.
- **Total/active params** for MoE models MUST come from the official release announcement or model card.
- Add a comment with the source URL:
  ```typescript
  // Refs: https://huggingface.co/meta-llama/Llama-3.2-15B
  ```

### Don't fabricate
- If a field isn't published (e.g., DeepSeek V3's exact architecture), **leave it as a placeholder and add a TODO comment**, or omit the entry entirely. Don't guess.
- For announced-but-unreleased models (e.g., Llama 4 Behemoth), mark the row with `// announced — not yet released` and only add if there's an official model card or config.

## How to add a GPU

Add a row to the `GPUS` array. The schema:

```typescript
{
  id: "rtx-5090",
  name: "RTX 5090 32GB",
  vendor: "NVIDIA",                    // "NVIDIA" | "AMD" | "Intel" | "Google" | "Apple" | "Groq" | "Cerebras" | ...
  category: "consumer",               // "datacenter" | "workstation" | "consumer" | "mac" | "tpu" | "lpu" | "wse" | "legacy"
  memBandwidthGbps: 1792,             // HBM/unified memory bandwidth in GB/s
  flopsTflops: 105,                   // FP16/BF16 dense TFLOPS. Use `null` if vendor doesn't publish (e.g., B200)
  vramGb: 32,                         // VRAM in GB
  nvlinkGbps: 0,                      // 0 if no NVLink
  usdPerHour: 0.50,                   // Typical cloud $/hr; `null` if quote-only
  year: 2025,                         // Release year
  note: "Optional note about caveats", // e.g., "ROCm; performance depends on ROCm version"
  sourceUrl: "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series",
}
```

### Critical guardrails (per research brief)
- **Never use FP4/FP8/sparse FLOPS as FP16 dense**. NVIDIA Blackwell B200/B300 only publishes FP4/FP8 specs — set `flopsTflops: null` and add a note. The calc engine handles this gracefully with a conservative fallback.
- **Apple Silicon unified memory ≠ GPU VRAM**. Bandwidth figures are accurate but thermal behavior differs. Always include a `note` for Apple.
- **Cloud $/hr is approximate + region-specific**. Use `null` for "quote required". Add a comment with the source URL.

## How to add a quantization format

Add a row to the `QUANTIZATIONS` array. The schema:

```typescript
{
  id: "gguf-q4km",
  label: "GGUF Q4_K_M",
  bytesPerParam: 0.55,                 // EFFECTIVE bytes per param (averaged across tensor mix)
  efficiency: 0.90,                   // Dequant overhead multiplier. 1.0 = no overhead, 0.85 = 15% slower
                                       // >1.0 means FASTER than FP16 (e.g., FP8 on H100 = 1.5)
  description: "Recommended sweet spot for local Llama/Mistral on consumer GPUs.",
  family: "gguf",                      // "float" | "gguf" | "gptq" | "awq" | "exl2" | "fp8" | "nvfp4"
  useCase: "Recommended for local inference",  // Optional
}
```

### Source requirements
- For GGUF variants, use measured file sizes from the [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) or from actual `.gguf` files on HuggingFace.
- For `efficiency`, cite a benchmark source if possible. Don't invent numbers.

## How to submit a benchmark

We're building a benchmark provenance database. To submit a benchmark, open an issue with this template:

```markdown
**Model:** DeepSeek V3 671B (MoE)
**Quantization:** FP8
**GPU:** 8× H100 SXM5 80GB
**Engine:** vLLM v0.29.0
**Configuration:** `--max-num-seqs=256 --max-num-batched-tokens=8192 --tensor-parallel-size=8`
**Workload:**
- Input distribution: 4096 mean / 95th percentile 8192 tokens
- Output: 512 mean / 95th percentile 1024 tokens
- Concurrency: 256
- QPS: 18

**Measured:**
- Throughput (output tok/s, aggregate): 412
- TTFT p50: 220 ms
- TTFT p95: 580 ms
- ITL p50: 24 ms
- ITL p95: 38 ms

**Source URL:** https://github.com/vllm-project/vllm/issues/12345
**Methodology:** vLLM `benchmark_serving.py` with 1000 requests
**Date:** 2026-09-21
**Submitter:** @yourhandle
```

### Confidence tiers
1. **Tier 1**: Official vLLM/SGLang/TensorRT-LLM benchmarks, MLPerf submissions
2. **Tier 2**: Reproducible benchmark runs (you ran it yourself with documented config)
3. **Tier 3**: Community observations (less rigorous, clearly labeled)

All three are accepted, but they're displayed differently in the UI.

## How to improve a formula

The math lives in `calculate()` in `src/lib/token-calc.ts`. Every formula has a comment citing the source.

When improving a formula:
1. Open an issue first describing the problem + proposed fix
2. Cite the source (paper, blog, official docs)
3. Don't introduce "magic numbers" without justification
4. Update the comments to reflect the new source
5. Test against benchmark data if available

## Development setup

```bash
git clone https://github.com/tokcalc/tokcalc.git
cd tokcalc
bun install
bun run dev
```

Open `http://localhost:3000`. Make changes. Verify with `bun run lint`.

## PR checklist

Before opening a PR:
- [ ] `bun run lint` passes (warnings OK, errors not)
- [ ] All new model/GPU/quant entries have a source URL in a comment
- [ ] No fabricated data — leave `null` for unknown fields
- [ ] UI changes verified in both light and dark mode
- [ ] Description explains **why**, not just **what**
- [ ] No marketing language ("revolutionary", "the first ever", etc.)
- [ ] No console.log statements left behind

## License

By contributing, you agree that your contributions are licensed under the Apache 2.0 license (see [LICENSE](LICENSE)).

For benchmark data contributions, you agree they'll be licensed under CC-BY-SA 4.0 (attribution + share-alike) — this protects the community's data investment.
