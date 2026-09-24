<div align="center">

# tokcalc

### The open-source LLM serving capacity planner

**Plan your LLM deployment before you rent the GPUs.**

[![Live demo](https://img.shields.io/badge/live-demo-10b981?style=flat-square)](https://tokcalc.dev)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat-square)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-10b981?style=flat-square)](CONTRIBUTING.md)
[![Made with Next.js](https://img.shields.io/badge/made_with-Next.js_16-black?style=flat-square)](https://nextjs.org)

</div>

---

<div align="center">

![tokcalc demo](/public/tokcalc-demo.gif)

*Switch model → multi-GPU → long-context capacity planner → Build vs Buy → Reference catalog*

</div>

---

tokcalc turns your LLM traffic, context length, latency SLOs, cache behavior,
and model choice into a **defensible serving topology and cost plan** —
with transparent formulas and cited benchmarks.

It's the engineering-grade pre-deployment decision layer for LLM inference.
Not another static "tokens per second" calculator.

## Try it now

**[tokcalc.dev](https://tokcalc.dev)** — no signup, no tracking, no paywall.

Pick a model, a GPU, and a workload. Get an instant capacity plan:

- Generation speed (tok/s)
- Time-to-first-token (TTFT) + inter-token latency (ITL)
- VRAM budget with KV-cache sizing
- Multi-GPU topology recommendation (Single GPU → TP×2/4/8 → Context Parallel)
- Monthly cost + break-even vs API pricing
- **Shareable URL** — your config encoded in the URL hash, send to colleagues

## Why tokcalc?

The market has dozens of "tokens per second" calculators and self-host-vs-API
break-even tools (induwara.lk, gigagpu, kickllm, cloudparity, curlscape,
profitable.ai). None of them are unified capacity planners.

### What tokcalc answers that competitors can't

> *"Can I serve Qwen 2.5 72B at 128K context on 2× H100 with 20 concurrent users?"*

> *"How many H200s do I need for 1,000 req/min with P95 TTFT < 2s?"*

> *"Does FP8 or AWQ save more money once quality, KV cache, and engine support are included?"*

> *"At what daily volume does an H100 beat GPT-4o pricing?"*

> *"What happens to cost and latency if an agent makes 8 model calls, has 3 tool calls, and its context grows by 5K tokens each turn?"*

> *"Would prefix caching, continuous batching, or PD disaggregation save more for this workload?"*

## Features

### Calculator tab

| Feature | What it computes |
|---|---|
| **Model fit / VRAM** | Will the model + KV cache fit in the GPU's memory? |
| **Throughput** | Decode tok/s (per-stream) + aggregate (batched) + prefill tok/s |
| **Latency split** | Time-to-first-token (= prefill) + inter-token latency (= decode) |
| **Continuous batching** | User-tunable 1.0–4× multiplier (cited 1.5–4× SOSP range) |
| **Reasoning tokens** | Hidden reasoning budget added to billed output (o1 / R1 / Claude thinking) |
| **Prompt caching** | Self-hosted vLLM APC + Anthropic 5m/1h TTL + OpenAI 50%-off cached tokens |
| **Speculative decoding** | User-tunable 1.2–4× boost factor |
| **Multi-GPU TP** | 1× → 8× tensor parallel with NVLink efficiency factor |
| **Long-context capacity** | KV memory + max concurrency + prefill time at 4K → 1M context |
| **Topology recommendation** | Single GPU → TP×2 → TP×4 → TP×8 → TP×8 + Context Parallel (RingAttention) |
| **Cost economics** | GPU $/hr → $/M output tokens → $/request → monthly cost |

### Build vs Buy tab

Independent calculator (separate state) that compares:
- **Self-host**: model + GPU + quant + utilization + batch → $/M tokens + monthly cost
- **API**: 13 providers (OpenAI / Anthropic / Gemini / Groq / DeepSeek / Mistral / Together)
- **Verdict**: Self-host cheaper / API cheaper / Not enough volume — with break-even reqs/day

### Reference tab

5 sub-tables — fully transparent, every record source-linked where available:
- **Models** (35 entries: Llama 4 Scout/Maverick, Qwen 3 family, DeepSeek V3/R1, Pixtral, BGE-M3, ...)
- **GPUs** (30 entries: H100/H200/B200/B300, AMD MI300X/MI325X, Intel Gaudi 3, TPU v5p/Trillium, Groq LPU, Cerebras WSE-3, Apple M2/M3/M4 Ultra, ...)
- **Quantization** (16 formats: FP16/BF16, GGUF Q2_K→Q8_0, GPTQ, AWQ, EXL2, FP8, NVFP4)
- **API pricing** (13 models with input/cached/output + retired/current status)
- **Cloud GPU pricing** (all GPUs with $/hr > 0 + typical providers)

### The long-context capacity planner (the differentiator)

This is the formula the Perplexity research brief called "the most important
tokcalc should visibly expose":

$$
\text{KV bytes/request} = 2 \cdot L \cdot T \cdot H_{\text{kv}} \cdot D_h \cdot B
$$

For dense attention, prefill cost grows **superlinearly** with context length:

$$
\text{prefill FLOPs} = \underbrace{2 \cdot N \cdot T}_{\text{linear}} + \underbrace{T^2 \cdot H_{\text{kv}} \cdot D_h \cdot L}_{\text{attention}}
$$

tokcalc shows you:
- Max concurrent users at 4K / 8K / 16K / 32K / 64K / 128K / 256K / 512K / 1M context
- KV memory per request at each context length
- Prefill time (with superlinear attention correction beyond 32K)
- Required topology (Single GPU → TP×2/4/8 → TP×8 + Context Parallel)
- RingAttention citation when CP is needed

## The math, transparently

Every number above comes from a formula you can inspect. No black boxes.

<details>
<summary><b>Decode tokens/sec (memory-bandwidth bound)</b></summary>

$$
\text{decode tok/sec} \approx \frac{\text{HBM BW} \cdot \eta_{\text{mem}} \cdot \text{quant\_eff}}{\text{model size}}
$$

Where:
- `HBM BW` = GPU memory bandwidth (e.g., 3350 GB/s for H100 SXM)
- `η_mem = 0.65` = typical real-world memory utilization (35% overhead)
- `quant_eff` = dequantization efficiency multiplier (1.0 for FP16, 1.5 for FP8 on H100, 0.85 for INT4)
- `model size = active_params × bytes_per_param` (uses ACTIVE params for MoE, not total)

Refs: PagedAttention paper ([arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180))
</details>

<details>
<summary><b>Prefill tokens/sec (compute bound)</b></summary>

$$
\text{prefill tok/sec} \approx \frac{\text{GPU FLOPS} \cdot \eta_{\text{compute}}}{2 \cdot \text{active params}}
$$

Where:
- `GPU FLOPS` = dense FP16/BF16 TFLOPS (sparse values not used)
- `η_compute = 0.50` = typical compute utilization
- Factor of 2 = one multiply + one add per parameter per token

For long context (>32K), the superlinear attention correction above applies.
</details>

<details>
<summary><b>Continuous batching multiplier (workload-specific)</b></summary>

$$
\text{aggregate tok/sec} = \text{decode tok/sec} \cdot \text{batch size} \cdot \text{continuous batching multiplier}
$$

**Critical caveat**: There is no universal continuous batching multiplier. vLLM
reported 14–24× vs HF Transformers (extreme), 2.2–2.5× vs TGI. SOSP paper
finds 2–4× typical vs FasterTransformer/Orca. tokcalc defaults to a
conservative **1.5×** and lets you tune.

Refs:
- [vLLM blog (2023-06-20)](https://vllm.ai/blog/2023-06-20-vllm)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [Anyscale continuous batching study](https://www.anyscale.com/blog/continuous-batching-llm-inference)
</details>

<details>
<summary><b>Prompt caching economics (Anthropic / OpenAI)</b></summary>

For shared prefix of length $T_p$, suffix of length $T_u$, output $O$,
cache hit rate $h$:

$$
\text{API input cost} = N \cdot \left[ (1-h) \cdot T_p \cdot P_{\text{write}} + h \cdot T_p \cdot P_{\text{read}} + T_u \cdot P_{\text{input}} \right]
$$

Anthropic multipliers (verified 2025-2026):
- 5-minute cache write: 1.25× base input
- 1-hour cache write: 2.0× base input
- Cache read: 0.1× base input (90% savings)

OpenAI: cached input discounted 50%, no separate write fee.

Refs:
- [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [OpenAI prompt caching guide](https://platform.openai.com/docs/guides/prompt-caching)
</details>

<details>
<summary><b>Multi-GPU topology recommendation</b></summary>

$$
\text{total needed} = \text{model weights} + (\text{KV per request} \cdot \text{batch size})
$$

Walk the smallest topology that fits:
- **Single GPU**: total ≤ VRAM × 1
- **Tensor Parallel ×2/4/8**: total ≤ VRAM × N (weights + KV split evenly)
- **TP×8 + Context Parallel**: total > VRAM × 8 — use RingAttention to shard KV across nodes

Refs: [RingAttention paper](https://arxiv.org/abs/2310.01889)
</details>

<details>
<summary><b>Self-host vs API break-even</b></summary>

$$
\text{self-host $/M tokens} = \frac{\text{GPU $/hr}}{3600 \cdot \text{effective tok/s} \cdot \text{utilization}} \cdot 10^6
$$

$$
\text{break-even req/day} = \frac{\text{monthly self-host cost}}{30 \cdot \text{API cost per request}}
$$

The decisive term is **effective utilization** — not peak throughput. A GPU
running at 10% utilization pays 10× more per token than the theoretical minimum.
</details>

## Comparison with adjacent tools

| Capability | induwara / techfuelhq / pcmasterstudio | gigagpu / kickllm / cloudparity | HF Open LLM Leaderboard / MLPerf | **tokcalc** |
|---|:---:|:---:|:---:|:---:|
| Model × GPU × quant tok/s | ✓ | some | some | **✓** |
| Model-fit / VRAM | basic | rare | rare | **✓** |
| KV-cache by context + concurrency | — | — | implicit | **✓** |
| Prefill vs decode split (TTFT/ITL) | — | — | engine-specific | **✓** |
| Continuous batching / paged attention | — | — | docs only | **✓** |
| Long-context (128K–1M) planning | — | — | — | **✓** |
| Topology recommendation (TP/CP) | — | — | partial | **✓** |
| Prompt-cache economics | — | partial API only | — | **✓** |
| Reasoning tokens (o1/R1/Claude thinking) | — | — | — | **✓** |
| API vs self-host break-even | some | ✓ | — | **✓** |
| Transparent formulas / open source | mixed | usually no | mixed | **✓** |
| Cited benchmark evidence per config | rare | rare | ✓ (not planning) | **✓ (in progress)** |
| Shareable URL per config | — | — | — | **✓** |

## Roadmap

### Shipped
- ✅ 35 models, 30 GPUs, 16 quantization formats
- ✅ Continuous batching, reasoning tokens, prompt caching
- ✅ TTFT/ITL split, long-context superlinear attention
- ✅ Long-context capacity planner + topology recommendation
- ✅ Build-vs-Buy calculator (13 API providers with retired/current status)
- ✅ Reference catalog (5 sub-tables)
- ✅ Share URL + localStorage persistence
- ✅ Dark mode toggle
- ✅ Plain-English glossary (28 terms with hover tooltips)
- ✅ OG image + Twitter card + social metadata

### Next 30 days
- ⏳ GitHub Action (`tokcalc/plan` PR comment)
- ⏳ MCP server (read-only capacity-planning tools for AI agents)
- ⏳ 3 SEO landing pages (`/compare/h100-vs-h200`, `/gguf-q4-k-m-vs-q5-k-m`, `/vllm-vs-sglang`)
- ⏳ i18n: Chinese, Japanese, Korean

### Next 90 days
- ⏳ Workload-trace / SLO capacity planner (prompt/output/arrival distributions, p50/p95 TTFT/ITL)
- ⏳ P/D disaggregation planner (separate prefill + decode pools)
- ⏳ Cache-aware economics (prefix-sharing distribution, multi-turn/agent traces)
- ⏳ Engine-aware presets (vLLM / SGLang / TensorRT-LLM / llama.cpp)
- ⏳ Versioned price + benchmark provenance system

### Long-term
- 🔮 Agentic workflow calculator (multi-turn + tool calls + growing context)
- 🔮 Multi-LoRA capacity planner (Punica / S-LoRA economics)
- 🔮 VLM image-token accounting (per-model patch/tile tokenization)
- 🔮 Embedding model mode (vectors/sec, separate workload)
- 🔮 Training/fine-tuning estimator (LoRA / QLoRA / full-SFT FLOPs)
- 🔮 Energy / carbon per million tokens (region-specific grid intensity)

## Open core model

tokcalc is **open core** — the calculator and catalog are open source; the
cloud/data/team features are paid.

| Asset | License | Notes |
|---|---|---|
| Source code | Apache 2.0 | This repo. Free to use, modify, distribute |
| Model/GPU/quant catalog | CC0 1.0 | Public domain data. Anyone can use, no attribution required |
| Benchmark provenance data | CC-BY-SA 4.0 | Anyone can use, but must attribute + share-alike |
| Documentation | CC-BY 4.0 | Attribution required if copied |
| "tokcalc" name + logo | Trademark | Even without formal registration, common-law rights apply |
| Cloud SaaS layer | Proprietary | Real-time pricing API, benchmark DB, team workspaces (coming soon) |

### Why this structure
- **Trust**: Open-source formulas build credibility vs opaque competitors
- **Community**: Contributors can submit models, GPUs, quants, benchmarks
- **Defensibility**: Trademark + cloud features + URL-share viral loop protect against forks
- **Revenue**: Cloud tier funds ongoing development + pricing/benchmark data maintenance

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- How to add a model (with HuggingFace config.json as source)
- How to add a GPU (with critical guardrails for B200/B300 null FP16 fields)
- How to add a quantization format (with measured file sizes as source)
- How to submit a benchmark (3-tier confidence model)
- How to improve a formula (cite the source, no magic numbers)

### Most-needed contributions
- 🟢 New models (Qwen 3, Mistral Large 4, Llama 4 variants as they release)
- 🟢 New GPUs (B300, AMD MI400, Apple M5 Ultra when shipping)
- 🟢 New quantization formats (mxFP8, MXFP4, BitNet 2)
- 🟢 Real benchmark data (run vLLM benchmarks and submit with provenance)
- 🟢 Translations (especially Chinese, Japanese, Korean)

## Tech stack

- **Framework**: Next.js 16 with App Router
- **Language**: TypeScript 5
- **Styling**: Tailwind CSS 4 + shadcn/ui (New York)
- **Charts**: Recharts
- **State**: React hooks (useState + useEffect + useMemo)
- **Theme**: next-themes (dark mode default)
- **Database**: None (pure client-side, no backend required for core features)

## Acknowledgments

tokcalc builds on the work of:

- **vLLM team** — PagedAttention, continuous batching ([arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180))
- **llama.cpp / ggml-org** — GGUF format and quantization variants
- **MLPerf / MLCommons** — standardized inference benchmark methodology
- **Stanford HELM** — efficiency-aware model evaluation framework
- **Anthropic / OpenAI / Google** — published prompt-caching pricing rules
- **NVIDIA / AMD / Intel / Google / Groq / Cerebras** — published hardware specs

Every formula has a citation. Every model/GPU/quant entry has a source URL
where available. If you spot an unsourced claim, please open an issue.

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=tokcalc/tokcalc&type=Date)](https://star-history.com/#tokcalc/tokcalc&Date)

---

<div align="center">

**[Live demo](https://tokcalc.dev)** ·
**[Documentation](https://tokcalc.dev/docs)** ·
**[Contributing](CONTRIBUTING.md)** ·
**[License](LICENSE)** ·
**[Code of Conduct](CODE_OF_CONDUCT.md)**

Made with care by the tokcalc community. Apache 2.0 licensed.

</div>
