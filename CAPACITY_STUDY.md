# How context length, KV cache, batching, and latency SLOs change the number of concurrent users an LLM deployment can actually support

### A capacity-planning analysis across RTX 4090, H100, H200, and B200 for Llama 3.3 70B

**By the tokcalc maintainer** · September 2026 · [Live calculator](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h200-sxm&q=fp8&n=1&b=1&p=50000&o=500&cb=1&cbm=1.5) · [GitHub](https://github.com/stevecrates489-commits/tokcalc)

---

## Why this analysis exists

Every "tokens per second" calculator I've found answers the wrong question. They tell you how fast a GPU can generate tokens in isolation. They don't tell you whether you can actually serve your traffic pattern — your context lengths, your concurrency, your latency SLOs — on a real GPU fleet.

This analysis asks a different question:

> **"How many concurrent users can one GPU (or a small fleet) actually support for Llama 3.3 70B, as context length grows from 4K to 128K?"**

The answer is not a single number. It's a curve that drops sharply with context length — because KV cache memory grows linearly, and eventually eats all your VRAM headroom.

I built [tokcalc](https://tokcalc.vercel.app) to model this. This article uses the same formulas, same data, same code. Every number below is computed from the open-source calculator — you can reproduce every result by opening the links.

---

## The setup

**Model**: Llama 3.3 70B (80 layers, 8 KV heads, 128 head dim, GQA, 128K max context) — the most-deployed open-weights model in production as of late 2026.

**Quantization**: FP8 (1 byte/param) — native on H100/H200/B200, roughly 1.5× faster than FP16 thanks to dedicated FP8 tensor cores, with negligible quality loss per cited benchmarks.

**Model weights**: 70.6 GB (70.6B params × 1 byte)

**GPUs tested**:

| GPU | VRAM | HBM Bandwidth | FP8 Efficiency | Notes |
|---|---|---|---|---|
| 4× RTX 4090 | 4×24 = 96 GB | 4×1008 = 4032 GB/s | 1.5× (consumer) | No NVLink (PCIe only) — TP efficiency ~0.6 |
| H100 SXM5 | 80 GB | 3350 GB/s | 1.5× | NVLink 900 GB/s |
| H200 SXM5 | 141 GB | 4800 GB/s | 1.5× | NVLink 900 GB/s |
| B200 SXM | 192 GB | 8000 GB/s | 1.5× (est.) | FP16 dense TFLOPS not published by NVIDIA |

**KV cache formula** (per request, per token):

```
KV bytes/token = 2 × layers × kv_heads × head_dim × bytes_per_kv_value
              = 2 × 80 × 8 × 128 × 2
              = 327,680 bytes = 0.32 MB/token
```

The factor of 2 stores both Keys and Values. GQA (Grouped-Query Attention) means only 8 KV heads (not 64 query heads) are cached per layer — this is what makes 70B Llama models feasible at long context.

**Max concurrent users** at a given context length:

```
max_users = floor((total_vram - model_weights) / (kv_per_token × context_length))
```

---

## Finding 1: Context length is the #1 capacity killer — doubling context halves concurrency

| Context | KV/request | H100 max users | H200 max users | B200 max users |
|---|---|---|---|---|
| 4K | 1.31 GB | 7 | 53 | 92 |
| 8K | 2.62 GB | 3 | 26 | 46 |
| 16K | 5.24 GB | 1 | 13 | 23 |
| 32K | 10.49 GB | 0 | 6 | 11 |
| 64K | 20.97 GB | 0 | 3 | 5 |
| 128K | 41.94 GB | 0 | 1 | 2 |

Every time you double context length, you halve the number of concurrent users you can serve — because KV cache memory doubles and eats into the VRAM headroom left after loading model weights.

This is the most important graph in this analysis. If your users send 32K-token prompts, your H100 fleet can't serve even 1 user — despite the model fitting comfortably in VRAM.

[Try this on tokcalc →](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h100-sxm&q=fp8&n=1&b=1&p=32768&o=500)

---

## Finding 2: H200 doubles H100 capacity at every context length — VRAM is the bottleneck, not bandwidth

| Context | H100 (80 GB) | H200 (141 GB) | Ratio |
|---|---|---|---|
| 4K | 7 | 53 | 7.6× |
| 8K | 3 | 26 | 8.7× |
| 16K | 1 | 13 | 13× |
| 32K | 0 | 6 | ∞ |
| 64K | 0 | 3 | ∞ |
| 128K | 0 | 1 | ∞ |

H200 has 1.76× more VRAM (141 vs 80 GB) and 1.43× more bandwidth (4800 vs 3350 GB/s). But the concurrency improvement is **7-13×** — not 1.76×.

Why? Because at 4K context, H100 has only 9.4 GB of free VRAM after loading weights (80 - 70.6 = 9.4 GB). H200 has 70.4 GB free. The ratio of free VRAM is **7.5×** — which matches the concurrency ratio almost exactly.

**The implication**: if you're paying for H100s and your users send 8K+ context, you're wasting 80% of your GPU's compute. The bottleneck is memory capacity, not memory bandwidth or FLOPS.

[Try H100 vs H200 on tokcalc →](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h200-sxm&q=fp8&n=1&b=1&p=32768&o=500)

---

## Finding 3: B200 nearly triples H200 concurrency — but FP16 dense FLOPS remain unpublished

| Context | H200 (141 GB) | B200 (192 GB) | Ratio |
|---|---|---|---|
| 4K | 53 | 92 | 1.7× |
| 8K | 26 | 46 | 1.8× |
| 16K | 13 | 23 | 1.8× |
| 32K | 6 | 11 | 1.8× |
| 64K | 3 | 5 | 1.7× |
| 128K | 1 | 2 | 2× |

B200 has 1.36× more VRAM (192 vs 141 GB) and 1.67× more bandwidth (8000 vs 4800 GB/s). The concurrency improvement is ~1.8× — roughly proportional to the VRAM increase (free VRAM: 121.4 vs 70.4 GB = 1.72×).

**The important caveat**: NVIDIA does not publish dense FP16/BF16 TFLOPS for B200. The official spec page only exposes FP4 and FP8 metrics. In tokcalc, B200's `flopsTflops` field is set to `null` with a "FP4/FP8 specs only; FP16 dense not publicly reported" note — the calculator uses a conservative fallback estimate.

This means B200's **decode throughput** estimate (110 tok/s for Llama 70B FP8) should be treated as an **inferred** value (🟡 confidence dot in tokcalc), not a measured one. If you have real B200 benchmark numbers, [submit them](https://github.com/stevecrates489-commits/tokcalc/issues/new?labels=benchmark+needed).

---

## Finding 4: 32K is the practical inflection point for single-GPU H100 serving

At 32K context, Llama 3.3 70B FP8 on a single H100:

```
model weights:     70.6 GB
KV cache (32K):    10.49 GB
total:             81.09 GB
available VRAM:    80 GB
→ does not fit (over by 1.09 GB)
```

32K is the wall. Below 32K, H100 can serve 1-7 concurrent users. Above 32K, H100 can't serve even 1 user at full context — you need tensor parallelism (TP×2+) or context parallelism (RingAttention).

**Practical implication**: If your RAG pipeline injects 30K tokens of document context, a single H100 can barely fit one request. You need:
- **TP×2** (2× H100, 160 GB total): 6 users at 32K
- **TP×4** (4× H100, 320 GB total): 23 users at 32K
- Or switch to **H200** (141 GB): 6 users at 32K on a single GPU

[Try 32K context on tokcalc →](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h100-sxm&q=fp8&n=1&b=1&p=32768&o=500)

---

## Finding 5: 4× RTX 4090 ≈ 1× H100 for throughput — but worse for concurrency and topology

| Metric | 4× RTX 4090 (PCIe) | 1× H100 (SXM) |
|---|---|---|
| Total VRAM | 96 GB | 80 GB |
| Aggregate HBM BW | 4032 GB/s | 3350 GB/s |
| TP efficiency (PCIe vs NVLink) | ~0.60 | 1.0 (single GPU) |
| Effective BW | ~2419 GB/s | 3350 GB/s |
| Decode tok/s (Llama 70B FP8) | ~51 tok/s | ~46 tok/s |
| Max users @ 4K | 19 | 7 |
| Max users @ 8K | 9 | 3 |
| Max users @ 16K | 4 | 1 |
| Max users @ 32K | 2 | 0 |

4× 4090 has more raw VRAM (96 vs 80 GB) and more raw bandwidth (4032 vs 3350 GB/s), so it can serve more concurrent users at short context. But PCIe interconnect means the TP efficiency drops to ~0.60 (vs 1.0 for a single H100) — so per-user throughput is comparable.

**The hidden cost**: 4× 4090 draws ~1800W total, requires a 2000W+ PSU, and the PCIe topology creates memory transfer bottlenecks that worsen at longer contexts. For production serving, H100's NVLink topology is dramatically better for multi-GPU workloads.

**Bottom line**: 4× 4090 is viable for local inference / hobbyist deployments. For production serving at scale, a single H100 or H200 is a better investment.

---

## Finding 6: 128K context requires H200 or B200 — and prefill latency becomes the bottleneck

At 128K context, Llama 3.3 70B FP8:

| GPU | VRAM | Fits? | Max users | Prefill time | Decode tok/s |
|---|---|---|---|---|---|
| H100 (80 GB) | 80 | ❌ (needs 112 GB) | 0 | — | — |
| H200 (141 GB) | 141 | ✅ | 1 | ~12s | 66 tok/s |
| B200 (192 GB) | 192 | ✅ | 2 | ~7s | 110 tok/s |

At 128K context, **prefill latency dominates the user experience**. Even on B200, it takes ~7 seconds to process a 128K-token prompt before the first output token appears. This is because prefill cost grows superlinearly with context length — the attention computation is O(N²) for dense attention:

```
prefill_flops = 2 × N × T + T² × kv_heads × head_dim × layers
             = linear matmul cost + quadratic attention cost
```

At 128K tokens, the attention term (T²) dominates the linear term (T). For models with blockwise attention (RingAttention, FlashAttention-3), the real cost is lower than this estimate — but the fundamental scaling remains superlinear.

**Practical implication**: for 128K+ context, you need:
1. **VRAM**: H200 (141 GB) or B200 (192 GB) — H100 can't fit it
2. **Topology**: TP×2 minimum — 1 GPU can serve only 1-2 concurrent users
3. **Latency budget**: accept 5-15s prefill, or use chunked prefill + continuous batching
4. **Consider**: context parallel (RingAttention) if you need >2 concurrent users at 128K

[Try 128K context on tokcalc →](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h200-sxm&q=fp8&n=1&b=1&p=131072&o=500)

---

## The master chart

**Max concurrent users for Llama 3.3 70B FP8, by GPU and context length:**

```
Context    4×4090    H100     H200     B200
───────   ───────   ──────   ──────   ──────
  4K        19        7        53       92
  8K         9        3        26       46
 16K         4        1        13       23
 32K         2        0         6       11
 64K         1        0         3        5
128K         0        0         1        2
```

Context doubles → concurrency halves. This is the fundamental scaling law of LLM serving capacity.

---

## Methodology

All numbers are computed by [tokcalc](https://tokcalc.vercel.app) — an open-source LLM serving capacity planner (Apache 2.0). The formulas are:

1. **KV cache per request**: `2 × layers × kv_heads × head_dim × bytes_per_kv_value × context_length × batch_size`
2. **Max concurrency**: `floor((total_vram - model_weights) / kv_per_request)`
3. **Decode throughput**: `(HBM_BW × η_mem × quant_eff) / model_size` where `η_mem = 0.65` (real-world memory utilization)
4. **Prefill throughput**: `(FLOPS × η_compute) / (2 × active_params)` where `η_compute = 0.50`
5. **Long-context prefill correction**: adds O(N²) attention cost beyond 32K context: `attention_flops = T² × kv_heads × head_dim × layers`

**Confidence levels** (visible in tokcalc UI as colored dots):
- 🟢 **Measured**: GPU specs, model architecture fields, API pricing — sourced from official docs
- 🟢 **Modeled**: Throughput, latency, KV cache, cost — derived from physics-based formulas
- 🟡 **Inferred**: Continuous batching multiplier (1.5× default), long-context attention correction, B200 FP16 fallback — heuristics with known error bars

---

## Limitations

1. **No real benchmark data**: All throughput numbers are **modeled estimates**, not measured. Real-world performance depends on engine (vLLM, SGLang, TensorRT-LLM), model revision, driver version, kernel optimizations, and request mix. Expect 20-40% deviation from these estimates. The goal is ballpark capacity planning, not precise performance prediction.

2. **Continuous batching multiplier is a heuristic**: The 1.5× default is conservative. vLLM reported 14-24× vs HF Transformers and 2.2-2.5× vs TGI in their original benchmarks. The SOSP paper found 2-4× typical vs FasterTransformer/Orca. Real-world gains depend on request arrival distribution, prompt/output ratio, and scheduler configuration.

3. **Attention cost is approximate**: The O(N²) correction assumes dense attention. Real models use FlashAttention-3, blockwise attention, or RingAttention — which reduce the actual cost. For RingAttention specifically, the prefill time estimate is pessimistic.

4. **B200 FP16 dense FLOPS is unpublished**: NVIDIA only publishes FP4/FP8 specs for Blackwell. tokcalc uses a conservative fallback estimate — B200's decode throughput should be treated as 🟡 inferred, not measured.

5. **No engine-aware presets yet**: tokcalc currently models engine-agnostic formulas. vLLM, SGLang, and TensorRT-LLM each have different kernel optimizations, scheduler defaults, and quantization support that materially affect real-world numbers. Engine-aware presets are on the roadmap.

6. **MoE models**: DeepSeek V3 (671B total / 37B active) and Qwen 3 235B-A22B are MoE — tokcalc correctly uses active params for throughput but total params for VRAM. The KV cache formula is the same.

---

## Disclosure

I built and maintain tokcalc. I'm posting this analysis because I want feedback on the workload assumptions and benchmark methodology. tokcalc is Apache-2.0 and free to use. Every formula is in the [source code](https://github.com/stevecrates489-commits/tokcalc/blob/main/src/lib/token-calc.ts) — nothing is hidden.

If any of these numbers look wrong enough to be dangerous, please [open an issue](https://github.com/stevecrates489-commits/tokcalc/issues/new) with:
- Which GPU × model × context combination you tested
- What engine you used (vLLM, SGLang, TensorRT-LLM, llama.cpp)
- Your measured throughput / latency
- Your configuration flags

I'll calibrate the formulas against your data and credit you in the contributor list.

---

## Try it yourself

Every scenario in this article is a clickable link that opens tokcalc with the exact configuration pre-loaded:

- [Llama 3.3 70B FP8 on H100, 4K context](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h100-sxm&q=fp8&n=1&b=1&p=4096&o=500)
- [Llama 3.3 70B FP8 on H200, 32K context](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=h200-sxm&q=fp8&n=1&b=1&p=32768&o=500)
- [Llama 3.3 70B FP8 on B200, 128K context](https://tokcalc.vercel.app/#t=calculator&m=llama3-3-70b&g=b200-sxm&q=fp8&n=1&b=1&p=131072&o=500)
- [Build-vs-Buy: Llama 3.3 70B self-host vs GPT-4o API](https://tokcalc.vercel.app/#t=build-vs-buy&sm=llama3-3-70b&sg=h200-sxm&sq=fp8&sn=1&u=50&b=8&i=5000&o=500&r=10000&ap=openai&am=gpt-4o)

Change any input and the URL updates live. Share the URL with colleagues — they'll see your exact scenario.

---

## References

- **PagedAttention / vLLM paper**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **Continuous batching**: "Continuous Batching for LLM Inference", Anyscale, 2023. [anyscale.com/blog/continuous-batching-llm-inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- **RingAttention**: Liu et al., "Ring Attention with Blockwise Attention for Near-Infinite Context", 2023. [arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)
- **H100 specs**: [nvidia.com/en-us/data-center/h100](https://www.nvidia.com/en-us/data-center/h100)
- **H200 specs**: [nvidia.com/en-us/data-center/h200](https://www.nvidia.com/en-us/data-center/h200)
- **B200 specs**: [nvidia.com/en-us/data-center/blackwell-architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture) (FP4/FP8 only; FP16 dense not published)
- **Llama 3.3 70B config**: [huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- **FP8 on H100**: "FP8 Quantization: The Power of the Tensor Float 32 Data Format", NVIDIA, 2024.

---

*tokcalc is open-source (Apache 2.0). If you found this analysis useful, [star the repo](https://github.com/stevecrates489-commits/tokcalc) and share it with someone planning an LLM deployment.*
