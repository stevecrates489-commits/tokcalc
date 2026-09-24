/**
 * LLM Token/sec Calculator — Core Engine
 * ---------------------------------------
 * All formulas live here so they can be reused & tested independently.
 *
 * Theory reference:
 *   - Decode phase is MEMORY-BANDWIDTH bound (load weights once per token)
 *   - Prefill phase is COMPUTE bound (process prompt in parallel)
 *   - Real-world efficiency factor (~0.65) accounts for kernel overhead, KV cache reads, attention, sampling
 */

export type Quantization =
  | "fp32" | "fp16" | "bf16" | "int8" | "int4"
  | "gguf-q2k" | "gguf-q3km" | "gguf-q4km" | "gguf-q5km" | "gguf-q6k" | "gguf-q8"
  | "gptq4" | "awq4" | "exl2-6bpw" | "fp8" | "nvfp4";

export interface QuantMeta {
  id: Quantization;
  label: string;
  /** Effective bytes per parameter (averaged across tensor mix for GGUF) */
  bytesPerParam: number;
  /** Dequantization overhead multiplier (1.0 = none, 0.85 = 15% overhead)
   *  For FP8 on H100: efficiency >1.0 (faster than FP16 due to native FP8 kernels) */
  efficiency: number;
  description: string;
  /** Format family — for engine-matching UI */
  family: "float" | "gguf" | "gptq" | "awq" | "exl2" | "fp8" | "nvfp4";
  /** Typical use case */
  useCase?: string;
}

// GGUF variants: based on llama.cpp quantize README for Llama 3 8B file sizes.
// Refs: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
export const QUANTIZATIONS: QuantMeta[] = [
  // === Float / native precision ===
  { id: "fp32",  label: "FP32 (raw)",   bytesPerParam: 4.0,  efficiency: 1.00, family: "float", description: "32-bit float. Highest accuracy, 2x size of FP16." },
  { id: "fp16",  label: "FP16",         bytesPerParam: 2.0,  efficiency: 1.00, family: "float", description: "16-bit float. Standard for inference, no dequant overhead." },
  { id: "bf16",  label: "BF16",         bytesPerParam: 2.0,  efficiency: 1.00, family: "float", description: "Brain Float 16. Same size as FP16, better numerical range." },
  // === Generic integer quantization (legacy bucket for vLLM/TGI default) ===
  { id: "int8",  label: "INT8",         bytesPerParam: 1.0,  efficiency: 0.88, family: "float", description: "8-bit integer. Halves memory, ~12% dequant overhead." },
  { id: "int4",  label: "INT4 (generic)", bytesPerParam: 0.5, efficiency: 0.80, family: "float", description: "4-bit integer (generic). For specific format, pick GPTQ-4 / AWQ-4 / GGUF Q4_K_M." },

  // === GGUF variants (llama.cpp) ===
  { id: "gguf-q2k",  label: "GGUF Q2_K",   bytesPerParam: 0.37, efficiency: 0.92, family: "gguf", description: "~2.6 bpw effective. Smallest; significant accuracy loss, use only as last resort.", useCase: "Extreme low-memory edge devices" },
  { id: "gguf-q3km",  label: "GGUF Q3_K_M", bytesPerParam: 0.42, efficiency: 0.93, family: "gguf", description: "~3.9 bpw. Low quality; acceptable for some chat uses but not coding/math.", useCase: "Low-memory local inference" },
  { id: "gguf-q4km",  label: "GGUF Q4_K_M", bytesPerParam: 0.55, efficiency: 0.90, family: "gguf", description: "~4.8 bpw. Recommended sweet spot for local Llama/Mistral on consumer GPUs.", useCase: "Recommended for local inference" },
  { id: "gguf-q5km",  label: "GGUF Q5_K_M", bytesPerParam: 0.68, efficiency: 0.92, family: "gguf", description: "~5.7 bpw. Higher quality than Q4, ~30% larger. Good when memory allows.", useCase: "Quality-focused local inference" },
  { id: "gguf-q6k",   label: "GGUF Q6_K",   bytesPerParam: 0.82, efficiency: 0.95, family: "gguf", description: "~6.6 bpw. Near-FP16 quality, ~60% larger than Q4_K_M.", useCase: "Near-lossless local inference" },
  { id: "gguf-q8",    label: "GGUF Q8_0",   bytesPerParam: 1.07, efficiency: 0.98, family: "gguf", description: "~8.5 bpw. Effectively lossless, ~half the size of FP16.", useCase: "Reference / lossless storage" },

  // === GPTQ (post-training weight quantization) ===
  { id: "gptq4",  label: "GPTQ 4-bit",   bytesPerParam: 0.55, efficiency: 0.85, family: "gptq", description: "Post-training 4-bit weight quant. Supported by vLLM, TGI, TensorRT-LLM (varies by version).", useCase: "vLLM/TGI serverless" },
  // === AWQ (activation-aware weight quantization) ===
  { id: "awq4",   label: "AWQ 4-bit",    bytesPerParam: 0.55, efficiency: 0.88, family: "awq", description: "Activation-aware W4A16. Preserves salient weights; better accuracy than GPTQ in some setups.", useCase: "vLLM/TensorRT-LLM serving" },
  // === EXL2 (ExLlamaV2, mixed-bit) ===
  { id: "exl2-6bpw", label: "EXL2 6.0 bpw", bytesPerParam: 0.75, efficiency: 0.92, family: "exl2", description: "Mixed-bit quantization for ExLlamaV2. Common 6.0 bpw preset balances speed/quality on consumer NVIDIA.", useCase: "Consumer NVIDIA + ExLlamaV2" },
  // === FP8 (H100/H200 native) ===
  { id: "fp8",    label: "FP8 (E4M3)",   bytesPerParam: 1.0,  efficiency: 1.5,  family: "fp8", description: "8-bit floating-point. Native on H100/H200 — FASTER than FP16 (efficiency >1). Same VRAM as INT8 but better quality.", useCase: "H100/H200 with TRT-LLM/vLLM" },
  // === NVFP4 (Blackwell only) ===
  { id: "nvfp4",  label: "NVFP4 (Blackwell)", bytesPerParam: 0.5, efficiency: 1.2, family: "nvfp4", description: "4-bit floating-point with block scaling. Native on Blackwell (B200/B300). Roughly 2x faster than FP8 on B200 in cited tests.", useCase: "Blackwell B200/B300 only" },
];

export const QUANT_MAP: Record<Quantization, QuantMeta> =
  Object.fromEntries(QUANTIZATIONS.map((q) => [q.id, q])) as Record<Quantization, QuantMeta>;

export type GpuCategory = "datacenter" | "workstation" | "consumer" | "mac" | "tpu" | "lpu" | "wse" | "rdu" | "legacy";

export interface GpuSpec {
  id: string;
  name: string;
  vendor: string;
  category: GpuCategory;
  /** HBM bandwidth in GB/s */
  memBandwidthGbps: number;
  /** FP16/BF16 compute in TFLOPS (dense, not sparse). null = vendor does not publish dense FP16 (e.g. B200 exposes FP4/FP8 only) */
  flopsTflops: number | null;
  /** VRAM in GB */
  vramGb: number;
  /** NVLink bandwidth per GPU in GB/s (0 if no NVLink) */
  nvlinkGbps: number;
  /** Typical on-demand $/hr in USD (cloud, rough 2025-2026 estimate). null = quote required */
  usdPerHour: number | null;
  /** Release year */
  year: number;
  /** Special note (e.g. 'FP16 dense not publicly reported', 'rack-scale system', etc.) */
  note?: string;
  /** HuggingFace URL or vendor source for verification */
  sourceUrl?: string;
}

export const GPUS: GpuSpec[] = [
  // === NVIDIA Datacenter ===
  // Refs: https://www.nvidia.com/en-us/data-center/h100 · https://www.nvidia.com/en-us/data-center/h200 · https://www.nvidia.com/en-us/data-center/b200
  { id: "h100-sxm",  name: "H100 SXM5 80GB",    vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 3350, flopsTflops: 990,  vramGb: 80,  nvlinkGbps: 900, usdPerHour: 2.50, year: 2022, sourceUrl: "https://www.nvidia.com/en-us/data-center/h100" },
  { id: "h100-pcie", name: "H100 PCIe 80GB",    vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 2000, flopsTflops: 756,  vramGb: 80,  nvlinkGbps: 0,   usdPerHour: 2.00, year: 2023, sourceUrl: "https://www.nvidia.com/en-us/data-center/h100" },
  { id: "h200-sxm",  name: "H200 SXM5 141GB",   vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 4800, flopsTflops: 990,  vramGb: 141, nvlinkGbps: 900, usdPerHour: 4.00, year: 2024, sourceUrl: "https://www.nvidia.com/en-us/data-center/h200" },
  // B200: NVIDIA publishes FP4/FP8 but not dense FP16 — set to null per research guidance.
  { id: "b200-sxm",  name: "B200 SXM 192GB",    vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 8000, flopsTflops: null, vramGb: 192, nvlinkGbps: 1800,usdPerHour: 6.00, year: 2025, note: "FP4/FP8 specs only; FP16 dense not publicly reported", sourceUrl: "https://www.nvidia.com/en-us/data-center/blackwell-architecture" },
  // B300 (2025) — announced/limited availability. Specs subject to change.
  { id: "b300-sxm",  name: "B300 SXM 288GB",    vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 8000, flopsTflops: null, vramGb: 288, nvlinkGbps: 1800,usdPerHour: 8.00, year: 2025, note: "Announced; FP16 dense not published", sourceUrl: "https://www.nvidia.com/en-us/data-center/dgx-b300" },
  { id: "a100-80",  name: "A100 80GB SXM4",     vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 2000, flopsTflops: 312,  vramGb: 80,  nvlinkGbps: 600, usdPerHour: 1.20, year: 2020, sourceUrl: "https://www.nvidia.com/en-us/data-center/a100" },
  { id: "a100-40",  name: "A100 40GB SXM4",     vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 1550, flopsTflops: 312,  vramGb: 40,  nvlinkGbps: 600, usdPerHour: 0.90, year: 2020, sourceUrl: "https://www.nvidia.com/en-us/data-center/a100" },
  { id: "l40s",     name: "L40S 48GB",          vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 864,  flopsTflops: 91.6, vramGb: 48,  nvlinkGbps: 0,   usdPerHour: 0.70, year: 2023, sourceUrl: "https://www.nvidia.com/en-us/data-center/l40s" },
  { id: "a10g",     name: "A10G 24GB",          vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 600,  flopsTflops: 31,   vramGb: 24,  nvlinkGbps: 0,   usdPerHour: 0.36, year: 2021 },
  { id: "t4",       name: "T4 16GB",            vendor: "NVIDIA", category: "datacenter", memBandwidthGbps: 320,  flopsTflops: 8.1,  vramGb: 16,  nvlinkGbps: 0,   usdPerHour: 0.20, year: 2018 },
  // V100 — legacy but still in production for some workloads
  { id: "v100",     name: "V100 SXM2 32GB",     vendor: "NVIDIA", category: "legacy",    memBandwidthGbps: 900,  flopsTflops: 125,  vramGb: 32,  nvlinkGbps: 300, usdPerHour: 0.50, year: 2017, note: "Legacy architecture (Volta)" },

  // === NVIDIA Workstation (Blackwell + Ada) ===
  { id: "l4",       name: "L4 24GB",            vendor: "NVIDIA", category: "workstation", memBandwidthGbps: 300,  flopsTflops: 30.3, vramGb: 24,  nvlinkGbps: 0,   usdPerHour: 0.40, year: 2023 },
  { id: "rtx-6000a",name: "RTX 6000 Ada 48GB", vendor: "NVIDIA", category: "workstation", memBandwidthGbps: 960,  flopsTflops: 91.6, vramGb: 48,  nvlinkGbps: 0,   usdPerHour: 0.80, year: 2022 },
  // RTX PRO 6000 Blackwell — 96GB VRAM, 2025 release
  { id: "rtx-pro-6000-bw", name: "RTX PRO 6000 Blackwell 96GB", vendor: "NVIDIA", category: "workstation", memBandwidthGbps: 1792, flopsTflops: 125, vramGb: 96, nvlinkGbps: 0, usdPerHour: 1.50, year: 2025, sourceUrl: "https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family" },
  { id: "a4000",    name: "RTX A4000 16GB",     vendor: "NVIDIA", category: "workstation", memBandwidthGbps: 448,  flopsTflops: 19.5, vramGb: 16,  nvlinkGbps: 0,   usdPerHour: 0.30, year: 2021 },

  // === NVIDIA Consumer (Blackwell + Ada + Ampere) ===
  { id: "rtx-5090", name: "RTX 5090 32GB",      vendor: "NVIDIA", category: "consumer", memBandwidthGbps: 1792, flopsTflops: 105,  vramGb: 32,  nvlinkGbps: 0,   usdPerHour: 0.50, year: 2025, sourceUrl: "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series" },
  { id: "rtx-4090", name: "RTX 4090 24GB",      vendor: "NVIDIA", category: "consumer", memBandwidthGbps: 1008, flopsTflops: 82.6, vramGb: 24,  nvlinkGbps: 0,   usdPerHour: 0.40, year: 2022 },
  { id: "rtx-3090", name: "RTX 3090 24GB",      vendor: "NVIDIA", category: "consumer", memBandwidthGbps: 936,  flopsTflops: 35.6, vramGb: 24,  nvlinkGbps: 0,   usdPerHour: 0.25, year: 2020 },
  { id: "rtx-4080", name: "RTX 4080 16GB",      vendor: "NVIDIA", category: "consumer", memBandwidthGbps: 717,  flopsTflops: 48.7, vramGb: 16,  nvlinkGbps: 0,   usdPerHour: 0.30, year: 2022 },
  { id: "rtx-3080", name: "RTX 3080 10GB",      vendor: "NVIDIA", category: "consumer", memBandwidthGbps: 760,  flopsTflops: 34.1, vramGb: 10,  nvlinkGbps: 0,   usdPerHour: 0.20, year: 2020 },

  // === AMD Instinct (ROCm) ===
  // Refs: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
  { id: "mi300x",    name: "AMD MI300X 192GB",     vendor: "AMD", category: "datacenter", memBandwidthGbps: 5300, flopsTflops: 1307, vramGb: 192, nvlinkGbps: 0,   usdPerHour: 2.50, year: 2023, note: "ROCm; performance depends on ROCm version", sourceUrl: "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html" },
  { id: "mi325x",   name: "AMD MI325X 288GB",    vendor: "AMD", category: "datacenter", memBandwidthGbps: 6000, flopsTflops: 1307, vramGb: 288, nvlinkGbps: 0,   usdPerHour: 3.50, year: 2024, sourceUrl: "https://www.amd.com/en/products/accelerators/instinct.html" },

  // === Intel Gaudi ===
  // Refs: https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html
  { id: "gaudi3",    name: "Intel Gaudi 3 128GB",  vendor: "Intel", category: "datacenter", memBandwidthGbps: 3300, flopsTflops: 1835, vramGb: 128, nvlinkGbps: 0,   usdPerHour: 2.00, year: 2024, note: "Ethernet scale-out; not drop-in for CUDA", sourceUrl: "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html" },

  // === Google TPU (separate schema, but in same DB for convenience) ===
  // Refs: https://cloud.google.com/tpu/docs
  { id: "tpu-v5p",    name: "TPU v5p (per-chip, 95GB)",   vendor: "Google", category: "tpu", memBandwidthGbps: 819, flopsTflops: 459, vramGb: 95, nvlinkGbps: 0, usdPerHour: 4.20, year: 2023, note: "Pod-scale; per-chip specs only", sourceUrl: "https://cloud.google.com/tpu/docs" },
  { id: "tpu-trillium", name: "TPU v6e Trillium (32GB)", vendor: "Google", category: "tpu", memBandwidthGbps: 819, flopsTflops: 918, vramGb: 32, nvlinkGbps: 0, usdPerHour: 0.80, year: 2024, sourceUrl: "https://cloud.google.com/tpu/docs" },

  // === Groq LPU (special — designed for LLM inference, no FLOPS-comparable metric) ===
  // Refs: https://groq.com — listed as 'lpu' category; user can use 'continuous batching' multiplier on top
  { id: "groq-lpu",   name: "GroqLPU (per-chip, 230MB SRAM)", vendor: "Groq", category: "lpu", memBandwidthGbps: 9000, flopsTflops: 750, vramGb: 0.23, nvlinkGbps: 0, usdPerHour: null, year: 2024, note: "LPU architecture; SRAM-based, very different from HBM GPUs. Use cited model-specific benchmarks.", sourceUrl: "https://groq.com" },

  // === Cerebras CS-3 (wafer-scale) ===
  // Refs: https://www.cerebras.ai
  { id: "cerebras-cs3", name: "Cerebras CS-3 (WSE-3)",  vendor: "Cerebras", category: "wse", memBandwidthGbps: 20000, flopsTflops: 1250, vramGb: 44, nvlinkGbps: 0, usdPerHour: null, year: 2024, note: "Wafer-scale; not comparable to per-GPU specs. Use cited benchmarks.", sourceUrl: "https://www.cerebras.ai" },

  // === Apple Silicon (unified memory) ===
  // Refs: https://www.apple.com/newsroom/2023/06/apple-unveils-m2-ultra/
  { id: "m2u-800",  name: "Mac M2 Ultra (800GB)", vendor: "Apple", category: "mac", memBandwidthGbps: 800, flopsTflops: 27.0, vramGb: 192, nvlinkGbps: 0, usdPerHour: 0.0, year: 2023, note: "Unified memory; purchased hardware, not rentable" },
  { id: "m2u-192",  name: "Mac M2 Ultra (192GB)",  vendor: "Apple", category: "mac", memBandwidthGbps: 800, flopsTflops: 27.0, vramGb: 192, nvlinkGbps: 0, usdPerHour: 0.0, year: 2023, note: "Unified memory" },
  { id: "m3max-64", name: "Mac M3 Max (64GB)",    vendor: "Apple", category: "mac", memBandwidthGbps: 400, flopsTflops: 14.0, vramGb: 64,  nvlinkGbps: 0, usdPerHour: 0.0, year: 2023, note: "Unified memory" },
  { id: "m4max-128",name: "Mac M4 Max (128GB)",  vendor: "Apple", category: "mac", memBandwidthGbps: 546, flopsTflops: 17.0, vramGb: 128, nvlinkGbps: 0, usdPerHour: 0.0, year: 2024, note: "Unified memory" },
];

export const GPU_MAP: Record<string, GpuSpec> = Object.fromEntries(GPUS.map((g) => [g.id, g]));

export type ModelCategory = "text" | "vlm" | "embedding" | "code" | "reasoning";

export interface ModelSpec {
  id: string;
  name: string;
  family: string;
  /** Workload category — text, vision-language, embedding, code, or reasoning */
  category: ModelCategory;
  /** total params in billions */
  paramsB: number;
  /** active params in billions (for MoE; = paramsB for dense) */
  activeParamsB: number;
  /** number of transformer layers */
  layers: number;
  /** hidden dimension */
  hiddenDim: number;
  /** query heads */
  qHeads: number;
  /** kv heads (for GQA) */
  kvHeads: number;
  /** head dimension */
  headDim: number;
  /** vocab size */
  vocabSize: number;
  /** default max context */
  maxContext: number;
  /** is MoE */
  isMoE: boolean;
}

export const MODELS: ModelSpec[] = [
  // === Llama family ===
  { id: "llama3-8b",   name: "Llama 3 8B",          family: "Llama",   category: "text",      paramsB: 8.03, activeParamsB: 8.03,  layers: 32, hiddenDim: 4096,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 128256, maxContext: 8192,   isMoE: false },
  { id: "llama3-70b",  name: "Llama 3 70B",         family: "Llama",   category: "text",      paramsB: 70.6, activeParamsB: 70.6,  layers: 80, hiddenDim: 8192,  qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 128256, maxContext: 8192,   isMoE: false },
  { id: "llama3-405b", name: "Llama 3.1 405B",      family: "Llama",   category: "text",      paramsB: 405,  activeParamsB: 405,   layers: 126,hiddenDim: 16384, qHeads: 128,kvHeads: 8,  headDim: 128, vocabSize: 128256, maxContext: 131072, isMoE: false },
  { id: "llama3-3-70b",name: "Llama 3.3 70B",      family: "Llama",   category: "text",      paramsB: 70.6, activeParamsB: 70.6,  layers: 80, hiddenDim: 8192,  qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 128256, maxContext: 131072, isMoE: false },
  { id: "llama2-7b",   name: "Llama 2 7B",          family: "Llama",   category: "text",      paramsB: 6.74, activeParamsB: 6.74, layers: 32, hiddenDim: 4096,  qHeads: 32, kvHeads: 32, headDim: 128, vocabSize: 32000, maxContext: 4096,  isMoE: false },
  // Llama 4 — open-weight multimodal MoE (2025-04). Exact architecture fields not fully published
  // for all variants; we use Meta's release figures for total/active params per official blog.
  // Refs: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
  { id: "llama4-scout", name: "Llama 4 Scout 17B-16E (MoE, VLM)",    family: "Llama", category: "vlm",       paramsB: 109,  activeParamsB: 17,   layers: 48, hiddenDim: 5120,  qHeads: 40, kvHeads: 8,  headDim: 128, vocabSize: 202000, maxContext: 10000000, isMoE: true  },
  { id: "llama4-maverick", name: "Llama 4 Maverick 17B-128E (MoE, VLM)", family: "Llama", category: "vlm",   paramsB: 400,  activeParamsB: 17,   layers: 48, hiddenDim: 5120,  qHeads: 40, kvHeads: 8,  headDim: 128, vocabSize: 202000, maxContext: 1000000, isMoE: true  },

  // === Mistral family ===
  { id: "mistral-7b", name: "Mistral 7B v0.3",      family: "Mistral", category: "text",      paramsB: 7.24, activeParamsB: 7.24, layers: 32, hiddenDim: 4096,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 32768, maxContext: 32768,  isMoE: false },
  { id: "mixtral-8x7b",name: "Mixtral 8x7B (MoE)",  family: "Mistral", category: "text",      paramsB: 46.7, activeParamsB: 12.9, layers: 32, hiddenDim: 4096,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 32000, maxContext: 32768,  isMoE: true  },
  { id: "mixtral-8x22b",name:"Mixtral 8x22B (MoE)", family: "Mistral", category: "text",      paramsB: 141,  activeParamsB: 39,   layers: 56, hiddenDim: 6144,  qHeads: 48, kvHeads: 8,  headDim: 128, vocabSize: 32000, maxContext: 65536,  isMoE: true  },
  // Mistral Large 3 (2025-12) — open-weight MoE, 256K context. Refs: https://docs.mistral.ai/models/mistral-large-3-25-12
  { id: "mistral-large-3", name: "Mistral Large 3 (256K)", family: "Mistral", category: "text", paramsB: 235, activeParamsB: 71,  layers: 78, hiddenDim: 7168,  qHeads: 56, kvHeads: 8,  headDim: 128, vocabSize: 131072, maxContext: 262144, isMoE: true  },
  { id: "pixtral-12b", name: "Pixtral 12B (VLM)",   family: "Mistral", category: "vlm",       paramsB: 12,   activeParamsB: 12,   layers: 40, hiddenDim: 5120,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 131072, maxContext: 131072, isMoE: false },
  { id: "codestral-25", name: "Codestral 25.08",    family: "Mistral", category: "code",      paramsB: 22,   activeParamsB: 22,   layers: 56, hiddenDim: 6144,  qHeads: 48, kvHeads: 8,  headDim: 128, vocabSize: 32768, maxContext: 32768,  isMoE: false },

  // === Qwen family ===
  { id: "qwen2-7b",   name: "Qwen2 7B",            family: "Qwen",    category: "text",      paramsB: 7.62, activeParamsB: 7.62, layers: 28, hiddenDim: 3584,  qHeads: 28, kvHeads: 4,  headDim: 128, vocabSize: 152064, maxContext: 32768,  isMoE: false },
  { id: "qwen2-72b",  name: "Qwen2 72B",           family: "Qwen",    category: "text",      paramsB: 72.7, activeParamsB: 72.7, layers: 80, hiddenDim: 8192,  qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 152064, maxContext: 32768,  isMoE: false },
  { id: "qwen2-5-14b",name: "Qwen 2.5 14B",         family: "Qwen",    category: "text",      paramsB: 14.0, activeParamsB: 14.0, layers: 48, hiddenDim: 5120,  qHeads: 40, kvHeads: 8,  headDim: 128, vocabSize: 152064, maxContext: 131072, isMoE: false },
  { id: "qwen2-5-72b",name: "Qwen 2.5 72B",        family: "Qwen",    category: "text",      paramsB: 72.7, activeParamsB: 72.7, layers: 80, hiddenDim: 8192,  qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 152064, maxContext: 131072, isMoE: false },
  // Qwen 3 family (2025-04). Refs: https://huggingface.co/Qwen
  { id: "qwen3-4b",   name: "Qwen 3 4B",            family: "Qwen",    category: "text",      paramsB: 4.0,  activeParamsB: 4.0,  layers: 36, hiddenDim: 2560,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 151936, maxContext: 40960,  isMoE: false },
  { id: "qwen3-8b",   name: "Qwen 3 8B",            family: "Qwen",    category: "text",      paramsB: 8.0,  activeParamsB: 8.0,  layers: 36, hiddenDim: 4096,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 151936, maxContext: 40960,  isMoE: false },
  { id: "qwen3-14b",  name: "Qwen 3 14B",           family: "Qwen",    category: "text",      paramsB: 14.0, activeParamsB: 14.0, layers: 40, hiddenDim: 5120,  qHeads: 40, kvHeads: 8,  headDim: 128, vocabSize: 151936, maxContext: 40960,  isMoE: false },
  { id: "qwen3-32b",  name: "Qwen 3 32B",           family: "Qwen",    category: "text",      paramsB: 32.0, activeParamsB: 32.0, layers: 64, hiddenDim: 5120,  qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 151936, maxContext: 40960,  isMoE: false },
  { id: "qwen3-30b-a3b", name: "Qwen 3 30B-A3B (MoE)", family: "Qwen", category: "text",      paramsB: 30.0, activeParamsB: 3.0,  layers: 48, hiddenDim: 2048,  qHeads: 32, kvHeads: 4,  headDim: 128, vocabSize: 151936, maxContext: 40960,  isMoE: true  },
  { id: "qwen3-235b-a22b", name: "Qwen 3 235B-A22B (MoE)", family: "Qwen", category: "text", paramsB: 235,  activeParamsB: 22,   layers: 94, hiddenDim: 4096,  qHeads: 64, kvHeads: 4,  headDim: 128, vocabSize: 151936, maxContext: 40960,  isMoE: true  },
  { id: "qwen2-5-vl-7b",  name: "Qwen 2.5-VL 7B (VLM)",   family: "Qwen", category: "vlm",  paramsB: 7.0,  activeParamsB: 7.0,  layers: 28, hiddenDim: 3584,  qHeads: 28, kvHeads: 4,  headDim: 128, vocabSize: 152064, maxContext: 128000, isMoE: false },
  { id: "qwen2-5-vl-72b", name: "Qwen 2.5-VL 72B (VLM)",  family: "Qwen", category: "vlm",  paramsB: 72.0, activeParamsB: 72.0, layers: 80, hiddenDim: 8192,  qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 152064, maxContext: 128000, isMoE: false },

  // === DeepSeek family ===
  { id: "deepseek-v3",name: "DeepSeek V3 671B (MoE)",family:"DeepSeek",category:"text",     paramsB: 671,  activeParamsB: 37,   layers: 61, hiddenDim: 7168,  qHeads: 128,kvHeads: 128,headDim: 128, vocabSize: 102400, maxContext: 65536,  isMoE: true  },
  { id: "deepseek-r1",name: "DeepSeek R1 671B (MoE)",family:"DeepSeek",category:"reasoning", paramsB: 671,  activeParamsB: 37,   layers: 61, hiddenDim: 7168,  qHeads: 128,kvHeads: 128,headDim: 128, vocabSize: 102400, maxContext: 65536,  isMoE: true  },
  { id: "deepseek-coder-v2", name: "DeepSeek Coder V2 236B (MoE)", family:"DeepSeek", category:"code", paramsB: 236, activeParamsB: 21, layers: 47, hiddenDim: 6144, qHeads: 64, kvHeads: 8,  headDim: 128, vocabSize: 102400, maxContext: 131072, isMoE: true  },

  // === Google Gemma ===
  { id: "gemma2-9b", name: "Gemma 2 9B",           family: "Gemma",   category: "text",      paramsB: 9.0,  activeParamsB: 9.0,  layers: 42, hiddenDim: 3584,  qHeads: 16, kvHeads: 8,  headDim: 256, vocabSize: 256000, maxContext: 8192,   isMoE: false },
  { id: "gemma2-27b", name: "Gemma 2 27B",         family: "Gemma",   category: "text",      paramsB: 27.0, activeParamsB: 27.0, layers: 46, hiddenDim: 4608,  qHeads: 32, kvHeads: 16, headDim: 128, vocabSize: 256000, maxContext: 8192,   isMoE: false },

  // === Microsoft Phi ===
  { id: "phi3-7b",    name: "Phi-3 Mini 3.8B",     family: "Phi",     category: "text",      paramsB: 3.8,  activeParamsB: 3.8,   layers: 32, hiddenDim: 3072,  qHeads: 32, kvHeads: 32, headDim: 96,  vocabSize: 32000, maxContext: 4096,   isMoE: false },
  { id: "phi4-14b",   name: "Phi-4 14B",            family: "Phi",     category: "text",      paramsB: 14.0, activeParamsB: 14.0, layers: 40, hiddenDim: 5120,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 100352, maxContext: 16384,  isMoE: false },

  // === Small / open research ===
  { id: "smollm2-1.7b",name: "SmolLM2 1.7B Instruct", family: "HuggingFaceTB", category: "text", paramsB: 1.7, activeParamsB: 1.7, layers: 24, hiddenDim: 2048, qHeads: 32, kvHeads: 32, headDim: 64,  vocabSize: 128256, maxContext: 8192,   isMoE: false },
  { id: "falcon3-10b",name: "Falcon 3 10B Instruct", family: "TII",     category: "text",      paramsB: 10.0, activeParamsB: 10.0, layers: 40, hiddenDim: 4096,  qHeads: 32, kvHeads: 8,  headDim: 128, vocabSize: 49152, maxContext: 32768,  isMoE: false },
  { id: "olmo2-13b", name: "OLMo 2 13B",            family: "AllenAI",  category: "text",      paramsB: 13.0, activeParamsB: 13.0, layers: 40, hiddenDim: 5120,  qHeads: 40, kvHeads: 8,  headDim: 128, vocabSize: 50281, maxContext: 4096,   isMoE: false },
  { id: "gpt-neox-20b",name:"GPT-NeoX 20B",         family:"EleutherAI",category: "text",     paramsB: 20.0, activeParamsB: 20.0, layers: 44, hiddenDim: 6144,  qHeads: 64, kvHeads: 64, headDim: 96,  vocabSize: 50432, maxContext: 2048,   isMoE: false },

  // === Embedding models (separate workload — encoder, not autoregressive) ===
  // Refs: https://huggingface.co/BAAI/bge-m3 · https://huggingface.co/intfloat/multilingual-e5-large
  { id: "bge-m3",     name: "BGE-M3 (embed, 1024-dim)",   family: "BAAI",     category: "embedding", paramsB: 0.568, activeParamsB: 0.568, layers: 24, hiddenDim: 1024, qHeads: 16, kvHeads: 16, headDim: 64,  vocabSize: 250002, maxContext: 8192, isMoE: false },
  { id: "e5-mlarge",  name: "Multilingual E5 Large (1024-dim)", family: "intfloat", category: "embedding", paramsB: 0.56, activeParamsB: 0.56, layers: 24, hiddenDim: 1024, qHeads: 16, kvHeads: 16, headDim: 64, vocabSize: 250047, maxContext: 514, isMoE: false },
  { id: "gte-large",  name: "GTE Large (1024-dim)",    family: "thenlper", category: "embedding", paramsB: 0.44, activeParamsB: 0.44, layers: 24, hiddenDim: 1024, qHeads: 16, kvHeads: 16, headDim: 64, vocabSize: 30522, maxContext: 512,  isMoE: false },
];

export const MODEL_MAP: Record<string, ModelSpec> = Object.fromEntries(MODELS.map((m) => [m.id, m]));

/** ---------- EFFICIENCY CONSTANTS ---------- */
export const ETA_MEM = 0.65;       // Memory bandwidth utilization (real-world)
export const ETA_COMPUTE = 0.50;   // Compute utilization
export const ETA_NVLINK = 0.85;    // NVLink tensor-parallel communication efficiency
export const KERNEL_OVERHEAD_MS = 30; // per-request overhead

/**
 * Confidence levels surfaced in the UI to help users trust (or question)
 * specific numbers. The research brief called this "honest uncertainty" —
 * a key credibility signal for technical audiences (HN/Reddit/ML engineers).
 *
 *  - "measured"     → sourced directly from official spec sheets, model cards,
 *                     API pricing pages, or HuggingFace config.json
 *  - "modeled"      → derived from physics-based formulas with cited sources
 *                     (e.g., decode = HBM_BW × η_mem / model_size)
 *  - "inferred"     → derived from heuristics with known error bars
 *                     (e.g., continuous batching multiplier 1.5× default,
 *                     long-context attention O(N²) correction, B200 null FP16 fallback)
 *  - "user-supplied" → user-provided value (e.g., GPU $/hr override)
 */
export type Confidence = "measured" | "modeled" | "inferred" | "user-supplied";

/**
 * Map of metric name → confidence level.
 * Surfaced in the UI as a colored dot + tooltip next to each metric.
 */
export type ConfidenceMap = Record<string, Confidence>;

/** Continuous-batching multiplier guidance:
 *  - Research warns NO universal multiplier exists
 *  - vLLM reported 14–24x vs HF Transformers (extreme), 2.2–2.5x vs TGI
 *  - SOSP paper: 2–4x throughput vs FasterTransformer/Orca at same latency
 *  - Conservative default: 1.5x (clearly cited as "benchmark-derived range, not universal")
 *  Refs: https://arxiv.org/abs/2309.06180 (vLLM/PagedAttention paper)
 */
export const DEFAULT_BATCHING_MULTIPLIER = 1.5;

/** Anthropic prompt caching multipliers (verified 2025):
 *  - 5-minute cache write: 1.25x base input
 *  - 1-hour cache write: 2.0x base input
 *  - cache read: 0.1x base input (90% savings)
 *  - OpenAI: 50% off cached input (no separate write fee)
 *  Refs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
 */
export const ANTHROPIC_CACHE_WRITE_5M = 1.25;
export const ANTHROPIC_CACHE_WRITE_1H = 2.0;
export const ANTHROPIC_CACHE_READ = 0.1;
export const OPENAI_CACHE_DISCOUNT = 0.5;

export type CacheTTL = "none" | "5m" | "1h";

export interface CalcInput {
  modelId: string;
  gpuId: string;
  quantization: Quantization;
  numGpus: number;            // tensor parallel degree
  batchSize: number;          // concurrent requests
  promptTokens: number;        // input length (excluding cache prefix)
  outputTokens: number;        // requested output (visible answer)
  gpuHourlyCost?: number;     // optional override
  useSpeculative?: boolean;   // speculative decoding?
  speculativeBoost?: number;  // multiplier e.g. 2.0
  // === Phase 2 additions ===
  useContinuousBatching?: boolean;            // continuous batching toggle
  continuousBatchingMultiplier?: number;     // user-tunable 1.0–4.0x
  reasoningTokens?: number;                   // hidden reasoning budget (o1/R1/thinking)
  // Prompt caching (self-hosted or API)
  cachePrefixTokens?: number;                 // reusable prefix length (e.g. system prompt + RAG context)
  cacheHitRate?: number;                       // 0–1 fraction of requests that hit the cache
  cacheTTL?: CacheTTL;                        // "none" | "5m" | "1h"
  cacheProvider?: "self-hosted" | "anthropic" | "openai";
}

export interface CalcResult {
  // ---- Raw measurements ----
  modelSizeGb: number;
  kvCachePerTokenKb: number;
  kvCacheTotalGb: number;
  totalVramNeededGb: number;
  vramFits: boolean;

  // ---- Throughput ----
  decodeTokensPerSec: number;      // per-stream (batch=1)
  aggregateTokensPerSec: number;   // batched (with continuous batching multiplier if enabled)
  prefillTokensPerSec: number;     // prefill throughput
  batchCrossover: number;          // batch size where compute becomes bound

  // ---- Latency (new: split into TTFT + ITL) ----
  prefillTimeMs: number;           // = TTFT (time-to-first-token)
  decodeTimePerTokenMs: number;    // = ITL (inter-token latency)
  totalDecodeTimeMs: number;
  totalLatencyMs: number;
  endToEndTokensPerSec: number;    // output / total time
  ttftMs: number;                  // alias for prefillTimeMs — for clarity
  itlMs: number;                   // alias for decodeTimePerTokenMs — for clarity

  // ---- Reasoning ----
  billedOutputTokens: number;     // visible + hidden reasoning tokens

  // ---- Prompt caching ----
  cacheHitRate: number;
  cachePrefixTokens: number;
  cacheTTL: CacheTTL;
  cacheProvider: string;
  prefillTokensAvoided: number;   // how many prefill tokens saved per cache hit
  cacheWriteCostMultiplier: number;  // 1.25 (5m) or 2.0 (1h) for Anthropic; 1.0 for OpenAI
  cacheReadCostMultiplier: number;   // 0.1 (Anthropic) or 0.5 (OpenAI) or 0 (self-hosted)
  effectivePrefillTokens: number;  // after cache hit adjustment
  prefillTimeWithCacheMs: number;  // actual prefill time accounting for cache
  cacheSavingsPct: number;         // % reduction in prefill time from caching

  // ---- Cost ----
  costPerHour: number;
  costPerMillionOutputTokens: number;
  costPerRequest: number;
  costPerRequestWithCache: number; // including cache write/read cost

  // ---- Components for display ----
  effectiveBandwidthGbps: number;
  effectiveFlopsTflops: number;
  quantEfficiency: number;
  speculativeBoost: number;
  multiGpuEfficiency: number;
  continuousBatchingMultiplier: number;  // actual multiplier applied
  longContextWarning?: string;          // set when prompt > 32K
  // Per-metric confidence map — surfaced as colored dots in the UI
  // (research: "honest uncertainty" = credibility signal for HN/Reddit/ML engineers)
  confidence: ConfidenceMap;
}

/**
 * Main calculation function.
 *
 * Math basis:
 *   decode_tokens_per_sec  ≈  (HBM_BW × η_mem × quant_eff) / model_size   (batch=1, mem-bound)
 *   prefill_tokens_per_sec ≈  (FLOPS × η_compute) / (2 × active_params)
 *   batch_crossover        ≈  (model_size × FLOPS × η_compute) / (2 × HBM_BW × η_mem × quant_eff)
 *                            = bytes_per_param × FLOPS × η_compute / (2 × HBM_BW × η_mem × quant_eff)
 *   total_latency          = prefill_time + output × decode_time + overhead
 *   cost_per_M_tokens     = (gpu_hourly_cost / 3600) / (decode_tokens_per_sec × batch) × 1e6
 */
export function calculate(input: CalcInput): CalcResult {
  const model = MODEL_MAP[input.modelId];
  const gpu = GPU_MAP[input.gpuId];
  const quant = QUANT_MAP[input.quantization];

  if (!model) throw new Error(`Unknown model: ${input.modelId}`);
  if (!gpu) throw new Error(`Unknown GPU: ${input.gpuId}`);

  // ---- Model size ----
  // For MoE, only active params are loaded per token; full params still occupy VRAM.
  const modelSizeGb = (model.activeParamsB * quant.bytesPerParam);

  // ---- Effective bandwidth / compute ----
  const quantEff = quant.efficiency;
  const tp = Math.max(1, input.numGpus);
  const multiGpuEfficiency = tp > 1 ? ETA_NVLINK : 1.0;
  const speculativeBoost = input.useSpeculative ? (input.speculativeBoost ?? 2.0) : 1.0;
  const continuousBatchingMultiplier = input.useContinuousBatching
    ? (input.continuousBatchingMultiplier ?? DEFAULT_BATCHING_MULTIPLIER)
    : 1.0;

  // Aggregate bandwidth across TP GPUs
  const effectiveBandwidthGbps = gpu.memBandwidthGbps * tp * ETA_MEM * multiGpuEfficiency;
  // Some accelerators (B200/B300) don't publish dense FP16 — fall back to ~2.5x FP8 as conservative estimate.
  const gpuFlops = gpu.flopsTflops ?? 1500;  // conservative fallback if null
  const effectiveFlopsTflops = gpuFlops * tp * ETA_COMPUTE;

  // ---- KV cache ----
  // KV cache per token = 2 (K&V) × layers × kvHeads × headDim × 2 bytes (FP16)
  const kvBytesPerToken = 2 * model.layers * model.kvHeads * model.headDim * 2;
  const kvCachePerTokenKb = kvBytesPerToken / 1024;
  // KV cache size depends on total context (prefix + suffix) × batch
  const totalContextTokens = input.promptTokens + (input.cachePrefixTokens ?? 0);
  const kvCacheTotalGb =
    (kvBytesPerToken * totalContextTokens * input.batchSize) / 1e9;

  // ---- Total VRAM ----
  const fullWeightsGb = model.paramsB * quant.bytesPerParam;
  const totalVramNeededGb = fullWeightsGb + kvCacheTotalGb;
  const vramFits = totalVramNeededGb <= gpu.vramGb * tp;

  // ---- Throughput ----
  const decodeTokensPerSecRaw = effectiveBandwidthGbps / modelSizeGb * quantEff * speculativeBoost;
  const decodeTokensPerSec = vramFits ? decodeTokensPerSecRaw : 0;

  // Prefill / compute-bound ceiling
  const prefillTokensPerSec = (effectiveFlopsTflops * 1e12) / (2 * model.activeParamsB * 1e9);
  const computeCeiling = prefillTokensPerSec; // same ceiling applies

  // Batch crossover (unit-corrected with ×1000 for TFLOPS/GB-s ratio)
  const batchCrossover =
    (quant.bytesPerParam * gpuFlops * 1000 * ETA_COMPUTE) /
    (2 * gpu.memBandwidthGbps * ETA_MEM * quantEff * (tp > 1 ? multiGpuEfficiency : 1));

  // Aggregate tokens/sec — with continuous batching multiplier applied
  const memoryBoundAggregate = decodeTokensPerSec * input.batchSize * continuousBatchingMultiplier;
  const aggregateTokensPerSec = vramFits
    ? Math.min(memoryBoundAggregate, computeCeiling)
    : 0;

  // ---- Prompt caching ----
  const cachePrefixTokens = input.cachePrefixTokens ?? 0;
  const cacheHitRate = input.cacheHitRate ?? 0;
  const cacheTTL = input.cacheTTL ?? "none";
  const cacheProvider = input.cacheProvider ?? "self-hosted";

  // Compute cache write/read multipliers based on provider
  let cacheWriteCostMultiplier = 1.0;
  let cacheReadCostMultiplier = 0.0; // self-hosted: cache reads are "free" (just memory access)
  if (cacheProvider === "anthropic") {
    cacheWriteCostMultiplier = cacheTTL === "1h" ? ANTHROPIC_CACHE_WRITE_1H : ANTHROPIC_CACHE_WRITE_5M;
    cacheReadCostMultiplier = ANTHROPIC_CACHE_READ;
  } else if (cacheProvider === "openai") {
    cacheWriteCostMultiplier = 1.0; // OpenAI: no separate write fee
    cacheReadCostMultiplier = OPENAI_CACHE_DISCOUNT;
  }

  // Effective prefill tokens after cache hit (fraction of prefix avoided per request, on average)
  // For self-hosted: full prefix is avoided on hit. For API: cost is reduced, not avoided.
  const prefillTokensAvoided = Math.round(cachePrefixTokens * cacheHitRate);
  const effectivePrefillTokens = Math.max(0, totalContextTokens - prefillTokensAvoided);
  const prefillTimeWithCacheMs = effectivePrefillTokens > 0
    ? (effectivePrefillTokens / prefillTokensPerSec) * 1000
    : 0;
  const cacheSavingsPct = totalContextTokens > 0
    ? (prefillTokensAvoided / totalContextTokens) * 100
    : 0;

  // ---- Latency ----
  // Use cache-adjusted prefill time if caching is enabled
  const prefillTimeMs = input.promptTokens > 0
    ? (input.useContinuousBatching || cachePrefixTokens > 0
        ? prefillTimeWithCacheMs
        : (input.promptTokens / prefillTokensPerSec) * 1000)
    : 0;
  const decodeTimePerTokenMs = decodeTokensPerSec > 0 ? (1000 / decodeTokensPerSec) : Infinity;

  // ---- Reasoning tokens (billed as output but invisible to user) ----
  const reasoningTokens = input.reasoningTokens ?? 0;
  const billedOutputTokens = input.outputTokens + reasoningTokens;
  const totalDecodeTimeMs = billedOutputTokens * decodeTimePerTokenMs;
  const totalLatencyMs = prefillTimeMs + totalDecodeTimeMs + KERNEL_OVERHEAD_MS;
  const endToEndTokensPerSec =
    totalLatencyMs > 0 ? (billedOutputTokens / totalLatencyMs) * 1000 : 0;

  // ---- Cost ----
  const costPerHour = input.gpuHourlyCost !== undefined ? input.gpuHourlyCost : (gpu.usdPerHour ?? 0);
  const costPerSecond = costPerHour / 3600;
  const costPerRequest =
    (totalLatencyMs / 1000) * costPerSecond * tp;
  const costPerMillionOutputTokens =
    aggregateTokensPerSec > 0 ? (costPerSecond * tp * 1e6) / aggregateTokensPerSec : Infinity;

  // Cost with cache — for self-hosted, same as costPerRequest (latency-driven)
  // For API providers, cache economics would be computed separately in the Build-vs-Buy tab
  const costPerRequestWithCache = costPerRequest;

  // ---- Long-context warning ----
  let longContextWarning: string | undefined;
  if (totalContextTokens > 32768) {
    longContextWarning = `Context ${totalContextTokens.toLocaleString()} tokens > 32K. ` +
      `Attention cost grows superlinearly with context for dense models; ` +
      `real-world throughput may be 20–40% lower than this estimate.`;
  }

  return {
    modelSizeGb,
    kvCachePerTokenKb,
    kvCacheTotalGb,
    totalVramNeededGb,
    vramFits,
    decodeTokensPerSec,
    aggregateTokensPerSec,
    prefillTokensPerSec,
    batchCrossover,
    prefillTimeMs,
    decodeTimePerTokenMs,
    totalDecodeTimeMs,
    totalLatencyMs,
    endToEndTokensPerSec,
    ttftMs: prefillTimeMs,
    itlMs: decodeTimePerTokenMs,
    billedOutputTokens,
    cacheHitRate,
    cachePrefixTokens,
    cacheTTL,
    cacheProvider,
    prefillTokensAvoided,
    cacheWriteCostMultiplier,
    cacheReadCostMultiplier,
    effectivePrefillTokens,
    prefillTimeWithCacheMs,
    cacheSavingsPct,
    costPerHour: costPerHour * tp,
    costPerMillionOutputTokens,
    costPerRequest,
    costPerRequestWithCache,
    effectiveBandwidthGbps,
    effectiveFlopsTflops,
    quantEfficiency: quantEff,
    speculativeBoost,
    multiGpuEfficiency,
    continuousBatchingMultiplier,
    longContextWarning,
    // Per-metric confidence levels — surfaced as colored dots in the UI
    confidence: {
      // Throughput metrics
      decodeTokensPerSec: "modeled",        // Formula: HBM_BW × η_mem × quant_eff / model_size
      aggregateTokensPerSec: continuousBatchingMultiplier > 1 ? "inferred" : "modeled",
                                            // Multiplier is heuristic, default 1.5× with cited 2-4× range
      prefillTokensPerSec: totalContextTokens > 32768 ? "inferred" : "modeled",
                                            // Long-context superlinear attention is approximation
      batchCrossover: "modeled",           // Crossover formula is exact

      // Latency metrics
      prefillTimeMs: totalContextTokens > 32768 ? "inferred" : "modeled",
      decodeTimePerTokenMs: "modeled",     // = 1000 / decode_tok_per_sec
      totalLatencyMs: totalContextTokens > 32768 ? "inferred" : "modeled",
                                            // Inherits from prefill (long context dominates)
      ttftMs: totalContextTokens > 32768 ? "inferred" : "modeled",
      itlMs: "modeled",

      // Memory metrics
      modelSizeGb: "modeled",              // params × bytes_per_param
      kvCacheTotalGb: "modeled",           // Exact formula: 2·L·H_kv·D_h·B·T·batch
      totalVramNeededGb: "modeled",        // Sum of weights + KV
      vramFits: "modeled",

      // Cost metrics
      costPerHour: input.gpuHourlyCost !== undefined ? "user-supplied" : "measured",
                                            // User override or default from spec
      costPerMillionOutputTokens: "modeled",
      costPerRequest: "modeled",

      // Cache metrics
      cacheSavingsPct: "modeled",
      prefillTokensAvoided: "modeled",
      cacheWriteCostMultiplier: "measured",  // From Anthropic/OpenAI official docs
      cacheReadCostMultiplier: "measured",

      // Other
      billedOutputTokens: "user-supplied",  // User sets reasoning tokens
      effectiveBandwidthGbps: "measured",   // GPU spec × user-multipliers
      effectiveFlopsTflops: gpu.flopsTflops === null ? "inferred" : "measured",
                                            // B200/B300 null → conservative fallback
    },
  };
}

/** Format helpers */
export function fmtTokens(n: number): string {
  if (!isFinite(n)) return "—";
  if (n >= 1000) return n.toFixed(0);
  if (n >= 100) return n.toFixed(1);
  if (n >= 10) return n.toFixed(2);
  return n.toFixed(3);
}

export function fmtBytes(gb: number): string {
  if (gb >= 1024) return (gb / 1024).toFixed(2) + " TB";
  if (gb >= 1) return gb.toFixed(2) + " GB";
  return (gb * 1024).toFixed(1) + " MB";
}

export function fmtMs(ms: number): string {
  if (!isFinite(ms)) return "—";
  if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
  if (ms >= 1) return ms.toFixed(1) + " ms";
  return (ms * 1000).toFixed(1) + " µs";
}

export function fmtMoney(n: number): string {
  if (!isFinite(n) || n === 0) return "—";
  if (n >= 1000) return "$" + n.toFixed(0);
  if (n >= 1) return "$" + n.toFixed(2);
  if (n >= 0.01) return "$" + n.toFixed(4);
  return "$" + n.toFixed(6);
}

/* ============================================================
   LONG-CONTEXT CAPACITY HELPERS
   ============================================================
   The research brief calls this "the most important formula tokcalc should visibly expose":

     KV bytes/request = 2 · L · T · H_kv · D_h · B

   The factor of 2 stores both keys and values. With GQA (num_kv_heads < num_attention_heads),
   only the kv_heads' worth of cache is stored per layer.

   For dense attention, prefill cost grows SUPERLINEARLY:
     linear_flops    = 2 · N · T         (matmuls through linear layers)
     attention_flops ≈ T²/2 · H_kv · D_h · L   (QK^T + softmax · V)
   At long contexts (≥32K), attention becomes a significant fraction of total prefill FLOPs.
   Refs: RingAttention paper (https://arxiv.org/abs/2310.01889)
*/

/** Bytes of KV cache per single token (for one request, one head). */
export function computeKVBytesPerToken(
  model: ModelSpec,
  bytesPerKVValue: number = 2, // FP16 = 2 bytes; can be INT8=1, FP4=0.5
): number {
  // 2 = K and V; model.layers = number of transformer blocks
  return 2 * model.layers * model.kvHeads * model.headDim * bytesPerKVValue;
}

/** Total KV cache size in GB for `contextTokens` tokens × `batchSize` requests. */
export function computeKVCacheGb(
  model: ModelSpec,
  contextTokens: number,
  batchSize: number = 1,
  bytesPerKVValue: number = 2,
): number {
  const bytes = computeKVBytesPerToken(model, bytesPerKVValue) * contextTokens * batchSize;
  return bytes / 1e9;
}

/** Maximum concurrent users that fit in VRAM at the given context length.
 *  = floor((total_vram - model_weights) / kv_per_request_at_context)
 */
export function computeMaxConcurrency(
  model: ModelSpec,
  gpu: GpuSpec,
  numGpus: number,
  contextTokens: number,
  quantBytesPerParam: number,
  bytesPerKVValue: number = 2,
): number {
  const weightsBytes = model.paramsB * 1e9 * quantBytesPerParam;
  const kvBytesPerRequest = computeKVBytesPerToken(model, bytesPerKVValue) * contextTokens;
  const totalVramBytes = gpu.vramGb * numGpus * 1e9;
  const availableKvBytes = totalVramBytes - weightsBytes;
  if (availableKvBytes <= 0 || kvBytesPerRequest <= 0) return 0;
  return Math.floor(availableKvBytes / kvBytesPerRequest);
}

/** Prefill latency in milliseconds for a given context length, accounting for
 *  superlinear attention cost beyond 32K tokens (dense attention only).
 *  For RingAttention/blockwise-attention, this correction overestimates the cost.
 */
export function computeLongContextPrefillMs(
  model: ModelSpec,
  effectiveFlopsTflops: number,
  promptTokens: number,
): number {
  if (promptTokens <= 0 || effectiveFlopsTflops <= 0) return 0;
  const T = promptTokens;
  const N = model.activeParamsB * 1e9;
  const linearFlops = 2 * N * T;
  // Attention O(N²) — only meaningful for long context
  // Each layer: QK^T = T·T·headDim, softmax·V = T·T·headDim
  const attentionFlops = (T * T) * model.kvHeads * model.headDim * model.layers;
  const totalFlops = linearFlops + attentionFlops;
  const effFlopsPerSec = effectiveFlopsTflops * 1e12;
  return (totalFlops / effFlopsPerSec) * 1000;
}

export interface TopologyRecommendation {
  topology: string;
  reason: string;
  fits: boolean;
  neededGpus: number;
  hasContextParallel: boolean;
}

/** Recommend a multi-GPU topology for the given config.
 *  - Single GPU: model + KV fit in 1 GPU
 *  - TP×2/4/8: shard weights across N GPUs (each holds 1/N of weights + KV)
 *  - TP×8 + Context Parallel: even TP×8 isn't enough — need RingAttention to shard KV across nodes
 */
export function recommendTopology(
  model: ModelSpec,
  gpu: GpuSpec,
  contextTokens: number,
  batchSize: number,
  quantBytesPerParam: number,
  bytesPerKVValue: number = 2,
): TopologyRecommendation {
  const weightsGb = model.paramsB * quantBytesPerParam;
  const kvPerRequestGb = computeKVGb(model, contextTokens, 1, bytesPerKVValue);
  const totalKvGb = kvPerRequestGb * batchSize;
  const totalNeededGb = weightsGb + totalKvGb;

  // Try single-GPU first
  if (totalNeededGb <= gpu.vramGb) {
    return {
      topology: "Single GPU",
      reason: `${totalNeededGb.toFixed(1)} GB ≤ ${gpu.vramGb} GB — fits in 1 ${gpu.name.split(" ")[0]}`,
      fits: true,
      neededGpus: 1,
      hasContextParallel: false,
    };
  }
  // Try TP×2, TP×4, TP×8
  for (const tp of [2, 4, 8]) {
    if (totalNeededGb <= gpu.vramGb * tp) {
      return {
        topology: `Tensor Parallel ×${tp}`,
        reason: `${totalNeededGb.toFixed(1)} GB > ${gpu.vramGb} GB single → shard across ${tp} GPUs (${gpu.vramGb * tp} GB total, weights + KV split evenly)`,
        fits: true,
        neededGpus: tp,
        hasContextParallel: false,
      };
    }
  }
  // TP×8 not enough → need Context Parallel (RingAttention)
  return {
    topology: "TP×8 + Context Parallel",
    reason: `${totalNeededGb.toFixed(1)} GB exceeds ${gpu.vramGb * 8} GB even with TP×8. Use Context Parallel (RingAttention) to shard KV cache across nodes. Ref: arxiv.org/abs/2310.01889`,
    fits: false,
    neededGpus: 8,
    hasContextParallel: true,
  };
}

/** Helper for above — used internally. */
function computeKVGb(
  model: ModelSpec,
  contextTokens: number,
  batchSize: number,
  bytesPerKVValue: number,
): number {
  return computeKVCacheGb(model, contextTokens, batchSize, bytesPerKVValue);
}

/** Format a context length nicely: 8192 → "8K", 131072 → "128K", 1000000 → "1M". */
export function fmtContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    // Show "1M", "1.5M", "2M"
    const m = tokens / 1_000_000;
    if (m >= 1) {
      const rounded = Math.round(m * 10) / 10;
      return Number.isInteger(rounded) ? `${rounded}M` : `${rounded.toFixed(1)}M`;
    }
  }
  if (tokens >= 1000) {
    // Floor for power-of-2 context sizes (4096→"4K", 131072→"131K")
    // Use 1 decimal only if not within 1 of an integer
    const k = tokens / 1000;
    const rounded = Math.round(k);
    if (Math.abs(k - rounded) < 0.5) {
      return `${rounded}K`;
    }
    return `${k.toFixed(1)}K`;
  }
  return String(tokens);
}
