/**
 * Engine-aware presets for tokcalc.
 *
 * When the user picks an inference engine (vLLM, SGLang, TRT-LLM, llama.cpp),
 * the calculator adjusts:
 *   1. Efficiency factors (ETA_MEM, ETA_COMPUTE) — each engine has different
 *      real-world utilization due to kernel optimizations, scheduler design, etc.
 *   2. Default continuous batching multiplier — vLLM/SGLang have it on by
 *      default (2.0×), llama.cpp doesn't (1.0×)
 *   3. Supported quantizations — filters the quant dropdown to only show
 *      formats the selected engine supports
 *   4. Engine-specific notes — helps users understand trade-offs
 *
 * Per Perplexity research (Prompt #2):
 *   "There is no universally best engine; workload shape matters."
 *   "comparisons emphasize that throughput leadership varies with workload,
 *    structured output, prefix reuse, and operational maturity"
 */

import type { Quantization } from "./token-calc";

export type EngineId = "vllm" | "sglang" | "trtllm" | "llamacpp" | "generic";

export interface EnginePreset {
  id: EngineId;
  name: string;
  shortName: string;
  description: string;
  /** Memory bandwidth utilization (real-world) — replaces ETA_MEM when engine is selected */
  etaMem: number;
  /** Compute utilization — replaces ETA_COMPUTE */
  etaCompute: number;
  /** Default continuous batching multiplier when engine is selected */
  defaultBatchingMultiplier: number;
  /** Which quantization formats this engine supports */
  supportedQuants: Quantization[];
  /** Engine-specific notes shown in UI */
  notes: string;
  /** Source URL for the engine */
  sourceUrl: string;
  /** Whether continuous batching is on by default in this engine */
  batchingOnByDefault: boolean;
  /** Whether prefix caching is on by default */
  prefixCacheOnByDefault: boolean;
  /** Key features of this engine */
  features: string[];
}

export const ENGINE_PRESETS: EnginePreset[] = [
  {
    id: "vllm",
    name: "vLLM",
    shortName: "vLLM",
    description: "High-throughput LLM serving with PagedAttention + continuous batching. The most popular open-source serving engine.",
    etaMem: 0.65,        // Default — PagedAttention reduces waste to <4%
    etaCompute: 0.50,    // Default
    defaultBatchingMultiplier: 2.0,  // vLLM enables continuous batching by default
    supportedQuants: ["fp32", "fp16", "bf16", "int8", "fp8", "gptq4", "awq4", "int4"],
    notes: "PagedAttention + continuous batching ON by default. Real-world throughput is 2-4× naive batching. Supports FP8 natively on H100/H200. GGUF not supported — use llama.cpp for that.",
    sourceUrl: "https://docs.vllm.ai",
    batchingOnByDefault: true,
    prefixCacheOnByDefault: true,  // APC (Automatic Prefix Caching)
    features: [
      "PagedAttention (KV cache memory management)",
      "Continuous batching (iteration-level scheduling)",
      "Automatic prefix caching (APC)",
      "Tensor parallelism (up to 8×)",
      "Speculative decoding",
      "FP8 native on H100/H200",
    ],
  },
  {
    id: "sglang",
    name: "SGLang",
    shortName: "SGLang",
    description: "High-performance serving with RadixAttention (prefix-tree KV cache) + structured generation. Best for agent workloads with repeated prefixes.",
    etaMem: 0.68,        // Slightly higher — RadixAttention is more efficient than APC
    etaCompute: 0.52,
    defaultBatchingMultiplier: 2.0,  // Continuous batching on by default
    supportedQuants: ["fp32", "fp16", "bf16", "int8", "fp8", "gptq4", "awq4"],
    notes: "RadixAttention provides automatic prefix reuse without explicit cache management. Better for agent/RAG workloads with shared system prompts. Slightly higher memory efficiency than vLLM for prefix-heavy workloads.",
    sourceUrl: "https://github.com/sgl-project/sglang",
    batchingOnByDefault: true,
    prefixCacheOnByDefault: true,  // RadixAttention IS the prefix cache
    features: [
      "RadixAttention (prefix-tree KV cache)",
      "Continuous batching",
      "Structured generation (JSON mode)",
      "Tensor parallelism",
      "Speculative decoding",
      "PD disaggregation (experimental)",
    ],
  },
  {
    id: "trtllm",
    name: "TensorRT-LLM",
    shortName: "TRT-LLM",
    description: "NVIDIA-optimized inference engine. Best raw throughput on NVIDIA GPUs with quantized models. Requires model compilation.",
    etaMem: 0.70,        // NVIDIA-optimized kernels achieve higher utilization
    etaCompute: 0.55,
    defaultBatchingMultiplier: 2.0,
    supportedQuants: ["fp32", "fp16", "bf16", "int8", "fp8", "nvfp4", "int4"],
    notes: "NVIDIA-optimized kernels achieve 5-10% higher utilization than vLLM on same hardware. NVFP4 native on Blackwell (B200/B300). Requires model compilation (TRT engine build) — slower to iterate than vLLM. Inflight batching is TRT-LLM's version of continuous batching.",
    sourceUrl: "https://github.com/NVIDIA/TensorRT-LLM",
    batchingOnByDefault: true,  // Inflight batching
    prefixCacheOnByDefault: true,
    features: [
      "NVIDIA-optimized kernels (FlashAttention, fused MLP)",
      "Inflight batching (continuous batching equivalent)",
      "FP8 + NVFP4 native on H100/H200/B200",
      "Model compilation (TRT engine build)",
      "Tensor parallelism",
      "Plugin system for custom layers",
    ],
  },
  {
    id: "llamacpp",
    name: "llama.cpp",
    shortName: "llama.cpp",
    description: "C++ inference engine for CPU, Apple Silicon, and consumer GPUs. Best for local/edge deployment. GGUF format.",
    etaMem: 0.55,        // Lower — CPU/GPU mixed, less optimized than vLLM
    etaCompute: 0.40,    // Lower — CPU-bound on non-GPU paths
    defaultBatchingMultiplier: 1.0,  // No continuous batching by default
    supportedQuants: ["fp32", "fp16", "bf16", "int8", "gguf-q2k", "gguf-q3km", "gguf-q4km", "gguf-q5km", "gguf-q6k", "gguf-q8", "int4"],
    notes: "Best for local/edge: Mac M-series (MLX backend), consumer GPUs (CUDA), CPU-only machines. Supports all GGUF quantization variants (Q2_K through Q8_0). No continuous batching — throughput is single-stream only. KV cache is contiguous (not paged).",
    sourceUrl: "https://github.com/ggml-org/llama.cpp",
    batchingOnByDefault: false,
    prefixCacheOnByDefault: false,
    features: [
      "GGUF format (Q2_K through Q8_0)",
      "CPU inference (no GPU needed)",
      "Apple Silicon (Metal backend)",
      "Consumer GPU support (CUDA)",
      "Lowest memory footprint",
      "No server needed (CLI + server mode)",
    ],
  },
  {
    id: "generic",
    name: "Generic (engine-agnostic)",
    shortName: "Generic",
    description: "No specific engine selected. Uses conservative default efficiency factors. Best when you don't know which engine you'll use yet.",
    etaMem: 0.65,        // Default — same as current ETA_MEM
    etaCompute: 0.50,    // Default
    defaultBatchingMultiplier: 1.5,  // Conservative default
    supportedQuants: [
      "fp32", "fp16", "bf16", "int8", "int4",
      "gguf-q2k", "gguf-q3km", "gguf-q4km", "gguf-q5km", "gguf-q6k", "gguf-q8",
      "gptq4", "awq4", "exl2-6bpw", "fp8", "nvfp4",
    ],
    notes: "Conservative estimates. When you know your engine, select it for more accurate results. Each engine has different real-world utilization due to kernel optimizations and scheduler design.",
    sourceUrl: "",
    batchingOnByDefault: false,
    prefixCacheOnByDefault: false,
    features: [],
  },
];

export const ENGINE_MAP: Record<EngineId, EnginePreset> = Object.fromEntries(
  ENGINE_PRESETS.map((e) => [e.id, e]),
) as Record<EngineId, EnginePreset>;

/**
 * Get the engine preset by ID (defaults to "generic").
 */
export function getEngine(id: EngineId | string | undefined): EnginePreset {
  return ENGINE_MAP[id as EngineId] || ENGINE_MAP.generic;
}

/**
 * Check if a quantization format is supported by an engine.
 */
export function isQuantSupported(engine: EngineId, quant: Quantization): boolean {
  const preset = getEngine(engine);
  return preset.supportedQuants.includes(quant);
}

/**
 * Get the list of quantizations supported by an engine, as Quantization[].
 */
export function getSupportedQuants(engine: EngineId): Quantization[] {
  return getEngine(engine).supportedQuants;
}
