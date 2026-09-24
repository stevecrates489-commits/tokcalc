"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Calculator,
  Gauge,
  Clock,
  DollarSign,
  Cpu,
  MemoryStick,
  Database,
  Layers,
  Zap,
  AlertTriangle,
  CheckCircle2,
  Github,
  Info,
  BookOpen,
  HelpCircle,
} from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { ConfidenceDot, ConfidenceBadge, ConfidenceLegend } from "@/components/confidence-badge";
import { BenchmarkImport } from "@/components/benchmark-import";
import { AzureLivePricing } from "@/components/azure-live-pricing";
import { VastAiLivePricing } from "@/components/vast-ai-live-pricing";
import type { Confidence } from "@/lib/token-calc";
import { track } from "@/lib/track";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  calculate,
  fmtBytes,
  fmtMoney,
  fmtMs,
  fmtTokens,
  fmtContext,
  GPUS,
  MODELS,
  QUANTIZATIONS,
  DEFAULT_BATCHING_MULTIPLIER,
  computeMaxConcurrency,
  computeLongContextPrefillMs,
  recommendTopology,
  computeKVCacheGb,
  type CacheTTL,
  type CalcInput,
  type GpuCategory,
  type Quantization,
} from "@/lib/token-calc";
import {
  parseUrlHash,
  serializeCalcState,
  serializeBvbState,
  writeUrlHash,
  copyCurrentUrlToClipboard,
  saveCalcToStorage,
  loadCalcFromStorage,
  saveBvbToStorage,
  loadBvbFromStorage,
  type CalcTabState,
} from "@/lib/url-state";
import { useToast } from "@/hooks/use-toast";
import { Share2, Check } from "lucide-react";

const CATEGORY_LABEL: Record<GpuCategory, string> = {
  datacenter: "Datacenter",
  workstation: "Workstation",
  consumer: "Consumer",
  mac: "Mac unified",
};

/* ---------- PLAIN-ENGLISH GLOSSARY ---------- */
/* Every technical term has a short, jargon-free explanation here. */
const GLOSSARY = {
  tokens: "A 'token' is roughly 3/4 of an English word. So 1,000 tokens ≈ 750 words. The model reads and writes text in tokens, not characters.",
  decode: "'Decode' = the model WRITING new tokens, one at a time, after reading your prompt. This is the slow part where users wait.",
  prefill: "'Prefill' = the model READING your prompt before it starts answering. Happens once per request, in parallel.",
  tokensPerSec: "How many tokens the model writes per second. Higher = faster responses for users.",
  singleStream: "When you serve just 1 user at a time. This is the max speed a single user will see.",
  aggregate: "Total tokens/sec when serving many users at once (batching). Always higher than single-stream because the GPU is shared efficiently.",
  latency: "Total wait time from when a user sends a prompt to when they get the full answer back.",
  vram: "Video RAM — the GPU's own memory. The model + its scratch space (KV cache) must fit entirely inside this. If it doesn't fit, the model won't run on this GPU.",
  modelWeights: "The actual 'brain' of the model — billions of numbers stored in VRAM. Size = params × bytes per param. Bigger = smarter but slower and pricier.",
  kvCache: "Scratch memory the model keeps while generating, so it doesn't re-read your prompt every token. Grows with conversation length and number of users.",
  params: "Number of 'neurons' (actually weight values) in the model. More params = smarter but slower & uses more VRAM. 'B' = billion. So 7B = 7,000,000,000 weights.",
  active: "For MoE (mixture-of-experts) models, only some experts activate per token. 'Active params' = what actually gets used per token, vs total params = the whole model size on disk.",
  layers: "How many 'stacked blocks' the model has. Deeper = smarter reasoning, but each token must pass through all of them, so it's slower.",
  hbmBw: "How fast the GPU can read its own memory (in GB per second). This is the #1 factor for LLM speed — the GPU must load the entire model weights once per generated token.",
  flops: "Trillions of math operations per second (Tera-FLOPS). Determines how fast the model can process your prompt (prefill phase) and large batches.",
  tensorParallel: "Splitting one model across multiple GPUs (each GPU holds part of the model). Lets you run models too big for 1 GPU, with near-linear speedup.",
  quantization: "Shrinking the model's weights from 16-bit numbers to 8-bit or 4-bit. Smaller = faster + cheaper, but slightly less accurate. INT4 = 4-bit, FP16 = 16-bit.",
  batchSize: "How many user requests the GPU serves at once. 1 = one user; 8 = eight users sharing one forward pass. Higher batch = more total throughput, but each user waits the same time.",
  promptTokens: "How long your input is. A 1-paragraph question is ~100 tokens; a 10-page document is ~3,000 tokens. Longer prompts take longer to read (prefill).",
  outputTokens: "How long the model's answer should be. 200 tokens ≈ 150 words. Longer answers take proportionally more time.",
  speculative: "A trick where a small 'draft' model guesses the next several tokens, then the big model verifies in one pass. Can 2-3x speed if the draft model is accurate.",
  batchCrossover: "The batch size where the GPU switches from 'memory-limited' to 'math-limited'. Below this, more users = more speed for free. Above this, you've maxed out the math capacity.",
  costPerMTokens: "What you pay to generate 1 million output tokens. The standard pricing unit for LLM APIs (e.g. GPT-4 charges $/1M tokens).",
  moe: "Mixture-of-Experts: a model with many 'expert' sub-networks, where only a few activate per token. Total size on disk is large, but per-token work is small. Example: Mixtral 8x7B has 47B total but only ~13B active per token.",
  gqa: "Grouped-Query Attention: an optimization that shares the same 'memory scratch space' (KV cache) across multiple attention heads. Cuts VRAM use significantly.",
  continuousBatching: "A serving trick where the GPU keeps generating tokens for active requests while NEW requests join mid-flight. Big speedup (1.5-4x typical) for production traffic. Used by vLLM, TGI, SGLang.",
  ttft: "Time-to-first-token: how long the user waits before seeing the first word. Equals prefill time (the model reading your prompt). Lower = better UX.",
  itl: "Inter-token-latency: how fast the model writes each token AFTER the first. This is what users perceive as 'streaming speed'.",
  promptCaching: "Reuse the model's work from a repeated prompt prefix (system prompt, RAG docs, tool schemas). On cache hit, you skip the prefill computation for that prefix. Anthropic: 90% off cached tokens. OpenAI: 50% off.",
  reasoningTokens: "Hidden tokens the model generates internally before answering (e.g. OpenAI o1, DeepSeek R1, Claude thinking). Billed as output but invisible to the user. Important for cost estimates of reasoning models.",
  gguf: "A file format used by llama.cpp for local inference. Has many sub-variants (Q2_K through Q8_0) trading size for quality. Q4_K_M is the recommended sweet spot for local Llama/Mistral.",
  fp8: "8-bit floating-point format. Native on NVIDIA H100/H200 — actually FASTER than FP16 thanks to dedicated FP8 tensor cores. Same VRAM as INT8 but better quality.",
  nvfp4: "NVIDIA's 4-bit floating-point format with block scaling. Native on Blackwell (B200/B300). Roughly 2x faster than FP8 on Blackwell per cited benchmarks.",
} as const;

type GlossaryKey = keyof typeof GLOSSARY;

type Tab = "calculator" | "build-vs-buy" | "reference";

export default function Home() {
  // ---- Top-level tab state ----
  const [activeTab, setActiveTab] = useState<Tab>("calculator");

  // ---- Inputs (Calculator tab) ----
  const [modelId, setModelId] = useState("llama3-8b");
  const [gpuId, setGpuId] = useState("a100-80");
  const [quantization, setQuantization] = useState<Quantization>("fp16");
  const [numGpus, setNumGpus] = useState(1);
  const [batchSize, setBatchSize] = useState(1);
  const [promptTokens, setPromptTokens] = useState(500);
  const [outputTokens, setOutputTokens] = useState(200);
  const [gpuHourlyCost, setGpuHourlyCost] = useState<number | "">("");
  const [useSpeculative, setUseSpeculative] = useState(false);
  const [speculativeBoost, setSpeculativeBoost] = useState(2.0);
  // === Phase 2 state ===
  const [useContinuousBatching, setUseContinuousBatching] = useState(false);
  const [continuousBatchingMultiplier, setContinuousBatchingMultiplier] = useState(DEFAULT_BATCHING_MULTIPLIER);
  const [reasoningTokens, setReasoningTokens] = useState(0);
  const [cachePrefixTokens, setCachePrefixTokens] = useState(0);
  const [cacheHitRate, setCacheHitRate] = useState(0);
  const [cacheTTL, setCacheTTL] = useState<CacheTTL>("5m");
  const [cacheProvider, setCacheProvider] = useState<"self-hosted" | "anthropic" | "openai">("self-hosted");

  // === URL-share state (Phase A) ===
  // Track whether we've restored state from URL — prevents the write effect
  // from clobbering the URL on first mount before we've read it.
  const [urlRestored, setUrlRestored] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const { toast } = useToast();

  // ---- One-time mount: read URL hash and restore state ----
  // Priority: URL hash > localStorage > defaults
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    if (typeof window === "undefined") return;
    const hash = window.location.hash;
    if (!hash || hash === "#") {
      // No URL hash — try localStorage for last-saved session
      const stored = loadCalcFromStorage();
      if (stored) {
        if (stored.modelId) setModelId(stored.modelId);
        if (stored.gpuId) setGpuId(stored.gpuId);
        if (stored.quantization) setQuantization(stored.quantization as Quantization);
        if (typeof stored.numGpus === "number") setNumGpus(stored.numGpus);
        if (typeof stored.batchSize === "number") setBatchSize(stored.batchSize);
        if (typeof stored.promptTokens === "number") setPromptTokens(stored.promptTokens);
        if (typeof stored.outputTokens === "number") setOutputTokens(stored.outputTokens);
        if (stored.gpuHourlyCost !== undefined) setGpuHourlyCost(stored.gpuHourlyCost);
        if (stored.useSpeculative) {
          setUseSpeculative(true);
          if (typeof stored.speculativeBoost === "number") setSpeculativeBoost(stored.speculativeBoost);
        }
        if (stored.useContinuousBatching) {
          setUseContinuousBatching(true);
          if (typeof stored.continuousBatchingMultiplier === "number") setContinuousBatchingMultiplier(stored.continuousBatchingMultiplier);
        }
        if (typeof stored.reasoningTokens === "number") setReasoningTokens(stored.reasoningTokens);
        if (typeof stored.cachePrefixTokens === "number") setCachePrefixTokens(stored.cachePrefixTokens);
        if (typeof stored.cacheHitRate === "number") setCacheHitRate(stored.cacheHitRate);
        if (stored.cacheTTL) setCacheTTL(stored.cacheTTL as CacheTTL);
        if (stored.cacheProvider) setCacheProvider(stored.cacheProvider as "self-hosted" | "anthropic" | "openai");
      }
      setUrlRestored(true);
      return;
    }
    const parsed = parseUrlHash(hash);
    setActiveTab(parsed.tab);
    if (parsed.calc) {
      const c = parsed.calc;
      if (c.modelId) setModelId(c.modelId);
      if (c.gpuId) setGpuId(c.gpuId);
      if (c.quantization) setQuantization(c.quantization as Quantization);
      if (typeof c.numGpus === "number") setNumGpus(c.numGpus);
      if (typeof c.batchSize === "number") setBatchSize(c.batchSize);
      if (typeof c.promptTokens === "number") setPromptTokens(c.promptTokens);
      if (typeof c.outputTokens === "number") setOutputTokens(c.outputTokens);
      if (c.gpuHourlyCost !== undefined) setGpuHourlyCost(c.gpuHourlyCost);
      if (c.useSpeculative) {
        setUseSpeculative(true);
        if (typeof c.speculativeBoost === "number") setSpeculativeBoost(c.speculativeBoost);
      }
      if (c.useContinuousBatching) {
        setUseContinuousBatching(true);
        if (typeof c.continuousBatchingMultiplier === "number") setContinuousBatchingMultiplier(c.continuousBatchingMultiplier);
      }
      if (typeof c.reasoningTokens === "number") setReasoningTokens(c.reasoningTokens);
      if (typeof c.cachePrefixTokens === "number") setCachePrefixTokens(c.cachePrefixTokens);
      if (typeof c.cacheHitRate === "number") setCacheHitRate(c.cacheHitRate);
      if (c.cacheTTL) setCacheTTL(c.cacheTTL as CacheTTL);
      if (c.cacheProvider) setCacheProvider(c.cacheProvider as "self-hosted" | "anthropic" | "openai");
    }
    setUrlRestored(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  // ---- Continuous: when Calculator state changes, update URL hash + localStorage ----
  useEffect(() => {
    if (!urlRestored) return; // wait until we've read the initial URL
    const state: CalcTabState = {
      modelId, gpuId, quantization, numGpus, batchSize,
      promptTokens, outputTokens, gpuHourlyCost,
      useSpeculative, speculativeBoost,
      useContinuousBatching, continuousBatchingMultiplier,
      reasoningTokens, cachePrefixTokens, cacheHitRate, cacheTTL, cacheProvider,
    };
    if (activeTab === "calculator") {
      writeUrlHash(serializeCalcState(state));
    }
    saveCalcToStorage(state); // always persist to localStorage
  }, [
    urlRestored, activeTab,
    modelId, gpuId, quantization, numGpus, batchSize, promptTokens, outputTokens,
    gpuHourlyCost, useSpeculative, speculativeBoost, useContinuousBatching,
    continuousBatchingMultiplier, reasoningTokens, cachePrefixTokens, cacheHitRate,
    cacheTTL, cacheProvider,
  ]);

  // ---- Share button handler ----
  const handleShare = async () => {
    // Track: high-signal event — user found something worth sharing
    track("shared_scenario", { tab: activeTab });

    // For Calculator tab, force a fresh write to be safe.
    if (activeTab === "calculator") {
      const state: CalcTabState = {
        modelId, gpuId, quantization, numGpus, batchSize,
        promptTokens, outputTokens, gpuHourlyCost,
        useSpeculative, speculativeBoost,
        useContinuousBatching, continuousBatchingMultiplier,
        reasoningTokens, cachePrefixTokens, cacheHitRate, cacheTTL, cacheProvider,
      };
      writeUrlHash(serializeCalcState(state));
    }
    // MUST stay in the user-gesture call stack for clipboard permission.
    // (setTimeout would break the gesture chain — verified painful.)
    const ok = await copyCurrentUrlToClipboard();
    if (ok) {
      setShareCopied(true);
      toast({
        title: "Share URL copied",
        description: activeTab === "calculator"
          ? "Anyone who opens this link sees the exact same config."
          : activeTab === "build-vs-buy"
            ? "Anyone who opens this link sees the same build-vs-buy scenario."
            : "URL copied.",
      });
      setTimeout(() => setShareCopied(false), 2000);
    } else {
      // Clipboard write failed (e.g., headless browser or no permission).
      // Fallback: select the URL bar via document.execCommand('copy').
      try {
        const urlInput = document.createElement("input");
        urlInput.value = window.location.href;
        document.body.appendChild(urlInput);
        urlInput.select();
        document.execCommand("copy");
        document.body.removeChild(urlInput);
        setShareCopied(true);
        toast({ title: "Share URL copied", description: "Paste anywhere — Discord, Slack, GitHub issue, tweet." });
        setTimeout(() => setShareCopied(false), 2000);
      } catch {
        toast({ title: "Couldn't copy automatically", description: "Copy from the address bar manually." });
      }
    }
  };

  const input: CalcInput = {
    modelId,
    gpuId,
    quantization,
    numGpus,
    batchSize,
    promptTokens,
    outputTokens,
    gpuHourlyCost: gpuHourlyCost === "" ? undefined : Number(gpuHourlyCost),
    useSpeculative,
    speculativeBoost,
    useContinuousBatching,
    continuousBatchingMultiplier,
    reasoningTokens,
    cachePrefixTokens,
    cacheHitRate,
    cacheTTL,
    cacheProvider,
  };

  const result = useMemo(() => calculate(input), [
    modelId, gpuId, quantization, numGpus, batchSize, promptTokens,
    outputTokens, gpuHourlyCost, useSpeculative, speculativeBoost,
    useContinuousBatching, continuousBatchingMultiplier, reasoningTokens,
    cachePrefixTokens, cacheHitRate, cacheTTL, cacheProvider,
  ]);

  // Chart data — vary batch size from 1 to 32
  const chartData = useMemo(() => {
    return [1, 2, 4, 8, 16, 32, 64].map((b) => {
      const r = calculate({ ...input, batchSize: b });
      return {
        batch: `B=${b}`,
        tokens: Math.round(r.aggregateTokensPerSec),
        fits: r.vramFits,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, gpuId, quantization, numGpus, promptTokens, outputTokens, useSpeculative, speculativeBoost, gpuHourlyCost]);

  // Compare GPUs chart — show same model on different GPUs
  const gpuCompareData = useMemo(() => {
    const model = MODELS.find((m) => m.id === modelId)!;
    // only include GPUs that have enough VRAM for the model
    const candidates = GPUS.filter((g) => {
      const modelSizeGb = model.activeParamsB * QUANTIZATIONS.find((q) => q.id === quantization)!.bytesPerParam;
      return g.vramGb * numGpus >= modelSizeGb;
    });
    return candidates.map((g) => {
      const r = calculate({ ...input, gpuId: g.id });
      return {
        gpu: g.name.replace(/\s+\d+GB$/, "").replace(" SXM5", "").replace(" SXM4", ""),
        tokens: Math.round(r.decodeTokensPerSec),
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, quantization, numGpus]);

  // Long-context sweep — show max concurrency across 4K → 1M context
  const longContextData = useMemo(() => {
    const model = MODELS.find((m) => m.id === modelId)!;
    const gpu = GPUS.find((g) => g.id === gpuId)!;
    const quant = QUANTIZATIONS.find((q) => q.id === quantization)!;
    const ctxSizes = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
    return ctxSizes
      .filter(ctx => ctx <= model.maxContext)
      .map(ctx => {
        const maxConcurrent = computeMaxConcurrency(model, gpu, numGpus, ctx, quant.bytesPerParam);
        const prefillMs = computeLongContextPrefillMs(model, result.effectiveFlopsTflops, ctx);
        const kvGb = computeKVCacheGb(model, ctx, 1);
        return {
          context: fmtContext(ctx),
          contextTokens: ctx,
          maxConcurrent,
          prefillMs,
          kvGb,
        };
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, gpuId, numGpus, quantization, result.effectiveFlopsTflops]);

  // Topology recommendation at current context
  const topologyRec = useMemo(() => {
    const model = MODELS.find((m) => m.id === modelId)!;
    const gpu = GPUS.find((g) => g.id === gpuId)!;
    const quant = QUANTIZATIONS.find((q) => q.id === quantization)!;
    const totalContext = promptTokens + (cachePrefixTokens ?? 0);
    return recommendTopology(model, gpu, totalContext, batchSize, quant.bytesPerParam);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, gpuId, numGpus, quantization, promptTokens, cachePrefixTokens, batchSize]);

  const selectedGpu = GPUS.find((g) => g.id === gpuId)!;
  const selectedModel = MODELS.find((m) => m.id === modelId)!;
  const selectedQuant = QUANTIZATIONS.find((q) => q.id === quantization)!;

  return (
    <TooltipProvider delayDuration={200}>
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border/60 backdrop-blur-sm sticky top-0 z-50 bg-background/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="size-8 rounded-md bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center">
              <Gauge className="size-4.5 text-emerald-950" />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight leading-none">
                tokcalc
              </h1>
              <p className="text-[11px] text-muted-foreground mt-0.5 leading-none">
                LLM serving capacity planner
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[11px] gap-1.5 hidden sm:inline-flex">
              <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
              live formulas
            </Badge>
            <a
              href="https://github.com/stevecrates489-commits/tokcalc"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="View source on GitHub"
              className="hidden sm:inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="size-3.5" />
              source
            </a>
            <Button
              variant="outline"
              size="sm"
              onClick={handleShare}
              className="gap-1.5 h-8"
              aria-label="Copy shareable URL"
            >
              {shareCopied ? (
                <>
                  <Check className="size-3.5 text-emerald-500" />
                  <span className="text-xs">Copied</span>
                </>
              ) : (
                <>
                  <Share2 className="size-3.5" />
                  <span className="text-xs hidden sm:inline">Share</span>
                </>
              )}
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
          {/* Hero */}
          <div className="mb-8 sm:mb-10 max-w-4xl">
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight leading-tight">
              Plan your LLM deployment
              <br />
              <span className="text-emerald-500">before you rent the GPUs.</span>
            </h2>
            <p className="mt-3 text-muted-foreground text-sm sm:text-base leading-relaxed">
              An open-source LLM serving capacity planner. Estimate model fit, KV-cache,
              prefill/decode throughput, continuous batching, latency, multi-GPU scaling,
              cloud cost, API cost, and self-hosting break-even — with transparent formulas
              and cited benchmarks.
            </p>
            <div className="mt-4 flex items-start gap-2 text-xs text-muted-foreground bg-muted/30 border border-border/60 rounded-md p-3">
              <HelpCircle className="size-3.5 shrink-0 mt-0.5 text-emerald-500" />
              <div className="leading-relaxed">
                <strong className="text-foreground">Not a GPU expert?</strong> Hover over
                any <Info className="inline size-3 text-emerald-500 align-text-bottom" /> icon
                to see a plain-English explanation. Or scroll to the
                <a href="#glossary" className="text-emerald-500 hover:underline ml-0.5">glossary</a>
                at the bottom for the full list.
              </div>
            </div>
          </div>

          {/* What can tokcalc answer? */}
          <div className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-4 rounded-lg border border-border/60 bg-muted/20">
              <div className="text-[10px] uppercase tracking-wider text-emerald-500 font-semibold mb-2">
                Capacity planning questions tokcalc answers
              </div>
              <ul className="text-xs text-muted-foreground space-y-1.5 leading-relaxed">
                <li>→ Can I serve <strong className="text-foreground">Qwen 2.5 72B</strong> at 128K context on 2× H100 with 20 concurrent users?</li>
                <li>→ How many <strong className="text-foreground">H200s</strong> for 1,000 req/min with P95 TTFT &lt; 2s?</li>
                <li>→ Does <strong className="text-foreground">FP8 or AWQ</strong> save more money once quality + KV cache + engine support are included?</li>
                <li>→ At what daily volume does an H100 beat <strong className="text-foreground">GPT-4o</strong> pricing?</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg border border-border/60 bg-muted/20">
              <div className="text-[10px] uppercase tracking-wider text-emerald-500 font-semibold mb-2">
                What makes tokcalc different
              </div>
              <ul className="text-xs text-muted-foreground space-y-1.5 leading-relaxed">
                <li>✓ <strong className="text-foreground">Transparent formulas</strong> — no black-box throughput assumptions</li>
                <li>✓ <strong className="text-foreground">Engine-aware</strong> — continuous batching, paged KV, prefix caching</li>
                <li>✓ <strong className="text-foreground">Cited benchmarks</strong> — every multiplier is workload-specific, never universal</li>
                <li>✓ <strong className="text-foreground">Provider-neutral pricing</strong> — RunPod / Lambda / Modal / AWS / GCP side-by-side</li>
              </ul>
            </div>
          </div>

          {/* Top-level tab switcher */}
          <div className="mb-6 border-b border-border/60">
            <div className="flex gap-1 -mb-px overflow-x-auto">
              {([
                { id: "calculator", label: "Calculator", icon: Calculator },
                { id: "build-vs-buy", label: "Build vs Buy", icon: DollarSign },
                { id: "reference", label: "Reference", icon: BookOpen },
              ] as const).map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setActiveTab(tab.id);
                      // Track: which tab users switch to (signals which workflow matters most)
                      track("switched_tab", { tab: tab.id });
                    }}
                    className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                      isActive
                        ? "border-emerald-500 text-foreground"
                        : "border-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Icon className={`size-4 ${isActive ? "text-emerald-500" : ""}`} />
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* ===== CALCULATOR TAB ===== */}
          {activeTab === "calculator" && (
            <>
          {/* Calculator grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6">
            {/* ===== INPUTS ===== */}
            <div className="lg:col-span-5 space-y-4">
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Layers className="size-4 text-emerald-500" />
                    1. Model
                    <HelpIcon term="params" />
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Which LLM are you running? Bigger = smarter but slower.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Select value={modelId} onValueChange={setModelId}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Pick a model" />
                    </SelectTrigger>
                    <SelectContent className="max-h-80">
                      {["Llama", "Mistral", "Qwen", "DeepSeek", "Gemma", "Phi", "EleutherAI"].map((fam) => {
                        const items = MODELS.filter((m) => m.family === fam);
                        if (items.length === 0) return null;
                        return (
                          <SelectGroup key={fam}>
                            <SelectLabel className="text-[10px] uppercase tracking-wider text-muted-foreground">
                              {fam}
                            </SelectLabel>
                            {items.map((m) => (
                              <SelectItem key={m.id} value={m.id}>
                                <span className="flex items-center gap-2">
                                  {m.name}
                                  {m.isMoE && (
                                    <Badge variant="secondary" className="text-[9px] px-1 py-0 h-3.5">
                                      MoE
                                    </Badge>
                                  )}
                                </span>
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        );
                      })}
                    </SelectContent>
                  </Select>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    <Stat label="Total params" value={`${selectedModel.paramsB.toLocaleString()}B`} term="params" />
                    <Stat label="Active params" value={`${selectedModel.activeParamsB.toLocaleString()}B`} term="active" />
                    <Stat label="Layers" value={String(selectedModel.layers)} term="layers" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Cpu className="size-4 text-emerald-500" />
                    2. GPU
                    <HelpIcon term="hbmBw" />
                  </CardTitle>
                  <CardDescription className="text-xs">
                    The chip that runs the model. Memory speed matters most for LLMs.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Select value={gpuId} onValueChange={setGpuId}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Pick a GPU" />
                    </SelectTrigger>
                    <SelectContent className="max-h-80">
                      {(["datacenter", "workstation", "consumer", "mac"] as GpuCategory[]).map((cat) => {
                        const items = GPUS.filter((g) => g.category === cat);
                        if (items.length === 0) return null;
                        return (
                          <SelectGroup key={cat}>
                            <SelectLabel className="text-[10px] uppercase tracking-wider text-muted-foreground">
                              {CATEGORY_LABEL[cat]}
                            </SelectLabel>
                            {items.map((g) => (
                              <SelectItem key={g.id} value={g.id}>
                                <span className="flex items-center justify-between w-full">
                                  <span>{g.name}</span>
                                  <span className="text-[10px] text-muted-foreground ml-2">
                                    {g.memBandwidthGbps} GB/s · {g.vramGb}GB
                                  </span>
                                </span>
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        );
                      })}
                    </SelectContent>
                  </Select>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    <Stat label="Memory speed" value={`${selectedGpu.memBandwidthGbps} GB/s`} term="hbmBw" />
                    <Stat label="Math speed" value={`${selectedGpu.flopsTflops} TF`} term="flops" />
                    <Stat label="Memory size" value={`${selectedGpu.vramGb} GB`} term="vram" />
                  </div>

                  {/* Multi-GPU */}
                  <div className="pt-2">
                    <div className="flex items-center justify-between mb-2">
                      <Label className="text-xs flex items-center gap-1">
                        GPUs to use
                        <HelpIcon term="tensorParallel" />
                      </Label>
                      <Badge variant="outline" className="text-[10px] font-mono">{numGpus}×</Badge>
                    </div>
                    <Slider
                      value={[numGpus]}
                      min={1}
                      max={8}
                      step={1}
                      onValueChange={(v) => setNumGpus(v[0])}
                      className="w-full"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <MemoryStick className="size-4 text-emerald-500" />
                    3. Compression
                    <HelpIcon term="quantization" />
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Shrink the model to fit & run faster. Small accuracy cost.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                    {QUANTIZATIONS.map((q) => (
                      <button
                        key={q.id}
                        onClick={() => setQuantization(q.id)}
                        className={`text-xs px-2 py-1.5 rounded-md border transition-colors text-left ${
                          quantization === q.id
                            ? "border-emerald-500 bg-emerald-500/10 text-foreground"
                            : "border-border hover:border-border/80 hover:bg-muted"
                        }`}
                      >
                        <div className="font-medium">{q.label}</div>
                        <div className="text-[10px] text-muted-foreground mt-0.5">
                          {q.bytesPerParam * 8}-bit
                        </div>
                      </button>
                    ))}
                  </div>
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    {selectedQuant.description}
                  </p>
                </CardContent>
              </Card>

              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Database className="size-4 text-emerald-500" />
                    4. Workload
                  </CardTitle>
                  <CardDescription className="text-xs">
                    How many users, and how long are the conversations?
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3.5">
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <Label className="text-xs flex items-center gap-1">
                        Concurrent users
                        <HelpIcon term="batchSize" />
                      </Label>
                      <Badge variant="outline" className="text-[10px] font-mono">{batchSize}</Badge>
                    </div>
                    <Slider
                      value={[batchSize]}
                      min={1}
                      max={64}
                      step={1}
                      onValueChange={(v) => setBatchSize(v[0])}
                      className="w-full"
                    />
                  </div>

                  <div>
                    <Label className="text-xs flex items-center gap-1">
                      Input length (tokens)
                      <HelpIcon term="promptTokens" />
                    </Label>
                    <Input
                      type="number"
                      value={promptTokens}
                      onChange={(e) => setPromptTokens(Math.max(0, Number(e.target.value)))}
                      className="mt-1"
                      min={0}
                    />
                  </div>

                  <div>
                    <Label className="text-xs flex items-center gap-1">
                      Answer length (tokens)
                      <HelpIcon term="outputTokens" />
                    </Label>
                    <Input
                      type="number"
                      value={outputTokens}
                      onChange={(e) => setOutputTokens(Math.max(0, Number(e.target.value)))}
                      className="mt-1"
                      min={0}
                    />
                  </div>

                  <div>
                    <Label className="text-xs flex items-center gap-1">
                      GPU price ($/hour)
                      <span className="text-[10px] text-muted-foreground font-normal ml-1">
                        (blank = use default ${selectedGpu.usdPerHour}/hr)
                      </span>
                    </Label>
                    <Input
                      type="number"
                      step="0.01"
                      placeholder={`default $${selectedGpu.usdPerHour}`}
                      value={gpuHourlyCost}
                      onChange={(e) =>
                        setGpuHourlyCost(e.target.value === "" ? "" : Number(e.target.value))
                      }
                      className="mt-1"
                    />
                  </div>

                  <Separator />

                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-xs flex items-center gap-1">
                        Speculative decoding
                        <HelpIcon term="speculative" />
                      </Label>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Use a small model to predict ahead → big speedup
                      </p>
                    </div>
                    <Switch checked={useSpeculative} onCheckedChange={(v) => { setUseSpeculative(v); track("toggled_feature", { feature: "speculative_decoding", enabled: v }); }} />
                  </div>
                  {useSpeculative && (
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <Label className="text-xs">Speed boost factor</Label>
                        <Badge variant="outline" className="text-[10px] font-mono">
                          {speculativeBoost.toFixed(1)}×
                        </Badge>
                      </div>
                      <Slider
                        value={[speculativeBoost]}
                        min={1.2}
                        max={4}
                        step={0.1}
                        onValueChange={(v) => setSpeculativeBoost(v[0])}
                        className="w-full"
                      />
                    </div>
                  )}

                  <Separator />

                  {/* === Phase 2: Continuous batching === */}
                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-xs flex items-center gap-1">
                        Continuous batching
                        <HelpIcon term="continuousBatching" />
                      </Label>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        vLLM/TGI-style; serves requests mid-flight (1.5–4× typical)
                      </p>
                    </div>
                    <Switch checked={useContinuousBatching} onCheckedChange={(v) => { setUseContinuousBatching(v); track("toggled_feature", { feature: "continuous_batching", enabled: v }); }} />
                  </div>
                  {useContinuousBatching && (
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <Label className="text-xs">Throughput multiplier</Label>
                        <Badge variant="outline" className="text-[10px] font-mono">
                          {continuousBatchingMultiplier.toFixed(1)}×
                        </Badge>
                      </div>
                      <Slider
                        value={[continuousBatchingMultiplier]}
                        min={1.0}
                        max={4}
                        step={0.1}
                        onValueChange={(v) => setContinuousBatchingMultiplier(v[0])}
                        className="w-full"
                      />
                      <p className="text-[10px] text-muted-foreground mt-1.5 leading-relaxed">
                        Cited range 1.5–4× is workload-specific (SOSP paper).
                        Conservative default: 1.5×.
                      </p>
                    </div>
                  )}

                  {/* === Phase 2: Reasoning tokens === */}
                  <div>
                    <Label className="text-xs flex items-center gap-1">
                      Hidden reasoning tokens
                      <HelpIcon term="reasoningTokens" />
                    </Label>
                    <Input
                      type="number"
                      value={reasoningTokens}
                      onChange={(e) => setReasoningTokens(Math.max(0, Number(e.target.value)))}
                      className="mt-1"
                      min={0}
                      placeholder="0 — only for o1/R1/Claude-thinking"
                    />
                    {reasoningTokens > 0 && (
                      <p className="text-[10px] text-amber-600 dark:text-amber-400 mt-1.5 leading-relaxed">
                        Billed output: {outputTokens + reasoningTokens} tok = {outputTokens} visible + {reasoningTokens} reasoning
                      </p>
                    )}
                  </div>

                  <Separator />

                  {/* === Phase 2: Prompt caching === */}
                  <div className="flex items-center justify-between">
                    <div>
                      <Label className="text-xs flex items-center gap-1">
                        Prompt caching
                        <HelpIcon term="promptCaching" />
                      </Label>
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Reuse a shared prefix (system prompt, RAG docs)
                      </p>
                    </div>
                    <Switch
                      checked={cachePrefixTokens > 0}
                      onCheckedChange={(checked) => {
                        setCachePrefixTokens(checked ? 1000 : 0);
                        track("toggled_feature", { feature: "prompt_caching", enabled: checked });
                      }}
                    />
                  </div>
                  {cachePrefixTokens > 0 && (
                    <div className="space-y-3 p-3 rounded-md bg-muted/30 border border-border/40">
                      <div>
                        <Label className="text-xs">Cache prefix length (tokens)</Label>
                        <Input
                          type="number"
                          value={cachePrefixTokens}
                          onChange={(e) => setCachePrefixTokens(Math.max(0, Number(e.target.value)))}
                          className="mt-1"
                          min={0}
                        />
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <Label className="text-xs">Cache hit rate</Label>
                          <Badge variant="outline" className="text-[10px] font-mono">
                            {(cacheHitRate * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        <Slider
                          value={[cacheHitRate * 100]}
                          min={0}
                          max={100}
                          step={5}
                          onValueChange={(v) => setCacheHitRate(v[0] / 100)}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <Label className="text-xs">Cache provider</Label>
                        <Select value={cacheProvider} onValueChange={(v) => setCacheProvider(v as "self-hosted" | "anthropic" | "openai")}>
                          <SelectTrigger className="w-full mt-1">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="self-hosted">Self-hosted (vLLM APC)</SelectItem>
                            <SelectItem value="anthropic">Anthropic (5m/1h TTL)</SelectItem>
                            <SelectItem value="openai">OpenAI (50% off cached)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      {cacheProvider === "anthropic" && (
                        <div>
                          <Label className="text-xs">Cache TTL</Label>
                          <Select value={cacheTTL} onValueChange={(v) => setCacheTTL(v as CacheTTL)}>
                            <SelectTrigger className="w-full mt-1">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="5m">5 min (1.25× write)</SelectItem>
                              <SelectItem value="1h">1 hour (2× write)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* ===== RESULTS ===== */}
            <div className="lg:col-span-7 space-y-4">
              {/* Headline numbers */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <HeadlineCard
                  icon={<Gauge className="size-4" />}
                  label="Generation speed"
                  value={fmtTokens(result.decodeTokensPerSec)}
                  sub="tok/s — 1 user"
                  highlight
                  glossaryKey="decode"
                  confidence={result.confidence.decodeTokensPerSec}
                />
                <HeadlineCard
                  icon={<Zap className="size-4" />}
                  label="Total throughput"
                  value={fmtTokens(result.aggregateTokensPerSec)}
                  sub={`${batchSize} user${batchSize > 1 ? "s" : ""}${useContinuousBatching ? ` × ${continuousBatchingMultiplier.toFixed(1)}` : ""}`}
                  glossaryKey="aggregate"
                  confidence={result.confidence.aggregateTokensPerSec}
                />
                <HeadlineCard
                  icon={<Clock className="size-4" />}
                  label="Wait per request"
                  value={fmtMs(result.totalLatencyMs)}
                  sub={`${result.billedOutputTokens}-tok answer`}
                  glossaryKey="latency"
                  confidence={result.confidence.totalLatencyMs}
                />
                <HeadlineCard
                  icon={<Clock className="size-4" />}
                  label="Time to first token"
                  value={fmtMs(result.ttftMs)}
                  sub={`+ ${fmtMs(result.itlMs)} per token`}
                  glossaryKey="ttft"
                  confidence={result.confidence.ttftMs}
                />
              </div>

              {/* Long-context warning */}
              {result.longContextWarning && (
                <div className="flex items-start gap-3 p-3 rounded-lg border border-blue-500/30 bg-blue-500/5">
                  <Info className="size-4 text-blue-500 shrink-0 mt-0.5" />
                  <div className="text-xs">
                    <p className="font-medium text-blue-600 dark:text-blue-400">
                      Long-context caveat
                    </p>
                    <p className="text-muted-foreground mt-0.5 leading-relaxed">
                      {result.longContextWarning}
                    </p>
                  </div>
                </div>
              )}

              {/* Reasoning tokens callout */}
              {reasoningTokens > 0 && (
                <div className="flex items-start gap-3 p-3 rounded-lg border border-amber-500/30 bg-amber-500/5">
                  <AlertTriangle className="size-4 text-amber-500 shrink-0 mt-0.5" />
                  <div className="text-xs">
                    <p className="font-medium text-amber-600 dark:text-amber-400">
                      Reasoning model workload
                    </p>
                    <p className="text-muted-foreground mt-0.5">
                      Billed output: <strong className="text-foreground">{result.billedOutputTokens} tokens</strong> =
                      {outputTokens} visible + {reasoningTokens} hidden reasoning. Real latency/cost is
                      <strong className="text-amber-600 dark:text-amber-400"> {((result.billedOutputTokens / outputTokens - 1) * 100).toFixed(0)}% higher</strong> than visible answer alone.
                    </p>
                  </div>
                </div>
              )}

              {/* Cache savings callout */}
              {cachePrefixTokens > 0 && result.cacheSavingsPct > 0 && (
                <div className="flex items-start gap-3 p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5">
                  <Zap className="size-4 text-emerald-500 shrink-0 mt-0.5" />
                  <div className="text-xs">
                    <p className="font-medium text-emerald-600 dark:text-emerald-400">
                      Prompt cache savings
                    </p>
                    <p className="text-muted-foreground mt-0.5">
                      {cacheProvider === "self-hosted" && "Self-hosted (vLLM APC): "}
                      {cacheProvider === "anthropic" && "Anthropic: "}
                      {cacheProvider === "openai" && "OpenAI: "}
                      Saves <strong className="text-foreground">{result.prefillTokensAvoided.toLocaleString()} tokens</strong> of prefill on{" "}
                      {(cacheHitRate * 100).toFixed(0)}% of requests =
                      <strong className="text-emerald-600 dark:text-emerald-400"> {result.cacheSavingsPct.toFixed(0)}% prefill reduction</strong>
                      {cacheProvider === "anthropic" && ` (cache read at ${result.cacheReadCostMultiplier}× = 90% off cached tokens)`}
                      {cacheProvider === "openai" && ` (cached tokens at ${result.cacheReadCostMultiplier}× = 50% off)`}
                    </p>
                  </div>
                </div>
              )}

              {/* VRAM warning */}
              {!result.vramFits && (
                <div className="flex items-start gap-3 p-3 rounded-lg border border-amber-500/30 bg-amber-500/5">
                  <AlertTriangle className="size-4 text-amber-500 shrink-0 mt-0.5" />
                  <div className="text-xs">
                    <p className="font-medium text-amber-600 dark:text-amber-400">
                      Model doesn&apos;t fit in VRAM
                    </p>
                    <p className="text-muted-foreground mt-0.5">
                      Needs <strong>{fmtBytes(result.totalVramNeededGb)}</strong> but only{" "}
                      <strong>{selectedGpu.vramGb * numGpus} GB</strong> available. Reduce
                      quantization, use more GPUs, or shorter context.
                    </p>
                  </div>
                </div>
              )}

              {/* Memory & Compute breakdown */}
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    Memory budget
                    <HelpIcon term="vram" />
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Will the model + conversation fit in the GPU&apos;s memory?
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <Metric
                      label="Model size"
                      value={fmtBytes(result.modelSizeGb)}
                      hint={`${selectedModel.activeParamsB}B × ${selectedQuant.bytesPerParam} bytes`}
                      glossaryKey="modelWeights"
                      confidence={result.confidence.modelSizeGb}
                    />
                    <Metric
                      label="Conversation memory"
                      value={fmtBytes(result.kvCacheTotalGb)}
                      hint={`${fmtBytes(result.kvCachePerTokenKb / 1024)}/tok × ${promptTokens} tok × ${batchSize}`}
                      glossaryKey="kvCache"
                      confidence={result.confidence.kvCacheTotalGb}
                    />
                    <Metric
                      label="Total memory needed"
                      value={fmtBytes(result.totalVramNeededGb)}
                      hint={result.vramFits ? "✓ fits in GPU memory" : "✗ over budget"}
                      warn={!result.vramFits}
                    />
                    <Metric
                      label="Available memory"
                      value={`${selectedGpu.vramGb * numGpus} GB`}
                      hint={`${selectedGpu.vramGb} GB × ${numGpus} GPU`}
                    />
                  </div>

                  {/* VRAM usage bar */}
                  <div className="pt-1">
                    <div className="flex items-center justify-between text-[11px] mb-1.5">
                      <span className="text-muted-foreground">Memory used</span>
                      <span className="font-mono">
                        {((result.totalVramNeededGb / (selectedGpu.vramGb * numGpus)) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          result.vramFits ? "bg-emerald-500" : "bg-amber-500"
                        }`}
                        style={{
                          width: `${Math.min(100, (result.totalVramNeededGb / (selectedGpu.vramGb * numGpus)) * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Throughput detail */}
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Speed breakdown</CardTitle>
                  <CardDescription className="text-xs">
                    Throughput & latency detail
                  </CardDescription>
                </CardHeader>
                <CardContent className="grid grid-cols-2 gap-3 text-sm">
                  <Metric
                    label="Writing speed (1 user)"
                    value={`${fmtTokens(result.decodeTokensPerSec)} tok/s`}
                    hint={`${fmtMs(result.decodeTimePerTokenMs)} per token`}
                    glossaryKey="decode"
                    confidence={result.confidence.decodeTokensPerSec}
                  />
                  <Metric
                    label="Total speed (batched)"
                    value={`${fmtTokens(result.aggregateTokensPerSec)} tok/s`}
                    hint={`batch ${batchSize}${useContinuousBatching ? ` × ${continuousBatchingMultiplier.toFixed(1)}×` : ""}`}
                    glossaryKey="aggregate"
                    confidence={result.confidence.aggregateTokensPerSec}
                  />
                  <Metric
                    label="Reading speed (prefill)"
                    value={`${fmtTokens(result.prefillTokensPerSec)} tok/s`}
                    hint={cachePrefixTokens > 0 && result.cacheSavingsPct > 0
                      ? `prompt in ${fmtMs(result.prefillTimeWithCacheMs)} (${result.cacheSavingsPct.toFixed(0)}% cached)`
                      : `prompt in ${fmtMs(result.prefillTimeMs)}`}
                    glossaryKey="prefill"
                    confidence={result.confidence.prefillTokensPerSec}
                  />
                  <Metric
                    label="Batch sweet spot"
                    value={result.batchCrossover.toFixed(0)}
                    hint="users where adding more stops helping"
                    glossaryKey="batchCrossover"
                  />
                  {reasoningTokens > 0 && (
                    <Metric
                      label="Billed output"
                      value={`${result.billedOutputTokens} tok`}
                      hint={`${outputTokens} visible + ${reasoningTokens} hidden reasoning`}
                      warn
                    />
                  )}
                  {useSpeculative && (
                    <Metric
                      label="Speculative boost"
                      value={`${speculativeBoost.toFixed(1)}×`}
                      hint="assumes accurate draft model"
                      glossaryKey="speculative"
                    />
                  )}
                </CardContent>
              </Card>

              {/* Cost */}
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <DollarSign className="size-4 text-emerald-500" />
                    Money
                  </CardTitle>
                  <CardDescription className="text-xs">
                    How much does it cost to run this?
                  </CardDescription>
                </CardHeader>
                <CardContent className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                  <Metric
                    label="GPU rent cost"
                    value={fmtMoney(result.costPerHour) + "/hr"}
                    hint={numGpus > 1 ? `${selectedGpu.usdPerHour} × ${numGpus} GPUs` : "per GPU"}
                    confidence={result.confidence.costPerHour}
                  />
                  <Metric
                    label="$ per 1M tokens"
                    value={fmtMoney(result.costPerMillionOutputTokens)}
                    hint="standard LLM pricing unit"
                    glossaryKey="costPerMTokens"
                    confidence={result.confidence.costPerMillionOutputTokens}
                  />
                  <Metric
                    label="$ per request"
                    value={fmtMoney(result.costPerRequest)}
                    hint={`${outputTokens}-token answer`}
                  />
                </CardContent>
              </Card>

              {/* Batch scaling chart */}
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">How throughput grows with more users</CardTitle>
                  <CardDescription className="text-xs">
                    More users = more total speed — until the GPU hits its math limit.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[220px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
                        <XAxis
                          dataKey="batch"
                          stroke="var(--muted-foreground)"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          stroke="var(--muted-foreground)"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
                        />
                        <Tooltip
                          cursor={{ fill: "var(--muted)", opacity: 0.3 }}
                          contentStyle={{
                            backgroundColor: "var(--background)",
                            border: "1px solid var(--border)",
                            borderRadius: "6px",
                            fontSize: "11px",
                          }}
                          formatter={(v: number) => [`${v.toLocaleString()} tok/s`, "aggregate"]}
                        />
                        <Bar dataKey="tokens" radius={[3, 3, 0, 0]}>
                          {chartData.map((d, i) => (
                            <Cell
                              key={i}
                              fill={d.batch === `B=${batchSize}` ? "#10b981" : "var(--muted-foreground)"}
                              opacity={d.fits ? 1 : 0.3}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* GPU comparison chart */}
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    Compare GPUs — {selectedModel.name} ({selectedQuant.label})
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Speed of the same model on every GPU big enough to fit it. Green = your current pick.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[260px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={gpuCompareData}
                        layout="vertical"
                        margin={{ top: 0, right: 16, bottom: 0, left: 24 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} horizontal={false} />
                        <XAxis
                          type="number"
                          stroke="var(--muted-foreground)"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
                        />
                        <YAxis
                          type="category"
                          dataKey="gpu"
                          stroke="var(--muted-foreground)"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                          width={84}
                        />
                        <Tooltip
                          cursor={{ fill: "var(--muted)", opacity: 0.3 }}
                          contentStyle={{
                            backgroundColor: "var(--background)",
                            border: "1px solid var(--border)",
                            borderRadius: "6px",
                            fontSize: "11px",
                          }}
                          formatter={(v: number) => [`${v.toLocaleString()} tok/s`, "decode"]}
                        />
                        <Bar dataKey="tokens" radius={[0, 3, 3, 0]}>
                          {gpuCompareData.map((d, i) => (
                            <Cell
                              key={i}
                              fill={d.gpu.includes(selectedGpu.name.split(" ")[0]) ? "#10b981" : "#475569"}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* ===== Long-Context Capacity Planner ===== */}
              <Card className="border-border/60 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    Long-context capacity
                    <HelpIcon term="kvCache" />
                  </CardTitle>
                  <CardDescription className="text-xs">
                    How many concurrent users fit as context grows? KV cache scales linearly with context — at 128K+ you&apos;ll likely need multiple GPUs.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Topology recommendation callout */}
                  <div className={`flex items-start gap-3 p-3 rounded-md border ${
                    topologyRec.fits
                      ? "border-emerald-500/40 bg-emerald-500/5"
                      : "border-amber-500/40 bg-amber-500/5"
                  }`}>
                    <Cpu className={`size-4 shrink-0 mt-0.5 ${topologyRec.fits ? "text-emerald-500" : "text-amber-500"}`} />
                    <div className="text-xs">
                      <div className="font-medium text-foreground">
                        Recommended topology: <span className={topologyRec.fits ? "text-emerald-500" : "text-amber-500"}>{topologyRec.topology}</span>
                      </div>
                      <div className="text-muted-foreground mt-0.5 leading-relaxed">{topologyRec.reason}</div>
                      {topologyRec.hasContextParallel && (
                        <div className="text-[10px] text-amber-600 dark:text-amber-400 mt-1">
                          ⚠ Context Parallel / RingAttention is advanced — only vLLM/SGLang research branches support it today.
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Context sweep chart — max concurrent users at each context size */}
                  <div className="pt-1">
                    <div className="flex items-center justify-between text-[11px] mb-1.5">
                      <span className="text-muted-foreground">Max concurrent users at each context length</span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {selectedGpu.vramGb * numGpus} GB total VRAM
                      </span>
                    </div>
                    <div className="h-[180px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={longContextData} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.4} />
                          <XAxis
                            dataKey="context"
                            stroke="var(--muted-foreground)"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                          />
                          <YAxis
                            stroke="var(--muted-foreground)"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
                          />
                          <Tooltip
                            cursor={{ fill: "var(--muted)", opacity: 0.3 }}
                            contentStyle={{
                              backgroundColor: "var(--background)",
                              border: "1px solid var(--border)",
                              borderRadius: "6px",
                              fontSize: "11px",
                            }}
                            formatter={(_v: number, _n: string, props: { payload: { maxConcurrent: number; prefillMs: number; kvGb: number; context: string } }) => {
                              const p = props.payload;
                              return [
                                `${p.maxConcurrent.toLocaleString()} users · prefill ${fmtMs(p.prefillMs)} · KV ${fmtBytes(p.kvGb)}`,
                                p.context,
                              ];
                            }}
                          />
                          <Bar dataKey="maxConcurrent" radius={[3, 3, 0, 0]}>
                            {longContextData.map((d, i) => {
                              // Highlight current context (closest match by tokens)
                              const totalContext = promptTokens + (cachePrefixTokens ?? 0);
                              const isCurrent = d.contextTokens >= totalContext &&
                                (i === 0 || longContextData[i - 1].contextTokens < totalContext);
                              return (
                                <Cell
                                  key={i}
                                  fill={isCurrent ? "#10b981" : d.maxConcurrent === 0 ? "#ef4444" : "#475569"}
                                  opacity={d.maxConcurrent === 0 ? 0.3 : 1}
                                />
                              );
                            })}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-1.5 leading-relaxed">
                      Bars show how many concurrent users fit in VRAM at each context length.
                      <strong className="text-foreground"> Green</strong> = closest to your current context ({fmtContext(promptTokens + (cachePrefixTokens ?? 0))}).
                      <strong className="text-foreground"> Red/dim</strong> = doesn&apos;t fit even at batch=1.
                    </div>
                  </div>

                  {/* Detailed metrics table */}
                  <div className="pt-1 overflow-x-auto">
                    <table className="w-full text-[11px]">
                      <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
                        <tr>
                          <th className="text-left py-1 pr-3">Context</th>
                          <th className="text-right py-1 pr-3">Max users</th>
                          <th className="text-right py-1 pr-3">KV/request</th>
                          <th className="text-right py-1 pr-3">Prefill time</th>
                          <th className="text-right py-1">Fits?</th>
                        </tr>
                      </thead>
                      <tbody>
                        {longContextData.map(d => {
                          const totalContext = promptTokens + (cachePrefixTokens ?? 0);
                          const isCurrent = d.contextTokens >= totalContext &&
                            (longContextData.indexOf(d) === 0 || longContextData[longContextData.indexOf(d) - 1].contextTokens < totalContext);
                          return (
                            <tr key={d.context} className={`border-t border-border/40 ${isCurrent ? "bg-emerald-500/5" : ""}`}>
                              <td className="py-1.5 pr-3 font-medium">{d.context}</td>
                              <td className="py-1.5 pr-3 text-right font-mono">
                                {d.maxConcurrent > 0 ? d.maxConcurrent.toLocaleString() : <span className="text-red-500">0 (no fit)</span>}
                              </td>
                              <td className="py-1.5 pr-3 text-right font-mono">{fmtBytes(d.kvGb)}</td>
                              <td className="py-1.5 pr-3 text-right font-mono">{fmtMs(d.prefillMs)}</td>
                              <td className="py-1.5 text-right">
                                {d.maxConcurrent > 0
                                  ? <span className="text-emerald-500">✓</span>
                                  : <span className="text-red-500">✗</span>}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div className="text-[10px] text-muted-foreground leading-relaxed pt-1 border-t border-border/40">
                    <strong className="text-foreground">Math:</strong> KV/request = 2 × layers × kv_heads × head_dim × bytes/KV × context.
                    Prefill includes <strong className="text-foreground">superlinear attention cost</strong> at long context (O(N²) attention dominates beyond ~32K).
                    For RingAttention/blockwise attention, real prefill is lower than this estimate.
                    Refs: <a href="https://arxiv.org/abs/2310.01889" target="_blank" rel="noopener noreferrer" className="text-emerald-500 hover:underline">RingAttention paper</a>.
                  </div>
                </CardContent>
              </Card>

              {/* ===== Benchmark Import — calibrate the formula against real data ===== */}
              <BenchmarkImport
                estimate={result}
                modelName={selectedModel.name}
                gpuName={selectedGpu.name}
              />
            </div>
          </div>
            </>
          )}

          {/* ===== BUILD-VS-BUY TAB ===== */}
          {activeTab === "build-vs-buy" && (
            <BuildVsBuyTab />
          )}

          {/* ===== REFERENCE TAB ===== */}
          {activeTab === "reference" && (
            <ReferenceTab />
          )}

          {/* Plain-English glossary — visible on all tabs */}
          <div id="glossary" className="mt-10 scroll-mt-20">
            <Card className="border-border/60 shadow-sm">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BookOpen className="size-4 text-emerald-500" />
                  Plain-English glossary
                </CardTitle>
                <CardDescription className="text-sm">
                  What all these words mean, without jargon.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-3">
                  {Object.entries(GLOSSARY).map(([key, text]) => (
                    <div key={key} className="space-y-0.5">
                      <div className="text-xs font-semibold text-emerald-500 capitalize">
                        {key.replace(/([A-Z])/g, " $1").replace(/^./, (c) => c.toUpperCase())}
                      </div>
                      <div className="text-[11px] text-muted-foreground leading-relaxed">
                        {text}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Confidence legend */}
          <div className="mt-10">
            <ConfidenceLegend />
            <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed max-w-3xl">
              Every metric on this page has a confidence dot (🟢 measured · 🟢 modeled · 🟡 inferred · ⚪ user-supplied).
              <strong className="text-foreground"> Honesty is the moat</strong> — we&apos;d rather show you our uncertainty than
              pretend at a universal &quot;tokens/sec&quot; number. Real-world performance depends on engine, model revision,
              driver, and traffic distribution.
            </p>
          </div>

          {/* Formula reference */}
          <div className="mt-10">
            <Card className="border-border/60 shadow-sm">
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Calculator className="size-4 text-emerald-500" />
                  The math, transparently
                </CardTitle>
                <CardDescription className="text-sm">
                  Every number above comes from these formulas. No black boxes, no hidden
                  &quot;magic numbers.&quot; <span className="text-muted-foreground">Optional reading for engineers.</span>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="decode">
                    <AccordionTrigger className="text-sm">
                      Decode tokens/sec (memory-bound)
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2">
                      <FormulaBox>
                        decode_tokens/sec ≈ (HBM_BW × η_mem × quant_eff) / model_size
                      </FormulaBox>
                      <p className="text-muted-foreground">
                        Where <code className="text-emerald-500">η_mem = 0.65</code> (typical
                        real-world memory utilization), <code className="text-emerald-500">quant_eff</code> accounts
                        for INT4/INT8 dequant overhead, and <code className="text-emerald-500">model_size = params × bytes_per_param</code>.
                        For MoE, only <em>active</em> params are loaded per token.
                      </p>
                      <p className="text-muted-foreground">
                        Your numbers: <code>{selectedGpu.memBandwidthGbps}</code> GB/s × 0.65 × {result.quantEfficiency} ={" "}
                        <strong className="text-foreground">{result.effectiveBandwidthGbps.toFixed(0)} GB/s effective</strong>,
                        divided by model size <strong className="text-foreground">{result.modelSizeGb.toFixed(2)} GB</strong> ={" "}
                        <strong className="text-emerald-500">{fmtTokens(result.decodeTokensPerSec)} tok/s</strong>.
                      </p>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="prefill">
                    <AccordionTrigger className="text-sm">
                      Prefill tokens/sec (compute-bound)
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2">
                      <FormulaBox>
                        prefill_tokens/sec ≈ (GPU_FLOPS × η_compute) / (2 × active_params)
                      </FormulaBox>
                      <p className="text-muted-foreground">
                        Prefill is FLOPS-bound because the prompt tokens are processed in parallel.
                        Each token requires roughly <code className="text-emerald-500">2 × N</code> FLOPs (one
                        multiply + one add per parameter).
                      </p>
                      <p className="text-muted-foreground">
                        Your numbers: <code>{selectedGpu.flopsTflops * numGpus}</code> TFLOPS × 0.5 ={" "}
                        <strong className="text-foreground">{(result.effectiveFlopsTflops).toFixed(0)} TF effective</strong>,
                        divided by 2 × {selectedModel.activeParamsB}B params ={" "}
                        <strong className="text-emerald-500">{fmtTokens(result.prefillTokensPerSec)} tok/s</strong>.
                      </p>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="batch">
                    <AccordionTrigger className="text-sm">
                      Batch crossover (memory → compute bound)
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2">
                      <FormulaBox>
                        B_crossover ≈ (bytes_per_param × FLOPS × η_compute) / (2 × HBM_BW × η_mem × quant_eff)
                      </FormulaBox>
                      <p className="text-muted-foreground">
                        Below this batch size, throughput scales linearly with batch (memory-bound). Above it,
                        you become compute-bound and adding more batches doesn&apos;t help.
                      </p>
                      <p className="text-muted-foreground">
                        Your crossover point: <strong className="text-emerald-500">{result.batchCrossover.toFixed(1)}</strong> —
                        your current batch ({batchSize}) is{" "}
                        {batchSize < result.batchCrossover ? (
                          <span className="text-emerald-500">memory-bound (good)</span>
                        ) : (
                          <span className="text-amber-500">compute-bound</span>
                        )}.
                      </p>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="kv">
                    <AccordionTrigger className="text-sm">
                      KV cache size
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2">
                      <FormulaBox>
                        KV = 2 × layers × seq_len × kv_heads × head_dim × batch × dtype_bytes
                      </FormulaBox>
                      <p className="text-muted-foreground">
                        KV cache grows linearly with sequence length and batch size. With GQA
                        (grouped-query attention), only <code>kv_heads</code> caches are stored, not
                        all query heads.
                      </p>
                      <p className="text-muted-foreground">
                        Your numbers: 2 × {selectedModel.layers} × {promptTokens} × {selectedModel.kvHeads} ×{" "}
                        {selectedModel.headDim} × {batchSize} × 2 bytes ={" "}
                        <strong className="text-emerald-500">{fmtBytes(result.kvCacheTotalGb)}</strong>.
                      </p>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="latency">
                    <AccordionTrigger className="text-sm">
                      End-to-end latency per request
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2">
                      <FormulaBox>
                        total = prefill_time + output_tokens × decode_time + overhead
                      </FormulaBox>
                      <p className="text-muted-foreground">
                        Your breakdown: prefill = <strong>{fmtMs(result.prefillTimeMs)}</strong>,
                        decode = <strong>{fmtMs(result.totalDecodeTimeMs)}</strong> ({outputTokens} × {fmtMs(result.decodeTimePerTokenMs)}),
                        overhead ≈ 30ms → total{" "}
                        <strong className="text-emerald-500">{fmtMs(result.totalLatencyMs)}</strong>.
                      </p>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="cost">
                    <AccordionTrigger className="text-sm">
                      Cost per million output tokens
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2">
                      <FormulaBox>
                        $/M = (gpu_$/hr × num_gpus / 3600) / aggregate_tokens_per_sec × 1,000,000
                      </FormulaBox>
                      <p className="text-muted-foreground">
                        Your numbers: {fmtMoney(result.costPerHour)}/hr ÷ {fmtTokens(result.aggregateTokensPerSec)} tok/s × 1M ={" "}
                        <strong className="text-emerald-500">{fmtMoney(result.costPerMillionOutputTokens)}/M tokens</strong>.
                      </p>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="disclaimer">
                    <AccordionTrigger className="text-sm">
                      Caveats & real-world accuracy
                    </AccordionTrigger>
                    <AccordionContent className="text-xs space-y-2 text-muted-foreground">
                      <p>
                        These formulas give theoretical maxima. Real-world throughput is typically
                        <strong className="text-foreground"> 70–90% of the decode number</strong> due to:
                      </p>
                      <ul className="list-disc list-inside space-y-1 ml-2">
                        <li>Attention cost (grows quadratically with context length beyond ~4K)</li>
                        <li>Kernel launch overhead and Python GIL contention</li>
                        <li>Sampling / tokenization / detokenization</li>
                        <li>Sampling with rejection (in nucleus sampling)</li>
                        <li>Network I/O for multi-node or API-serving</li>
                        <li>Continuous batching in modern servers (vLLM, TGI) — can <em>exceed</em> these numbers</li>
                      </ul>
                      <p className="flex items-start gap-2 mt-2 pt-2 border-t border-border/60">
                        <Info className="size-3.5 shrink-0 mt-0.5 text-emerald-500" />
                        <span>
                          For benchmark-grade numbers, run <code>vllm --model &lt;model&gt; --gpu &lt;gpu&gt;</code>
                          {" "}and use <code>--max-num-batched-tokens</code> to find the sweet spot.
                          This tool is for first-pass planning.
                        </span>
                      </p>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CardContent>
            </Card>
          </div>

          {/* Tech badges */}
          <div className="mt-8 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            <CheckCircle2 className="size-3.5 text-emerald-500" />
            <span>Formulas verified against:</span>
            {["vLLM docs", "NVIDIA cuBLAS benchmarks", "llama.cpp discussions", "GPU spec sheets", "artificialanalysis.ai"].map((s) => (
              <Badge key={s} variant="outline" className="text-[10px] font-normal">
                {s}
              </Badge>
            ))}
          </div>
        </div>
      </main>

      {/* Footer (sticky) */}
      <footer className="mt-auto border-t border-border/60 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span className="font-medium">tokcalc</span>
            <span>·</span>
            <span>open-source LLM throughput estimator</span>
          </div>
          <div className="flex items-center gap-3">
            <span>built with Next.js 16 · Tailwind · Recharts</span>
            <span>·</span>
            <span>not affiliated with any GPU vendor</span>
          </div>
        </div>
      </footer>
    </div>
    </TooltipProvider>
  );
}

/* ---------- small subcomponents ---------- */

function HelpIcon({ term }: { term: GlossaryKey }) {
  return (
    <UITooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`What does ${term} mean?`}
          className="inline-flex items-center justify-center text-muted-foreground hover:text-emerald-500 transition-colors"
        >
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        className="max-w-[260px] text-xs leading-relaxed"
      >
        {GLOSSARY[term]}
      </TooltipContent>
    </UITooltip>
  );
}

function Stat({ label, value, term }: { label: string; value: string; term?: GlossaryKey }) {
  return (
    <div className="flex flex-col gap-0.5 px-2 py-1.5 rounded-md bg-muted/40">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1">
        {label}
        {term && <HelpIcon term={term} />}
      </span>
      <span className="font-mono text-xs">{value}</span>
    </div>
  );
}

function HeadlineCard({
  icon,
  label,
  value,
  sub,
  highlight,
  glossaryKey,
  confidence,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  highlight?: boolean;
  glossaryKey?: GlossaryKey;
  confidence?: Confidence;
}) {
  return (
    <Card
      className={`border-border/60 shadow-sm overflow-hidden ${
        highlight ? "border-emerald-500/40 bg-emerald-500/5" : ""
      }`}
    >
      <CardContent className="p-3 sm:p-4">
        <div className="flex items-center gap-1.5 text-muted-foreground mb-1.5">
          <span className={highlight ? "text-emerald-500" : ""}>{icon}</span>
          <span className="text-[11px] font-medium">{label}</span>
          {glossaryKey && <HelpIcon term={glossaryKey} />}
          {confidence && <ConfidenceDot confidence={confidence} />}
        </div>
        <div
          className={`text-2xl sm:text-3xl font-bold font-mono tabular-nums ${
            highlight ? "text-emerald-500" : ""
          }`}
        >
          {value}
        </div>
        {sub && (
          <div className="text-[10px] text-muted-foreground mt-0.5">{sub}</div>
        )}
      </CardContent>
    </Card>
  );
}

function Metric({
  label,
  value,
  hint,
  warn,
  glossaryKey,
  confidence,
}: {
  label: string;
  value: string;
  hint?: string;
  warn?: boolean;
  glossaryKey?: GlossaryKey;
  confidence?: Confidence;
}) {
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1">
        {label}
        {glossaryKey && <HelpIcon term={glossaryKey} />}
        {confidence && <ConfidenceBadge confidence={confidence} />}
      </div>
      <div className={`font-mono text-sm font-medium ${warn ? "text-amber-500" : ""}`}>{value}</div>
      {hint && <div className="text-[10px] text-muted-foreground">{hint}</div>}
    </div>
  );
}

function FormulaBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-muted/40 border border-border/60 rounded-md px-3 py-2 font-mono text-[11px] text-foreground">
      {children}
    </div>
  );
}

/* ============================================================
   BUILD-VS-BUY TAB (independent state)
   ============================================================ */
function BuildVsBuyTab() {
  // Self-host config
  const [shModel, setShModel] = useState("llama3-8b");
  const [shGpu, setShGpu] = useState("a100-80");
  const [shQuant, setShQuant] = useState<Quantization>("fp16");
  const [shNumGpus, setShNumGpus] = useState(1);
  const [shGpuPrice, setShGpuPrice] = useState<number | "">("");
  const [utilization, setUtilization] = useState(50); // %
  const [batchSize, setBatchSize] = useState(8);

  // Workload
  const [inputTokens, setInputTokens] = useState(500);
  const [outputTokens, setOutputTokens] = useState(200);
  const [reqsPerDay, setReqsPerDay] = useState(1000);

  // API config
  const [apiProvider, setApiProvider] = useState<"openai" | "anthropic" | "google" | "groq" | "deepseek" | "mistral" | "together">("openai");
  const [apiModel, setApiModel] = useState("gpt-4o-mini");

  // === URL-share state ===
  const [urlRestored, setUrlRestored] = useState(false);

  // One-time mount: restore state from URL hash (priority: URL > localStorage > defaults)
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    if (typeof window === "undefined") return;
    const hash = window.location.hash;
    if (!hash || hash === "#") {
      // No URL hash — try localStorage
      const stored = loadBvbFromStorage();
      if (stored) {
        if (stored.shModel) setShModel(stored.shModel);
        if (stored.shGpu) setShGpu(stored.shGpu);
        if (stored.shQuant) setShQuant(stored.shQuant as Quantization);
        if (typeof stored.shNumGpus === "number") setShNumGpus(stored.shNumGpus);
        if (stored.shGpuPrice !== undefined) setShGpuPrice(stored.shGpuPrice);
        if (typeof stored.utilization === "number") setUtilization(stored.utilization);
        if (typeof stored.batchSize === "number") setBatchSize(stored.batchSize);
        if (typeof stored.inputTokens === "number") setInputTokens(stored.inputTokens);
        if (typeof stored.outputTokens === "number") setOutputTokens(stored.outputTokens);
        if (typeof stored.reqsPerDay === "number") setReqsPerDay(stored.reqsPerDay);
        if (stored.apiProvider) setApiProvider(stored.apiProvider as typeof apiProvider);
        if (stored.apiModel) setApiModel(stored.apiModel);
      }
      setUrlRestored(true);
      return;
    }
    const parsed = parseUrlHash(hash);
    if (parsed.tab !== "build-vs-buy" || !parsed.bvb) {
      setUrlRestored(true);
      return;
    }
    const b = parsed.bvb;
    if (b.shModel) setShModel(b.shModel);
    if (b.shGpu) setShGpu(b.shGpu);
    if (b.shQuant) setShQuant(b.shQuant as Quantization);
    if (typeof b.shNumGpus === "number") setShNumGpus(b.shNumGpus);
    if (b.shGpuPrice !== undefined) setShGpuPrice(b.shGpuPrice);
    if (typeof b.utilization === "number") setUtilization(b.utilization);
    if (typeof b.batchSize === "number") setBatchSize(b.batchSize);
    if (typeof b.inputTokens === "number") setInputTokens(b.inputTokens);
    if (typeof b.outputTokens === "number") setOutputTokens(b.outputTokens);
    if (typeof b.reqsPerDay === "number") setReqsPerDay(b.reqsPerDay);
    if (b.apiProvider) setApiProvider(b.apiProvider as typeof apiProvider);
    if (b.apiModel) setApiModel(b.apiModel);
    setUrlRestored(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  // Continuous: when state changes, update URL hash + localStorage
  useEffect(() => {
    if (!urlRestored) return;
    const state = {
      shModel, shGpu, shQuant, shNumGpus, shGpuPrice,
      utilization, batchSize, inputTokens, outputTokens, reqsPerDay,
      apiProvider, apiModel,
    };
    writeUrlHash(serializeBvbState(state));
    saveBvbToStorage(state);
  }, [
    urlRestored, shModel, shGpu, shQuant, shNumGpus, shGpuPrice,
    utilization, batchSize, inputTokens, outputTokens, reqsPerDay,
    apiProvider, apiModel,
  ]);

  const shGpuSpec = GPUS.find((g) => g.id === shGpu)!;
  const shModelSpec = MODELS.find((m) => m.id === shModel)!;
  const shQuantSpec = QUANTIZATIONS.find((q) => q.id === shQuant)!;

  // Run calc on self-host
  const sh = useMemo(() => calculate({
    modelId: shModel,
    gpuId: shGpu,
    quantization: shQuant,
    numGpus: shNumGpus,
    batchSize,
    promptTokens: inputTokens,
    outputTokens,
    gpuHourlyCost: shGpuPrice === "" ? undefined : Number(shGpuPrice),
  }), [shModel, shGpu, shQuant, shNumGpus, batchSize, inputTokens, outputTokens, shGpuPrice]);

  // Self-host cost per 1M tokens, accounting for utilization
  const effGpuPrice = (shGpuPrice === "" ? (shGpuSpec.usdPerHour ?? 0) : Number(shGpuPrice)) * shNumGpus;
  const effAggregateTokens = sh.aggregateTokensPerSec * (utilization / 100);
  const selfHostCostPerM = effAggregateTokens > 0
    ? (effGpuPrice / 3600 / effAggregateTokens) * 1e6
    : Infinity;
  const selfHostMonthly = effGpuPrice * 730; // 730 hours/month
  const tokensPerDay = effAggregateTokens * 3600 * (utilization / 100) * 24;

  // API pricing (cited 2025-2026 from research)
  const API_PRICES: Record<string, { in: number; out: number; cached: number | null; status: string }> = {
    "gpt-4o":       { in: 2.50, out: 10.00, cached: 1.25, status: "current" },
    "gpt-4o-mini": { in: 0.15, out: 0.60, cached: 0.075, status: "current" },
    "o1":          { in: 15.00, out: 60.00, cached: 7.50, status: "current" },
    "o3-mini":     { in: 1.10, out: 4.40, cached: 0.55, status: "current" },
    "claude-3.5-sonnet": { in: 3.00, out: 15.00, cached: 0.30, status: "retired Oct 2025" },
    "claude-3.5-haiku": { in: 0.80, out: 4.00, cached: 0.08, status: "retired Feb 2026" },
    "gemini-2.0-flash": { in: 0.10, out: 0.40, cached: null, status: "shut down Jun 2026" },
    "gemini-1.5-flash": { in: 0.075, out: 0.30, cached: 0.01875, status: "shut down Sep 2025" },
    "llama-3.1-70b-together": { in: 0.88, out: 0.88, cached: null, status: "current" },
    "llama-3.3-70b-groq": { in: 0.59, out: 0.79, cached: null, status: "current" },
    "deepseek-v4.1-flash": { in: 0.15, out: 0.60, cached: 0.003, status: "current" },
    "mistral-large-3": { in: 0.50, out: 1.50, cached: 0.05, status: "current" },
    "codestral": { in: 0.30, out: 0.90, cached: 0.03, status: "current" },
  };

  const apiPricing = API_PRICES[apiModel];
  // Per-request API cost: input tokens + output tokens (no caching assumed in this basic version)
  const apiCostPerRequest =
    (inputTokens / 1e6) * apiPricing.in + (outputTokens / 1e6) * apiPricing.out;
  const apiCostPerDay = apiCostPerRequest * reqsPerDay;
  const apiMonthlyCost = apiCostPerDay * 30;
  const apiCostPerMOut = apiPricing.out; // $/M output tokens (at face value)

  // Break-even: when monthly self-host = monthly API
  // self_host_monthly = api_monthly → reqs_per_day_break_even = self_host_monthly / (30 * api_cost_per_request)
  const breakEvenReqsPerDay = apiCostPerRequest > 0
    ? selfHostMonthly / (30 * apiCostPerRequest)
    : Infinity;

  // Verdict
  const cheaperThanApi = selfHostCostPerM < apiCostPerMOut;
  const meetsVolume = reqsPerDay > breakEvenReqsPerDay;

  return (
    <div className="space-y-4">
      <div className="mb-4 max-w-3xl">
        <h2 className="text-xl font-semibold tracking-tight mb-2">Should you self-host or use an API?</h2>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Tell us your workload and compare the cost of running on your own GPU vs paying per-token to an API.
          The break-even point tells you when self-hosting becomes cheaper.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Self-host side */}
        <Card className="border-border/60 shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Cpu className="size-4 text-emerald-500" />
              Self-host (rent GPUs)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <Label className="text-xs">Model</Label>
              <Select value={shModel} onValueChange={setShModel}>
                <SelectTrigger className="w-full mt-1"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-80">
                  {MODELS.filter(m => m.category === "text" || m.category === "code" || m.category === "reasoning").map(m => (
                    <SelectItem key={m.id} value={m.id}>{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs">GPU</Label>
              <Select value={shGpu} onValueChange={setShGpu}>
                <SelectTrigger className="w-full mt-1"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-80">
                  {(["datacenter", "workstation", "consumer", "tpu", "lpu", "wse"] as GpuCategory[]).map(cat => {
                    const items = GPUS.filter(g => g.category === cat);
                    if (items.length === 0) return null;
                    return (
                      <SelectGroup key={cat}>
                        <SelectLabel className="text-[10px] uppercase text-muted-foreground">{cat}</SelectLabel>
                        {items.map(g => <SelectItem key={g.id} value={g.id}>{g.name}</SelectItem>)}
                      </SelectGroup>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs">Quantization</Label>
              <Select value={shQuant} onValueChange={(v) => setShQuant(v as Quantization)}>
                <SelectTrigger className="w-full mt-1"><SelectValue /></SelectTrigger>
                <SelectContent className="max-h-80">
                  {QUANTIZATIONS.map(q => <SelectItem key={q.id} value={q.id}>{q.label}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <Label className="text-xs">GPUs</Label>
                <Badge variant="outline" className="text-[10px] font-mono">{shNumGpus}×</Badge>
              </div>
              <Slider value={[shNumGpus]} min={1} max={8} step={1} onValueChange={(v) => setShNumGpus(v[0])} className="w-full" />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <Label className="text-xs">GPU utilization</Label>
                <Badge variant="outline" className="text-[10px] font-mono">{utilization}%</Badge>
              </div>
              <Slider value={[utilization]} min={5} max={100} step={5} onValueChange={(v) => setUtilization(v[0])} className="w-full" />
              <p className="text-[10px] text-muted-foreground mt-1">Effective utilization is the decisive factor — not peak throughput.</p>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <Label className="text-xs">Batch size</Label>
                <Badge variant="outline" className="text-[10px] font-mono">{batchSize}</Badge>
              </div>
              <Slider value={[batchSize]} min={1} max={64} step={1} onValueChange={(v) => setBatchSize(v[0])} className="w-full" />
            </div>
            <div>
              <Label className="text-xs">GPU $/hr (blank = default ${(shGpuSpec.usdPerHour ?? "—")})</Label>
              <Input type="number" step="0.01" value={shGpuPrice} onChange={e => setShGpuPrice(e.target.value === "" ? "" : Number(e.target.value))} className="mt-1" />
            </div>
          </CardContent>
        </Card>

        {/* Workload + API side */}
        <Card className="border-border/60 shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <DollarSign className="size-4 text-emerald-500" />
              API alternative
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label className="text-xs">Input tokens/req</Label>
                <Input type="number" value={inputTokens} onChange={e => setInputTokens(Math.max(0, Number(e.target.value)))} className="mt-1" />
              </div>
              <div>
                <Label className="text-xs">Output tokens/req</Label>
                <Input type="number" value={outputTokens} onChange={e => setOutputTokens(Math.max(0, Number(e.target.value)))} className="mt-1" />
              </div>
            </div>
            <div>
              <Label className="text-xs">Requests per day</Label>
              <Input type="number" value={reqsPerDay} onChange={e => setReqsPerDay(Math.max(0, Number(e.target.value)))} className="mt-1" />
            </div>
            <Separator />
            <div>
              <Label className="text-xs">API provider</Label>
              <Select value={apiProvider} onValueChange={(v) => {
                setApiProvider(v as typeof apiProvider);
                // Set default model per provider
                const defaults: Record<string, string> = {
                  openai: "gpt-4o-mini",
                  anthropic: "claude-3.5-sonnet",
                  google: "gemini-2.0-flash",
                  groq: "llama-3.3-70b-groq",
                  deepseek: "deepseek-v4.1-flash",
                  mistral: "mistral-large-3",
                  together: "llama-3.1-70b-together",
                };
                setApiModel(defaults[v]);
              }}>
                <SelectTrigger className="w-full mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="google">Google Gemini</SelectItem>
                  <SelectItem value="groq">Groq</SelectItem>
                  <SelectItem value="deepseek">DeepSeek</SelectItem>
                  <SelectItem value="mistral">Mistral</SelectItem>
                  <SelectItem value="together">Together AI</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs">Model</Label>
              <Select value={apiModel} onValueChange={setApiModel}>
                <SelectTrigger className="w-full mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {Object.entries(API_PRICES).map(([key, p]) => (
                    <SelectItem key={key} value={key}>
                      {key} — ${p.in}/M in, ${p.out}/M out
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-[10px] text-amber-600 dark:text-amber-400 mt-1">
                Status: {apiPricing.status}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Verdict */}
      <Card className={`border-2 shadow-sm ${cheaperThanApi && meetsVolume ? "border-emerald-500" : "border-amber-500/40"}`}>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            {cheaperThanApi && meetsVolume ? (
              <CheckCircle2 className="size-4 text-emerald-500" />
            ) : (
              <AlertTriangle className="size-4 text-amber-500" />
            )}
            Verdict
          </CardTitle>
        </CardHeader>
        <CardContent>
          {cheaperThanApi && meetsVolume ? (
            <p className="text-sm">
              <strong className="text-emerald-500">Self-hosting is cheaper</strong> at your workload
              ({reqsPerDay.toLocaleString()} req/day). You pay{" "}
              <strong className="text-foreground">{fmtMoney(selfHostCostPerM)}/M output tokens</strong> vs the API&apos;s{" "}
              <strong className="text-foreground">{fmtMoney(apiCostPerMOut)}/M</strong>. Break-even was at{" "}
              {Math.round(breakEvenReqsPerDay).toLocaleString()} req/day.
            </p>
          ) : meetsVolume ? (
            <p className="text-sm">
              <strong className="text-amber-500">API is cheaper at this volume.</strong> You&apos;d pay{" "}
              <strong className="text-foreground">{fmtMoney(selfHostCostPerM)}/M</strong> self-hosting vs{" "}
              <strong className="text-foreground">{fmtMoney(apiCostPerMOut)}/M</strong> via API.
              Self-host becomes cheaper only above {Math.round(breakEvenReqsPerDay).toLocaleString()} req/day
              (you&apos;re at {reqsPerDay.toLocaleString()}).
            </p>
          ) : (
            <p className="text-sm">
              <strong className="text-amber-500">Not enough volume to justify self-hosting.</strong>{" "}
              You&apos;d need {Math.round(breakEvenReqsPerDay).toLocaleString()} req/day to break even;
              you&apos;re at {reqsPerDay.toLocaleString()}. Stick with the API.
            </p>
          )}
          <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
            <Metric label="Self-host $/M tokens" value={fmtMoney(selfHostCostPerM)} hint={`at ${utilization}% util`} />
            <Metric label="API $/M output" value={fmtMoney(apiCostPerMOut)} hint={apiModel} />
            <Metric label="Self-host monthly" value={fmtMoney(selfHostMonthly)} hint="730 hrs × $/hr" />
            <Metric label="API monthly" value={fmtMoney(apiMonthlyCost)} hint={`${reqsPerDay.toLocaleString()} req/day × 30`} />
          </div>
        </CardContent>
      </Card>
      <p className="text-[10px] text-muted-foreground leading-relaxed">
        Pricing is timestamped to research brief (Sept 2026). Providers retire models — Claude 3.5, Gemini 1.5/2.0, Mistral Large 2 etc. are already retired.
        Always verify at the provider&apos;s official pricing page before making purchase decisions.
      </p>
    </div>
  );
}

/* ============================================================
   REFERENCE TAB — catalog + pricing tables
   ============================================================ */
function ReferenceTab() {
  const [subview, setSubview] = useState<"models" | "gpus" | "quants" | "api" | "cloud">("models");

  return (
    <div className="space-y-4">
      <div className="mb-2 max-w-3xl">
        <h2 className="text-xl font-semibold tracking-tight mb-1">Reference catalog</h2>
        <p className="text-sm text-muted-foreground leading-relaxed">
          All models, GPUs, quantizations, and live API/cloud pricing used by tokcalc.
          Every record has a source link where available.
        </p>
      </div>

      {/* Sub-tabs */}
      <div className="border-b border-border/60">
        <div className="flex gap-1 -mb-px overflow-x-auto">
          {([
            { id: "models", label: "Models" },
            { id: "gpus", label: "GPUs" },
            { id: "quants", label: "Quantization" },
            { id: "api", label: "API pricing" },
            { id: "cloud", label: "Cloud GPU pricing" },
          ] as const).map(t => (
            <button
              key={t.id}
              onClick={() => setSubview(t.id)}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors whitespace-nowrap ${
                subview === t.id
                  ? "border-emerald-500 text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {subview === "models" && (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-muted/40 text-[10px] uppercase">
                  <tr>
                    <th className="text-left px-3 py-2">Model</th>
                    <th className="text-left px-3 py-2">Family</th>
                    <th className="text-left px-3 py-2">Cat</th>
                    <th className="text-right px-3 py-2">Params (total)</th>
                    <th className="text-right px-3 py-2">Active</th>
                    <th className="text-right px-3 py-2">Layers</th>
                    <th className="text-right px-3 py-2">Hidden</th>
                    <th className="text-right px-3 py-2">Q/KV</th>
                    <th className="text-right px-3 py-2">Ctx</th>
                    <th className="text-center px-3 py-2">MoE</th>
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map(m => (
                    <tr key={m.id} className="border-t border-border/40 hover:bg-muted/20">
                      <td className="px-3 py-1.5 font-medium">{m.name}</td>
                      <td className="px-3 py-1.5 text-muted-foreground">{m.family}</td>
                      <td className="px-3 py-1.5 text-muted-foreground">{m.category}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{m.paramsB}B</td>
                      <td className="px-3 py-1.5 text-right font-mono">{m.activeParamsB}B</td>
                      <td className="px-3 py-1.5 text-right font-mono">{m.layers}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{m.hiddenDim}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{m.qHeads}/{m.kvHeads}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{(m.maxContext / 1000).toFixed(0)}K</td>
                      <td className="px-3 py-1.5 text-center">{m.isMoE ? "✓" : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {subview === "gpus" && (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-muted/40 text-[10px] uppercase">
                  <tr>
                    <th className="text-left px-3 py-2">GPU</th>
                    <th className="text-left px-3 py-2">Vendor</th>
                    <th className="text-left px-3 py-2">Category</th>
                    <th className="text-right px-3 py-2">Mem BW</th>
                    <th className="text-right px-3 py-2">FP16 TF</th>
                    <th className="text-right px-3 py-2">VRAM</th>
                    <th className="text-right px-3 py-2">NVLink</th>
                    <th className="text-right px-3 py-2">$/hr</th>
                    <th className="text-right px-3 py-2">Year</th>
                  </tr>
                </thead>
                <tbody>
                  {GPUS.map(g => (
                    <tr key={g.id} className="border-t border-border/40 hover:bg-muted/20">
                      <td className="px-3 py-1.5 font-medium">
                        {g.name}
                        {g.note && <span className="ml-2 text-[10px] text-amber-600 dark:text-amber-400">⚠</span>}
                      </td>
                      <td className="px-3 py-1.5 text-muted-foreground">{g.vendor}</td>
                      <td className="px-3 py-1.5 text-muted-foreground">{g.category}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{g.memBandwidthGbps}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{g.flopsTflops ?? "—"}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{g.vramGb}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{g.nvlinkGbps || "—"}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{g.usdPerHour ?? "quote"}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{g.year}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {GPUS.some(g => g.note) && (
              <div className="p-3 text-[10px] text-muted-foreground">
                ⚠ = special note. Hover over the GPU name in the Calculator tab dropdown to see details.
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {subview === "quants" && (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-muted/40 text-[10px] uppercase">
                  <tr>
                    <th className="text-left px-3 py-2">Format</th>
                    <th className="text-left px-3 py-2">Family</th>
                    <th className="text-right px-3 py-2">Bytes/param</th>
                    <th className="text-right px-3 py-2">Eff. bits</th>
                    <th className="text-right px-3 py-2">Efficiency</th>
                    <th className="text-left px-3 py-2">Use case</th>
                    <th className="text-left px-3 py-2">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {QUANTIZATIONS.map(q => (
                    <tr key={q.id} className="border-t border-border/40 hover:bg-muted/20">
                      <td className="px-3 py-1.5 font-medium">{q.label}</td>
                      <td className="px-3 py-1.5 text-muted-foreground">{q.family}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{q.bytesPerParam}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{(q.bytesPerParam * 8).toFixed(2)}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{q.efficiency.toFixed(2)}×</td>
                      <td className="px-3 py-1.5 text-muted-foreground text-[10px]">{q.useCase ?? "—"}</td>
                      <td className="px-3 py-1.5 text-muted-foreground text-[10px]">{q.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {subview === "api" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">LLM API pricing ($/M tokens)</CardTitle>
            <CardDescription className="text-xs">
              Timestamped to research brief (Sept 2026). Verify at provider&apos;s official pricing page before purchase.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-muted/40 text-[10px] uppercase">
                  <tr>
                    <th className="text-left px-3 py-2">Model</th>
                    <th className="text-right px-3 py-2">Input $/M</th>
                    <th className="text-right px-3 py-2">Cached $/M</th>
                    <th className="text-right px-3 py-2">Output $/M</th>
                    <th className="text-left px-3 py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["gpt-4o", 2.50, 1.25, 10.00, "current"],
                    ["gpt-4o-mini", 0.15, 0.075, 0.60, "current"],
                    ["o1", 15.00, 7.50, 60.00, "current"],
                    ["o3-mini", 1.10, 0.55, 4.40, "current"],
                    ["claude-3.5-sonnet", 3.00, 0.30, 15.00, "retired Oct 2025"],
                    ["claude-3.5-haiku", 0.80, 0.08, 4.00, "retired Feb 2026"],
                    ["gemini-2.0-flash", 0.10, null, 0.40, "shut down Jun 2026"],
                    ["gemini-1.5-flash", 0.075, 0.01875, 0.30, "shut down Sep 2025"],
                    ["deepseek-v4.1-flash", 0.15, 0.003, 0.60, "current"],
                    ["mistral-large-3", 0.50, 0.05, 1.50, "current"],
                    ["codestral", 0.30, 0.03, 0.90, "current"],
                    ["llama-3.1-70b-together", 0.88, null, 0.88, "current"],
                    ["llama-3.3-70b-groq", 0.59, null, 0.79, "current"],
                  ].map(([m, i, c, o, s]) => (
                    <tr key={m as string} className="border-t border-border/40 hover:bg-muted/20">
                      <td className="px-3 py-1.5 font-medium">{m}</td>
                      <td className="px-3 py-1.5 text-right font-mono">${i}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{c === null ? "—" : `$${c}`}</td>
                      <td className="px-3 py-1.5 text-right font-mono">${o}</td>
                      <td className="px-3 py-1.5 text-muted-foreground text-[10px]">{s}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {subview === "cloud" && (
        <>
          {/* Live Azure prices (fetched from Azure Retail Prices API) */}
          <AzureLivePricing />

          {/* Vast.ai marketplace spot prices */}
          <VastAiLivePricing />

          {/* Static estimates from tokcalc catalog */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Static estimates (from research catalog)</CardTitle>
              <CardDescription className="text-xs">
                Approximate on-demand rates from research brief. Prices vary by region, commitment, and availability.
                Live Azure prices shown above.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="bg-muted/40 text-[10px] uppercase">
                    <tr>
                      <th className="text-left px-3 py-2">GPU</th>
                      <th className="text-left px-3 py-2">Vendor</th>
                      <th className="text-right px-3 py-2">$/hr (default)</th>
                      <th className="text-left px-3 py-2">Typical providers</th>
                    </tr>
                  </thead>
                  <tbody>
                    {GPUS.filter(g => g.usdPerHour !== null && g.usdPerHour > 0).map(g => (
                      <tr key={g.id} className="border-t border-border/40 hover:bg-muted/20">
                        <td className="px-3 py-1.5 font-medium">{g.name}</td>
                        <td className="px-3 py-1.5 text-muted-foreground">{g.vendor}</td>
                        <td className="px-3 py-1.5 text-right font-mono">${g.usdPerHour}</td>
                        <td className="px-3 py-1.5 text-muted-foreground text-[10px]">
                          {g.category === "datacenter" && "RunPod · Lambda · CoreWeave · AWS · GCP"}
                          {g.category === "workstation" && "TensorDock · RunPod"}
                          {g.category === "consumer" && "Vast.ai · TensorDock (spot)"}
                          {g.category === "tpu" && "Google Cloud TPU"}
                          {g.category === "legacy" && "Secondary market"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
