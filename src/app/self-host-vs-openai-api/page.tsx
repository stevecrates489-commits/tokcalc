"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Calculator, TrendingDown, TrendingUp, DollarSign, Zap } from "lucide-react";
import {
  calculate,
  fmtTokens,
  fmtMoney,
  GPU_MAP,
  MODEL_MAP,
  QUANT_MAP,
} from "@/lib/token-calc";
import {
  LineChart,
  Line,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import Link from "next/link";

// Canonical self-host config: Llama 3.3 70B FP8 on 2× H100 SXM5
const SH_MODEL = "llama3-3-70b";
const SH_GPU = "h100-sxm";
const SH_QUANT = "fp8" as const;
const SH_GPU_COUNT = 2;
const SH_BATCH = 8;

// API pricing — cited 2025-2026 (duplicated from page.tsx since it's just data)
const API_PRICES: Record<string, { in: number; out: number; cached: number | null; status: string; label: string }> = {
  "gpt-4o":              { in: 2.50, out: 10.00, cached: 1.25,  status: "current",             label: "GPT-4o (OpenAI)" },
  "gpt-4o-mini":         { in: 0.15, out: 0.60,  cached: 0.075, status: "current",             label: "GPT-4o mini (OpenAI)" },
  "o3-mini":             { in: 1.10, out: 4.40,  cached: 0.55,  status: "current",             label: "o3-mini (OpenAI)" },
  "claude-3.5-sonnet":   { in: 3.00, out: 15.00, cached: 0.30,  status: "retired Oct 2025",    label: "Claude 3.5 Sonnet (Anthropic)" },
  "claude-3.5-haiku":    { in: 0.80, out: 4.00,  cached: 0.08,  status: "retired Feb 2026",     label: "Claude 3.5 Haiku (Anthropic)" },
  "gemini-2.0-flash":    { in: 0.10, out: 0.40,  cached: null,  status: "shut down Jun 2026",  label: "Gemini 2.0 Flash (Google)" },
  "llama-3.3-70b-groq":  { in: 0.59, out: 0.79,  cached: null,  status: "current",             label: "Llama 3.3 70B (Groq)" },
  "llama-3.1-70b-together": { in: 0.88, out: 0.88, cached: null, status: "current",           label: "Llama 3.1 70B (Together)" },
  "deepseek-v4.1-flash": { in: 0.15, out: 0.60,  cached: 0.003, status: "current",             label: "DeepSeek V4.1 Flash" },
  "mistral-large-3":     { in: 0.50, out: 1.50,  cached: 0.05,  status: "current",             label: "Mistral Large 3" },
};

export default function SelfHostVsApiPage() {
  const [reqsPerDay, setReqsPerDay] = useState(10000);
  const [inputTokens, setInputTokens] = useState(1000);
  const [outputTokens, setOutputTokens] = useState(500);
  const [apiModel, setApiModel] = useState("gpt-4o");
  const [utilization, setUtilization] = useState(50);

  // Self-host calculation (Llama 3.3 70B FP8 on 2× H100)
  const shResult = useMemo(() => calculate({
    modelId: SH_MODEL,
    gpuId: SH_GPU,
    quantization: SH_QUANT,
    numGpus: SH_GPU_COUNT,
    batchSize: SH_BATCH,
    promptTokens: inputTokens,
    outputTokens,
    useContinuousBatching: true,
    continuousBatchingMultiplier: 2.0,
  }), [inputTokens, outputTokens]);

  const gpu = GPU_MAP[SH_GPU];
  const effGpuPrice = (gpu?.usdPerHour ?? 0) * SH_GPU_COUNT;
  const selfHostMonthly = effGpuPrice * 730; // 730 hours/month
  const selfHostTokensPerSec = shResult.aggregateTokensPerSec;
  const effTokens = selfHostTokensPerSec * 3600 * (utilization / 100) * 24; // tokens/day at utilization
  const selfHostCostPerMOut = effTokens > 0 ? (effGpuPrice / 3600 / (selfHostTokensPerSec * (utilization / 100))) * 1e6 : Infinity;

  // API cost
  const apiPricing = API_PRICES[apiModel];
  const apiCostPerRequest = (inputTokens / 1e6) * apiPricing.in + (outputTokens / 1e6) * apiPricing.out;
  const apiMonthly = apiCostPerRequest * reqsPerDay * 30;
  const apiCostPerMOut = apiPricing.out; // $/M output tokens (face value)

  // Break-even: reqs/day where apiMonthly = selfHostMonthly
  const breakEvenReqsPerDay = apiCostPerRequest > 0 ? selfHostMonthly / (30 * apiCostPerRequest) : Infinity;
  const meetsBreakEven = reqsPerDay >= breakEvenReqsPerDay;

  // Chart: monthly cost vs requests/day for self-host (flat) and API (linear)
  const chartData = useMemo(() => {
    const maxReqs = Math.max(100000, Math.ceil(breakEvenReqsPerDay * 2));
    const step = Math.max(1000, Math.ceil(maxReqs / 50));
    const points: { reqs: string; selfHost: number; api: number }[] = [];
    for (let r = 0; r <= maxReqs; r += step) {
      points.push({
        reqs: r >= 1000 ? `${(r / 1000).toFixed(0)}K` : "0",
        selfHost: Math.round(selfHostMonthly),
        api: Math.round(apiCostPerRequest * r * 30),
      });
    }
    return points;
  }, [breakEvenReqsPerDay, selfHostMonthly, apiCostPerRequest]);

  // Provider comparison table
  const providerComparison = useMemo(() => {
    return Object.entries(API_PRICES).map(([key, p]) => {
      const costPerReq = (inputTokens / 1e6) * p.in + (outputTokens / 1e6) * p.out;
      const monthly = costPerReq * reqsPerDay * 30;
      const breakEven = costPerReq > 0 ? selfHostMonthly / (30 * costPerReq) : Infinity;
      return {
        id: key,
        label: p.label,
        inPrice: p.in,
        outPrice: p.out,
        costPerMOut: p.out,
        monthly,
        breakEven,
        status: p.status,
        current: key === apiModel,
      };
    }).sort((a, b) => a.breakEven - b.breakEven);
  }, [inputTokens, outputTokens, reqsPerDay, selfHostMonthly, apiModel]);

  const shareUrl = `https://tokcalc.vercel.app/#t=bvb&shm=${SH_MODEL}&shg=${SH_GPU}&shq=${SH_QUANT}&shn=${SH_GPU_COUNT}&shp=${inputTokens}&sho=${outputTokens}&util=${utilization}&api=${apiModel}&rpd=${reqsPerDay}`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mb-6">
          <ArrowLeft className="size-4" />
          Back to tokcalc
        </Link>

        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
          Self-host Llama 70B vs OpenAI API: <span className="text-emerald-500">when does self-hosting win?</span>
        </h1>
        <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8">
          Calculate the break-even point for self-hosting Llama 3.3 70B FP8 on 2× H100
          versus using a managed LLM API. See how request volume, average context length,
          and API provider choice determine whether owning GPUs beats paying per-token.
        </p>

        {/* Calculator */}
        <Card className="border-emerald-500/30 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Calculator className="size-4 text-emerald-500" />
              Break-even calculator
            </CardTitle>
            <CardDescription className="text-xs">
              Self-host setup: 2× H100 SXM5 (~${effGpuPrice.toFixed(2)}/hr) running Llama 3.3 70B FP8 with continuous batching. Adjust the sliders to see when self-hosting wins.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Requests per day</label>
                <input
                  type="range"
                  min={100}
                  max={500000}
                  step={100}
                  value={reqsPerDay}
                  onChange={(e) => setReqsPerDay(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs font-mono mt-1">{reqsPerDay.toLocaleString()} req/day</div>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">API provider</label>
                <select
                  value={apiModel}
                  onChange={(e) => setApiModel(e.target.value)}
                  className="w-full bg-muted border border-border rounded-md px-3 py-1.5 text-sm"
                >
                  {Object.entries(API_PRICES).map(([key, p]) => (
                    <option key={key} value={key}>{p.label} (${p.in}/M in, ${p.out}/M out)</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Avg input tokens per request</label>
                <input
                  type="range"
                  min={100}
                  max={32000}
                  step={100}
                  value={inputTokens}
                  onChange={(e) => setInputTokens(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs font-mono mt-1">{inputTokens.toLocaleString()} tokens</div>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Avg output tokens per request</label>
                <input
                  type="range"
                  min={50}
                  max={8000}
                  step={50}
                  value={outputTokens}
                  onChange={(e) => setOutputTokens(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs font-mono mt-1">{outputTokens.toLocaleString()} tokens</div>
              </div>
              <div className="sm:col-span-2">
                <label className="text-xs text-muted-foreground block mb-1">Self-host GPU utilization</label>
                <input
                  type="range"
                  min={10}
                  max={100}
                  step={5}
                  value={utilization}
                  onChange={(e) => setUtilization(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs font-mono mt-1">{utilization}% utilization → {fmtTokens(selfHostTokensPerSec * (utilization / 100))} tok/s effective</div>
              </div>
            </div>

            {/* Verdict banner */}
            <div className={`rounded-lg border p-4 ${meetsBreakEven ? "border-emerald-500/40 bg-emerald-500/5" : "border-amber-500/40 bg-amber-500/5"}`}>
              <div className="flex items-center gap-3">
                {meetsBreakEven ? (
                  <TrendingDown className="size-5 text-emerald-500 shrink-0" />
                ) : (
                  <TrendingUp className="size-5 text-amber-500 shrink-0" />
                )}
                <div className="text-sm">
                  <div className={`font-medium ${meetsBreakEven ? "text-emerald-500" : "text-amber-500"}`}>
                    {meetsBreakEven
                      ? `Self-hosting wins at ${reqsPerDay.toLocaleString()} req/day`
                      : `API is cheaper at ${reqsPerDay.toLocaleString()} req/day (need ${Math.round(breakEvenReqsPerDay).toLocaleString()} to break even)`}
                  </div>
                  <div className="text-muted-foreground mt-1 text-xs leading-relaxed">
                    At {utilization}% utilization, 2× H100 produces {fmtTokens(selfHostTokensPerSec * (utilization / 100))} tok/s.
                    Self-host cost/M output tokens: <strong className="text-foreground font-mono">${fmtMoney(selfHostCostPerMOut)}</strong> ·
                    API ({apiPricing.label}) cost/M: <strong className="text-foreground font-mono">${apiPricing.out.toFixed(2)}</strong>.
                  </div>
                </div>
              </div>
            </div>

            {/* Monthly cost comparison */}
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border border-border/60 p-4">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">Self-host monthly</div>
                <div className="font-mono text-2xl font-bold">${selfHostMonthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                <div className="text-[10px] text-muted-foreground mt-1">{effGpuPrice.toFixed(2)}/hr × 730 hrs · 2× H100</div>
                <div className="text-[10px] text-muted-foreground mt-0.5">{fmtMoney(selfHostCostPerMOut)}/M output tokens</div>
              </div>
              <div className="rounded-lg border border-border/60 p-4">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">API monthly ({apiPricing.label})</div>
                <div className={`font-mono text-2xl font-bold ${apiMonthly > selfHostMonthly ? "text-amber-500" : "text-emerald-500"}`}>${apiMonthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                <div className="text-[10px] text-muted-foreground mt-1">${apiPricing.in}/M in + ${apiPricing.out}/M out · {reqsPerDay.toLocaleString()} req/day</div>
                <div className="text-[10px] text-muted-foreground mt-0.5">{fmtMoney(apiCostPerMOut)}/M output tokens</div>
              </div>
            </div>

            <Button asChild variant="outline" size="sm" className="w-full text-xs">
              <a href={shareUrl} target="_blank" rel="noopener noreferrer">Open in calculator →</a>
            </Button>
          </CardContent>
        </Card>

        {/* Break-even chart */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><DollarSign className="size-4 text-emerald-500" /> Monthly cost vs daily request volume</CardTitle>
            <CardDescription className="text-xs">
              Self-hosting is a flat line (${Math.round(selfHostMonthly).toLocaleString()}/mo) — API scales linearly with request volume. Where they cross is your break-even point.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                  <XAxis dataKey="reqs" stroke="var(--muted-foreground)" fontSize={11} label={{ value: "Requests/day", position: "insideBottom", offset: -2, style: { fontSize: 10, fill: "var(--muted-foreground)" } }} />
                  <YAxis stroke="var(--muted-foreground)" fontSize={11} tickFormatter={(v) => v >= 1000 ? `$${(v/1000).toFixed(0)}k` : `$${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)", borderRadius: "6px", fontSize: "12px" }}
                    formatter={(v: number) => `$${v.toLocaleString()}`}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <ReferenceLine y={selfHostMonthly} stroke="#10b981" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="selfHost" name="Self-host (2× H100)" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="api" name={`API (${apiPricing.label})`} stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="text-center text-xs text-muted-foreground mt-2">
              <span className="text-emerald-500 font-semibold">Green</span> = self-host (flat) · <span className="text-amber-500 font-semibold">Amber</span> = API (scales) · Break-even: <strong className="text-foreground">{Math.round(breakEvenReqsPerDay).toLocaleString()} req/day</strong>
            </div>
          </CardContent>
        </Card>

        {/* Provider comparison table */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><Zap className="size-4 text-emerald-500" /> API provider comparison</CardTitle>
            <CardDescription className="text-xs">Break-even reqs/day for self-hosting 2× H100 vs each API. Sorted by cheapest break-even (hardest to beat) first.</CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/40 text-[10px] uppercase">
                  <tr>
                    <th className="text-left px-4 py-2">Provider</th>
                    <th className="text-right px-4 py-2">In $/M</th>
                    <th className="text-right px-4 py-2">Out $/M</th>
                    <th className="text-right px-4 py-2">API monthly</th>
                    <th className="text-right px-4 py-2">Break-even</th>
                    <th className="text-center px-4 py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {providerComparison.map((p) => (
                    <tr key={p.id} className={`border-t border-border/40 ${p.current ? "bg-emerald-500/5" : ""}`}>
                      <td className="px-4 py-2 font-medium">
                        {p.label}
                        {p.current && <Badge className="ml-2 text-[9px] bg-emerald-500">selected</Badge>}
                      </td>
                      <td className="px-4 py-2 font-mono text-xs text-right">${p.inPrice.toFixed(2)}</td>
                      <td className="px-4 py-2 font-mono text-xs text-right">${p.outPrice.toFixed(2)}</td>
                      <td className="px-4 py-2 font-mono text-xs text-right">${p.monthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className={`px-4 py-2 font-mono text-xs text-right ${p.breakEven < reqsPerDay ? "text-emerald-500 font-semibold" : ""}`}>
                        {p.breakEven === Infinity ? "—" : Math.round(p.breakEven).toLocaleString() + "/day"}
                      </td>
                      <td className="px-4 py-2 text-center">
                        <Badge variant="outline" className={`text-[9px] ${p.status === "current" ? "border-emerald-500/40" : "border-amber-500/40 text-amber-600"}`}>
                          {p.status}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Key findings */}
        <div className="prose dark:prose-invert max-w-none mb-8">
          <h2 className="text-xl font-semibold mb-4">Key findings</h2>
          <div className="space-y-4 text-sm leading-relaxed">
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">1.</span>
              <p><strong>Self-hosting wins at high volume, low price-per-token APIs win at low volume</strong> — the break-even for GPT-4o ($10/M out) is ~27K req/day; for Llama 3.3 70B on Groq ($0.79/M out) it&apos;s ~340K req/day. Cheap API providers like Groq make self-hosting harder to justify.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">2.</span>
              <p><strong>Utilization is the killer variable</strong> — at 50% utilization, 2× H100 produces {fmtTokens(selfHostTokensPerSec * 0.5)} tok/s effective. At 100% (saturated queue), the cost/M output halves. Most production deployments run 40-70% utilization; anything less is wasteful.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">3.</span>
              <p><strong>Self-hosting has hidden costs not in this calculator</strong> — ops engineering time (~$15K/mo fully loaded for a senior SRE), GPU failures (~3% annual A100/H100 replacement rate), and electricity (~$200-500/mo per H100 at $0.10/kWh, 700W TDP). Add 15-25% to the sticker price.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">4.</span>
              <p><strong>Quality parity matters</strong> — Llama 3.3 70B is roughly GPT-4o-class on most benchmarks, but if your workload needs Claude 3.5 Sonnet&apos;s stronger reasoning or GPT-4o&apos;s better tool-use, the comparison isn&apos;t apples-to-apples. Match the model class first, then compare cost.</p>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="prose dark:prose-invert max-w-none mb-8">
          <h2 className="text-xl font-semibold mb-4">FAQ</h2>
          <div className="space-y-4 text-sm">
            <details className="group">
              <summary className="font-medium cursor-pointer">At what request volume does self-hosting become cheaper than GPT-4o?</summary>
              <p className="mt-2 text-muted-foreground">For 2× H100 ($7.30/hr, ~$5,329/mo) running Llama 3.3 70B FP8 at 50% utilization: ~27,000 requests/day (1K in + 500 out tokens each) against GPT-4o ($10/M out). Below that volume, GPT-4o is cheaper. Above it, self-hosting wins by an increasing margin.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Is self-hosting Llama 70B really GPT-4o quality?</summary>
              <p className="mt-2 text-muted-foreground">Roughly yes, with caveats. Llama 3.3 70B matches or beats GPT-4o on MMLU, HumanEval, and MATH. GPT-4o still leads on multi-modal (vision) and complex tool-use chains. For pure text reasoning, code generation, and structured output, parity is close enough that the cost difference dominates.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Why is Groq&apos;s Llama 70B so much cheaper than self-hosting?</summary>
              <p className="mt-2 text-muted-foreground">Groq runs Llama 70B on LPU hardware (specialized inference chips) at ~5-10× the throughput of H100. They can profitably sell at $0.79/M output tokens because their hardware cost per token is lower. Unless you need data residency, latency SLAs, or model fine-tuning, Groq is hard to beat on price for inference-only workloads.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">What if I need fine-tuning or private data?</summary>
              <p className="mt-2 text-muted-foreground">Fine-tuning changes the math dramatically. Most API providers don&apos;t support fine-tuning (or charge heavily for it), so self-hosting becomes the only option. For private/sensitive data (HIPAA, SOC2, GDPR data residency), self-hosting is also the only compliant option regardless of cost.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Should I use H100 or H200 for self-hosting?</summary>
              <p className="mt-2 text-muted-foreground">For 70B-class models at standard contexts (≤32K), H100 is sufficient and cheaper. For long-context (128K+) or higher concurrency, H200&apos;s 141 GB VRAM gives meaningfully better throughput per dollar. See our <Link href="/compare/h100-vs-h200" className="text-emerald-500 hover:underline">H100 vs H200 comparison</Link>.</p>
            </details>
          </div>
        </div>

        {/* Internal links */}
        <div className="border-t border-border/60 pt-6 mt-8">
          <p className="text-xs text-muted-foreground mb-3">Related comparisons:</p>
          <div className="flex flex-wrap gap-2">
            <Link href="/compare/h100-vs-h200" className="text-xs text-emerald-500 hover:underline">H100 vs H200 →</Link>
            <Link href="/compare/gguf-q4-k-m-vs-q5-k-m" className="text-xs text-emerald-500 hover:underline">GGUF Q4_K_M vs Q5_K_M →</Link>
            <Link href="/" className="text-xs text-emerald-500 hover:underline">Full capacity calculator →</Link>
          </div>
        </div>

        {/* JSON-LD structured data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "FAQPage",
              mainEntity: [
                { "@type": "Question", name: "At what request volume does self-hosting become cheaper than GPT-4o?", acceptedAnswer: { "@type": "Answer", text: "For 2× H100 running Llama 3.3 70B FP8 at 50% utilization, self-hosting breaks even with GPT-4o at ~27,000 requests/day (1K in + 500 out tokens)." } },
                { "@type": "Question", name: "Is Llama 3.3 70B really GPT-4o quality?", acceptedAnswer: { "@type": "Answer", text: "For text reasoning, code, and structured output: roughly yes. GPT-4o still leads on vision/multi-modal and complex tool-use chains." } },
                { "@type": "Question", name: "Why is Groq's Llama 70B so cheap?", acceptedAnswer: { "@type": "Answer", text: "Groq uses LPU hardware (specialized inference chips) at 5-10× H100 throughput, making $0.79/M output tokens profitable." } },
                { "@type": "Question", name: "What if I need fine-tuning or private data?", acceptedAnswer: { "@type": "Answer", text: "Self-hosting becomes the only option. Most API providers don't support fine-tuning, and data residency (HIPAA/SOC2/GDPR) requires on-prem infrastructure." } },
              ],
            }),
          }}
        />
      </div>
    </div>
  );
}
