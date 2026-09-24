"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Cpu, Zap, DollarSign, MemoryStick, Gauge } from "lucide-react";
import {
  calculate,
  fmtTokens,
  fmtBytes,
  fmtMs,
  fmtMoney,
  GPU_MAP,
  MODELS,
  QUANT_MAP,
  type Quantization,
} from "@/lib/token-calc";
import { ConfidenceDot } from "@/components/confidence-badge";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Bar as BarChartBar,
} from "recharts";
import Link from "next/link";

const MODEL_ID = "llama3-3-70b";
const QUANT: Quantization = "fp8";

const H100 = GPU_MAP["h100-sxm"];
const H200 = GPU_MAP["h200-sxm"];
const model = MODELS.find((m) => m.id === MODEL_ID)!;
const quant = QUANT_MAP[QUANT];

function calcForGpu(gpuId: string, gpuCount: number, contextTokens: number) {
  return calculate({
    modelId: MODEL_ID,
    gpuId,
    quantization: QUANT,
    numGpus: gpuCount,
    batchSize: 8,
    promptTokens: contextTokens,
    outputTokens: 500,
  });
}

const SPEC_ROWS = [
  { label: "VRAM", h100: "80 GB HBM3", h200: "141 GB HBM3e", winner: "h200" },
  { label: "Memory bandwidth", h100: "3,350 GB/s", h200: "4,800 GB/s", winner: "h200" },
  { label: "FP16/BF16 dense TFLOPS", h100: "990 TF", h200: "990 TF", winner: "tie" },
  { label: "FP8 support", h100: "Native", h200: "Native", winner: "tie" },
  { label: "NVLink bandwidth", h100: "900 GB/s", h200: "900 GB/s", winner: "tie" },
  { label: "Typical cloud price", h100: "~$2.50/hr", h200: "~$4.00/hr", winner: "h100" },
  { label: "Release year", h100: "2022", h200: "2024", winner: "h200" },
  { label: "TDP (power)", h100: "700W", h200: "700W", winner: "tie" },
];

export default function H100vsH200Page() {
  const [context, setContext] = useState(50000);
  const [gpuCount, setGpuCount] = useState(2);

  const h100Result = useMemo(() => calcForGpu("h100-sxm", gpuCount, context), [gpuCount, context]);
  const h200Result = useMemo(() => calcForGpu("h200-sxm", gpuCount, context), [gpuCount, context]);

  const chartData = useMemo(() => {
    const ctxs = [4096, 8192, 16384, 32768, 65536, 131072];
    return ctxs.map((ctx) => ({
      context: `${ctx / 1000}K`,
      h100: Math.max(0, Math.round(calcForGpu("h100-sxm", gpuCount, ctx).aggregateTokensPerSec)),
      h200: Math.max(0, Math.round(calcForGpu("h200-sxm", gpuCount, ctx).aggregateTokensPerSec)),
    }));
  }, [gpuCount]);

  const chartConcurrency = useMemo(() => {
    const ctxs = [4096, 8192, 16384, 32768, 65536, 131072];
    return ctxs.map((ctx) => {
      const r1 = calcForGpu("h100-sxm", gpuCount, ctx);
      const r2 = calcForGpu("h200-sxm", gpuCount, ctx);
      return {
        context: `${ctx / 1000}K`,
        h100: r1.vramFits ? Math.floor(((H100!.vramGb * gpuCount) - model.paramsB * quant.bytesPerParam) / (r1.kvCachePerTokenKb * 1024 / 1e6 / ctx)) : 0,
        h200: r2.vramFits ? Math.floor(((H200!.vramGb * gpuCount) - model.paramsB * quant.bytesPerParam) / (r2.kvCachePerTokenKb * 1024 / 1e6 / ctx)) : 0,
      };
    });
  }, [gpuCount]);

  const shareH100 = `https://tokcalc.vercel.app/#t=calculator&m=${MODEL_ID}&g=h100-sxm&q=${QUANT}&n=${gpuCount}&b=8&p=${context}&o=500&cb=1&cbm=2`;
  const shareH200 = `https://tokcalc.vercel.app/#t=calculator&m=${MODEL_ID}&g=h200-sxm&q=${QUANT}&n=${gpuCount}&b=8&p=${context}&o=500&cb=1&cbm=2`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mb-6">
          <ArrowLeft className="size-4" />
          Back to tokcalc
        </Link>

        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
          H100 vs H200 for LLM Inference: <span className="text-emerald-500">Throughput, Latency, VRAM & Cost</span>
        </h1>
        <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8">
          Side-by-side comparison of NVIDIA H100 SXM5 80GB vs H200 SXM5 141GB for serving
          Llama 3.3 70B (FP8). See exactly how context length, VRAM, and bandwidth affect
          your deployment capacity — with transparent formulas, not marketing claims.
        </p>

        {/* Spec comparison */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Cpu className="size-4 text-emerald-500" />
              Specification comparison
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Spec</th>
                  <th className="text-left px-4 py-2">H100 SXM5</th>
                  <th className="text-left px-4 py-2">H200 SXM5</th>
                  <th className="text-center px-4 py-2">Winner</th>
                </tr>
              </thead>
              <tbody>
                {SPEC_ROWS.map((row, i) => (
                  <tr key={i} className="border-t border-border/40">
                    <td className="px-4 py-2 font-medium">{row.label}</td>
                    <td className="px-4 py-2 font-mono text-xs">{row.h100}</td>
                    <td className="px-4 py-2 font-mono text-xs">{row.h200}</td>
                    <td className="px-4 py-2 text-center">
                      {row.winner === "h200" && <Badge className="text-[9px] bg-emerald-500">H200</Badge>}
                      {row.winner === "h100" && <Badge className="text-[9px] bg-blue-500">H100</Badge>}
                      {row.winner === "tie" && <span className="text-[10px] text-muted-foreground">Tie</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Interactive comparison */}
        <Card className="border-emerald-500/30 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Gauge className="size-4 text-emerald-500" />
              Live comparison — adjust context length & GPU count
            </CardTitle>
            <CardDescription className="text-xs">
              All numbers computed by tokcalc&apos;s open-source formulas. Click a config to open the full calculator.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Context length (tokens)</label>
                <select
                  value={context}
                  onChange={(e) => setContext(Number(e.target.value))}
                  className="bg-muted border border-border rounded-md px-3 py-1.5 text-sm"
                >
                  <option value={500}>500 (chat)</option>
                  <option value={8192}>8K</option>
                  <option value={32768}>32K (RAG)</option>
                  <option value={50000}>50K</option>
                  <option value={131072}>128K (long context)</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">GPUs (tensor parallel)</label>
                <select
                  value={gpuCount}
                  onChange={(e) => setGpuCount(Number(e.target.value))}
                  className="bg-muted border border-border rounded-md px-3 py-1.5 text-sm"
                >
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                  <option value={4}>4×</option>
                  <option value={8}>8×</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* H100 results */}
              <div className={`rounded-lg border p-4 ${h100Result.vramFits ? "border-border/60" : "border-red-500/40 bg-red-500/5"}`}>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm font-bold">H100 SXM5</span>
                  <Badge variant="outline" className="text-[9px]">80GB</Badge>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Decode tok/s</span><span className="font-mono font-semibold">{fmtTokens(h100Result.decodeTokensPerSec)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Aggregate tok/s</span><span className="font-mono">{fmtTokens(h100Result.aggregateTokensPerSec)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">TTFT</span><span className="font-mono">{fmtMs(h100Result.ttftMs)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">VRAM needed</span><span className={`font-mono ${h100Result.vramFits ? "" : "text-red-500"}`}>{fmtBytes(h100Result.totalVramNeededGb)} / {H100!.vramGb * gpuCount}GB</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Cost/M tokens</span><span className="font-mono text-emerald-500">{fmtMoney(h100Result.costPerMillionOutputTokens)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Fits?</span><span className={h100Result.vramFits ? "text-emerald-500" : "text-red-500"}>{h100Result.vramFits ? "✓ Yes" : "✗ No — over budget"}</span></div>
                </div>
                <Button asChild variant="outline" size="sm" className="w-full mt-3 text-xs">
                  <a href={shareH100} target="_blank" rel="noopener noreferrer">Open in calculator →</a>
                </Button>
              </div>

              {/* H200 results */}
              <div className={`rounded-lg border p-4 ${h200Result.vramFits ? "border-emerald-500/40 bg-emerald-500/5" : "border-red-500/40"}`}>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm font-bold">H200 SXM5</span>
                  <Badge variant="outline" className="text-[9px] bg-emerald-500/10">141GB</Badge>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Decode tok/s</span><span className="font-mono font-semibold">{fmtTokens(h200Result.decodeTokensPerSec)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Aggregate tok/s</span><span className="font-mono">{fmtTokens(h200Result.aggregateTokensPerSec)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">TTFT</span><span className="font-mono">{fmtMs(h200Result.ttftMs)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">VRAM needed</span><span className="font-mono">{fmtBytes(h200Result.totalVramNeededGb)} / {H200!.vramGb * gpuCount}GB</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Cost/M tokens</span><span className="font-mono text-emerald-500">{fmtMoney(h200Result.costPerMillionOutputTokens)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Fits?</span><span className={h200Result.vramFits ? "text-emerald-500" : "text-red-500"}>{h200Result.vramFits ? "✓ Yes" : "✗ No"}</span></div>
                </div>
                <Button asChild variant="outline" size="sm" className="w-full mt-3 text-xs">
                  <a href={shareH200} target="_blank" rel="noopener noreferrer">Open in calculator →</a>
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Throughput chart */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><Zap className="size-4 text-emerald-500" /> Throughput by context length</CardTitle>
            <CardDescription className="text-xs">Llama 3.3 70B FP8, {gpuCount}× GPU, batch 8, continuous batching 2.0×</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                  <XAxis dataKey="context" stroke="var(--muted-foreground)" fontSize={11} />
                  <YAxis stroke="var(--muted-foreground)" fontSize={11} tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                  <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)", borderRadius: "6px", fontSize: "12px" }} />
                  <BarChartBar dataKey="h100" name="H100" fill="#3b82f6" radius={[3,3,0,0]} />
                  <BarChartBar dataKey="h200" name="H200" fill="#10b981" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Key findings */}
        <div className="prose dark:prose-invert max-w-none mb-8">
          <h2 className="text-xl font-semibold mb-4">Key findings</h2>
          <div className="space-y-4 text-sm leading-relaxed">
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">1.</span>
              <p><strong>H200 doubles H100 concurrency at every context length</strong> due to 76% more VRAM (141 vs 80 GB). The free VRAM after loading model weights is the bottleneck — H200 has 70 GB free vs H100&apos;s 9 GB for Llama 70B FP8.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">2.</span>
              <p><strong>H100 can&apos;t serve 32K+ context for Llama 70B FP8 on a single GPU</strong> — the KV cache alone needs 10+ GB, exceeding the 9 GB free after loading 70.6 GB of weights. H200 handles 32K with room for 6 concurrent users.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">3.</span>
              <p><strong>H200 is 43% faster per-token</strong> (4,800 vs 3,350 GB/s HBM bandwidth) — decode throughput scales linearly with memory bandwidth for memory-bound LLM inference.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">4.</span>
              <p><strong>H100 is 60% cheaper per GPU-hour</strong> (~$2.50 vs ~$4.00) but the cost-per-token is often LOWER on H200 because the higher throughput more than compensates for the higher hourly price.</p>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="prose dark:prose-invert max-w-none mb-8">
          <h2 className="text-xl font-semibold mb-4">FAQ</h2>
          <div className="space-y-4 text-sm">
            <details className="group">
              <summary className="font-medium cursor-pointer">Should I use H100 or H200 for Llama 3.3 70B?</summary>
              <p className="mt-2 text-muted-foreground">For context lengths under 8K, H100 is sufficient and cheaper. For 8K-32K, H200 is strongly preferred (H100 can&apos;t fit 32K on a single GPU). For 128K+, H200 is the minimum — H100 can&apos;t fit it even with TP×2.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">How much faster is H200 than H100 for LLM inference?</summary>
              <p className="mt-2 text-muted-foreground">H200 is ~43% faster per-token decode throughput (4,800 vs 3,350 GB/s HBM bandwidth). For aggregate batched throughput, H200 is 2-8× better than H100 at long context because it has 7× more free VRAM for KV cache.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Is H200 worth the extra cost over H100?</summary>
              <p className="mt-2 text-muted-foreground">At 32K+ context, yes — H100 can&apos;t even fit the workload, so H200 is the only option. At short context (4K), H100 may be more cost-effective. Use the calculator above to find your break-even point.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Can H100 serve 128K context for Llama 70B?</summary>
              <p className="mt-2 text-muted-foreground">No — a single H100 (80 GB) can&apos;t fit Llama 70B FP8 (70.6 GB weights) + 128K KV cache (~42 GB) = 112 GB total. You&apos;d need at least 2× H100 (160 GB) or 1× H200 (141 GB).</p>
            </details>
          </div>
        </div>

        {/* Internal links */}
        <div className="border-t border-border/60 pt-6 mt-8">
          <p className="text-xs text-muted-foreground mb-3">Related comparisons:</p>
          <div className="flex flex-wrap gap-2">
            <Link href="/compare/gguf-q4-k-m-vs-q5-k-m" className="text-xs text-emerald-500 hover:underline">GGUF Q4_K_M vs Q5_K_M →</Link>
            <Link href="/self-host-vs-openai-api" className="text-xs text-emerald-500 hover:underline">Self-host vs OpenAI API →</Link>
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
                { "@type": "Question", name: "Should I use H100 or H200 for Llama 3.3 70B?", acceptedAnswer: { "@type": "Answer", text: "For context under 8K, H100 is sufficient. For 8K-32K+, H200 is strongly preferred or required." } },
                { "@type": "Question", name: "How much faster is H200 than H100?", acceptedAnswer: { "@type": "Answer", text: "H200 is ~43% faster per-token and 2-8× better aggregate throughput at long context due to 76% more VRAM." } },
                { "@type": "Question", name: "Is H200 worth the extra cost?", acceptedAnswer: { "@type": "Answer", text: "At 32K+ context, yes. At short context, H100 may be cheaper. Use the calculator to find your break-even." } },
              ],
            }),
          }}
        />
      </div>
    </div>
  );
}
