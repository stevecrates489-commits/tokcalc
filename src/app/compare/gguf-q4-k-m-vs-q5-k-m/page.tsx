"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, FileText, Gauge, MemoryStick, Sparkles } from "lucide-react";
import {
  calculate,
  fmtTokens,
  fmtBytes,
  fmtMs,
  fmtMoney,
  GPU_MAP,
  MODELS,
  MODEL_MAP,
  QUANT_MAP,
  type Quantization,
} from "@/lib/token-calc";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Bar as BarChartBar,
  Legend,
} from "recharts";
import Link from "next/link";

const MODEL_ID = "llama3-8b";
const QUANT_Q4: Quantization = "gguf-q4km";
const QUANT_Q5: Quantization = "gguf-q5km";

const GPU = GPU_MAP["rtx-4090"];
const model = MODELS.find((m) => m.id === MODEL_ID)!;
const q4 = QUANT_MAP[QUANT_Q4];
const q5 = QUANT_MAP[QUANT_Q5];

function calcForQuant(quant: Quantization, contextTokens: number) {
  return calculate({
    modelId: MODEL_ID,
    gpuId: "rtx-4090",
    quantization: quant,
    numGpus: 1,
    batchSize: 1,
    promptTokens: contextTokens,
    outputTokens: 256,
  });
}

const SPEC_ROWS = [
  { label: "Bits per weight (bpw)", q4: "~4.8 bpw", q5: "~5.7 bpw", winner: "q5" },
  { label: "Model file size (8B params)", q4: `${(8 * q4.bytesPerParam).toFixed(1)} GB`, q5: `${(8 * q5.bytesPerParam).toFixed(1)} GB`, winner: "q4" },
  { label: "VRAM after weights (24GB GPU)", q4: `${(24 - 8 * q4.bytesPerParam).toFixed(1)} GB free`, q5: `${(24 - 8 * q5.bytesPerParam).toFixed(1)} GB free`, winner: "q4" },
  { label: "Dequantization overhead", q4: "~10%", q5: "~8%", winner: "q5" },
  { label: "Quality (chat)", q4: "Excellent", q5: "Near-lossless", winner: "q5" },
  { label: "Quality (code/math)", q4: "Minor artifacts", q5: "Lossless-grade", winner: "q5" },
  { label: "Recommended use case", q4: "General chat on consumer GPU", q5: "Quality-critical local apps", winner: "tie" },
  { label: "llama.cpp default", q4: "Yes (recommended)", q5: "Optional", winner: "q4" },
];

export default function GgufQ4vsQ5Page() {
  const [context, setContext] = useState(8192);

  const q4Result = useMemo(() => calcForQuant(QUANT_Q4, context), [context]);
  const q5Result = useMemo(() => calcForQuant(QUANT_Q5, context), [context]);

  const chartData = useMemo(() => {
    const ctxs = [512, 2048, 8192, 16384, 32768, 65536];
    return ctxs.map((ctx) => ({
      context: ctx >= 1000 ? `${ctx / 1000}K` : `${ctx}`,
      q4_km: Math.max(0, Math.round(calcForQuant(QUANT_Q4, ctx).aggregateTokensPerSec)),
      q5_km: Math.max(0, Math.round(calcForQuant(QUANT_Q5, ctx).aggregateTokensPerSec)),
    }));
  }, []);

  const chartVram = useMemo(() => {
    const ctxs = [512, 2048, 8192, 16384, 32768, 65536];
    return ctxs.map((ctx) => {
      const r4 = calcForQuant(QUANT_Q4, ctx);
      const r5 = calcForQuant(QUANT_Q5, ctx);
      return {
        context: ctx >= 1000 ? `${ctx / 1000}K` : `${ctx}`,
        q4_km: +r4.totalVramNeededGb.toFixed(2),
        q5_km: +r5.totalVramNeededGb.toFixed(2),
      };
    });
  }, []);

  const shareQ4 = `https://tokcalc.vercel.app/#t=calculator&m=${MODEL_ID}&g=rtx-4090&q=${QUANT_Q4}&n=1&b=1&p=${context}&o=256`;
  const shareQ5 = `https://tokcalc.vercel.app/#t=calculator&m=${MODEL_ID}&g=rtx-4090&q=${QUANT_Q5}&n=1&b=1&p=${context}&o=256`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mb-6">
          <ArrowLeft className="size-4" />
          Back to tokcalc
        </Link>

        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
          GGUF Q4_K_M vs Q5_K_M: <span className="text-emerald-500">VRAM, Throughput & Quality for Local LLM Inference</span>
        </h1>
        <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8">
          Side-by-side comparison of the two most popular llama.cpp quantization formats for
          Llama 3 8B on a consumer RTX 4090 (24GB). See how file size, VRAM headroom, decode
          throughput, and quality differ — with transparent formulas, not hand-wavy claims.
        </p>

        {/* Spec comparison */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="size-4 text-emerald-500" />
              Specification comparison
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Spec</th>
                  <th className="text-left px-4 py-2">Q4_K_M</th>
                  <th className="text-left px-4 py-2">Q5_K_M</th>
                  <th className="text-center px-4 py-2">Winner</th>
                </tr>
              </thead>
              <tbody>
                {SPEC_ROWS.map((row, i) => (
                  <tr key={i} className="border-t border-border/40">
                    <td className="px-4 py-2 font-medium">{row.label}</td>
                    <td className="px-4 py-2 font-mono text-xs">{row.q4}</td>
                    <td className="px-4 py-2 font-mono text-xs">{row.q5}</td>
                    <td className="px-4 py-2 text-center">
                      {row.winner === "q5" && <Badge className="text-[9px] bg-emerald-500">Q5_K_M</Badge>}
                      {row.winner === "q4" && <Badge className="text-[9px] bg-blue-500">Q4_K_M</Badge>}
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
              Live comparison — adjust context length
            </CardTitle>
            <CardDescription className="text-xs">
              All numbers computed by tokcalc&apos;s open-source formulas on a single RTX 4090 (24 GB).
              Click a config to open the full calculator.
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
                  <option value={512}>512 (chat)</option>
                  <option value={2048}>2K (RAG)</option>
                  <option value={8192}>8K</option>
                  <option value={16384}>16K</option>
                  <option value={32768}>32K (long context)</option>
                  <option value={65536}>64K (max)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Q4_K_M results */}
              <div className={`rounded-lg border p-4 ${q4Result.vramFits ? "border-border/60" : "border-red-500/40 bg-red-500/5"}`}>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm font-bold">Q4_K_M</span>
                  <Badge variant="outline" className="text-[9px]">~4.8 bpw</Badge>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Decode tok/s</span><span className="font-mono font-semibold">{fmtTokens(q4Result.decodeTokensPerSec)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">TTFT</span><span className="font-mono">{fmtMs(q4Result.ttftMs)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">ITL</span><span className="font-mono">{fmtMs(q4Result.itlMs)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">VRAM needed</span><span className={`font-mono ${q4Result.vramFits ? "" : "text-red-500"}`}>{fmtBytes(q4Result.totalVramNeededGb)} / 24GB</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Free VRAM</span><span className="font-mono text-emerald-500">{fmtBytes(24 - q4Result.totalVramNeededGb)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Fits?</span><span className={q4Result.vramFits ? "text-emerald-500" : "text-red-500"}>{q4Result.vramFits ? "✓ Yes" : "✗ Over budget"}</span></div>
                </div>
                <Button asChild variant="outline" size="sm" className="w-full mt-3 text-xs">
                  <a href={shareQ4} target="_blank" rel="noopener noreferrer">Open in calculator →</a>
                </Button>
              </div>

              {/* Q5_K_M results */}
              <div className={`rounded-lg border p-4 ${q5Result.vramFits ? "border-emerald-500/40 bg-emerald-500/5" : "border-red-500/40"}`}>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm font-bold">Q5_K_M</span>
                  <Badge variant="outline" className="text-[9px] bg-emerald-500/10">~5.7 bpw</Badge>
                </div>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Decode tok/s</span><span className="font-mono font-semibold">{fmtTokens(q5Result.decodeTokensPerSec)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">TTFT</span><span className="font-mono">{fmtMs(q5Result.ttftMs)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">ITL</span><span className="font-mono">{fmtMs(q5Result.itlMs)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">VRAM needed</span><span className="font-mono">{fmtBytes(q5Result.totalVramNeededGb)} / 24GB</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Free VRAM</span><span className="font-mono text-emerald-500">{fmtBytes(24 - q5Result.totalVramNeededGb)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Fits?</span><span className={q5Result.vramFits ? "text-emerald-500" : "text-red-500"}>{q5Result.vramFits ? "✓ Yes" : "✗ Over budget"}</span></div>
                </div>
                <Button asChild variant="outline" size="sm" className="w-full mt-3 text-xs">
                  <a href={shareQ5} target="_blank" rel="noopener noreferrer">Open in calculator →</a>
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Throughput chart */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><Gauge className="size-4 text-emerald-500" /> Decode throughput by context length</CardTitle>
            <CardDescription className="text-xs">Llama 3 8B on 1× RTX 4090, batch 1, single-stream decode</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                  <XAxis dataKey="context" stroke="var(--muted-foreground)" fontSize={11} />
                  <YAxis stroke="var(--muted-foreground)" fontSize={11} tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                  <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)", borderRadius: "6px", fontSize: "12px" }} />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <BarChartBar dataKey="q4_km" name="Q4_K_M" fill="#3b82f6" radius={[3,3,0,0]} />
                  <BarChartBar dataKey="q5_km" name="Q5_K_M" fill="#10b981" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* VRAM chart */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2"><MemoryStick className="size-4 text-emerald-500" /> VRAM usage by context length</CardTitle>
            <CardDescription className="text-xs">Model weights + KV cache. RTX 4090 has 24 GB total VRAM.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartVram} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
                  <XAxis dataKey="context" stroke="var(--muted-foreground)" fontSize={11} />
                  <YAxis stroke="var(--muted-foreground)" fontSize={11} tickFormatter={(v) => `${v}GB`} />
                  <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)", borderRadius: "6px", fontSize: "12px" }} formatter={(v: number) => `${v} GB`} />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  <BarChartBar dataKey="q4_km" name="Q4_K_M" fill="#3b82f6" radius={[3,3,0,0]} />
                  <BarChartBar dataKey="q5_km" name="Q5_K_M" fill="#10b981" radius={[3,3,0,0]} />
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
              <p><strong>Q4_K_M is the recommended default for local inference</strong> — it&apos;s llama.cpp&apos;s own recommended sweet spot. For an 8B model, the file is ~4.4 GB vs Q5_K_M&apos;s ~5.4 GB, leaving ~1 GB more VRAM for KV cache.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">2.</span>
              <p><strong>Q5_K_M quality is near-lossless for chat</strong> but the difference matters for code, math, and long-context reasoning. If your workflow involves structured output (JSON mode) or multi-step reasoning, Q5_K_M is worth the ~1 GB extra.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">3.</span>
              <p><strong>Q4_K_M is slightly faster decode</strong> because the smaller weights mean less memory bandwidth pressure on memory-bound decode. The difference is 5-10% — measurable but not dramatic for an 8B model.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-500 font-bold">4.</span>
              <p><strong>Context length is where Q4 wins</strong> — at 32K context on a 24 GB GPU, Q5_K_M runs out of VRAM about 4-8K tokens earlier than Q4_K_M. For long-context RAG/agent work, Q4_K_M lets you fit more.</p>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="prose dark:prose-invert max-w-none mb-8">
          <h2 className="text-xl font-semibold mb-4">FAQ</h2>
          <div className="space-y-4 text-sm">
            <details className="group">
              <summary className="font-medium cursor-pointer">Which GGUF quant should I pick for local Llama 3 8B?</summary>
              <p className="mt-2 text-muted-foreground">Start with Q4_K_M — it&apos;s the llama.cpp default and works on virtually all consumer GPUs (8GB+). Upgrade to Q5_K_M only if you notice quality issues with code/math/structured output, and your GPU has the VRAM headroom.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Is Q5_K_M worth the extra ~1 GB over Q4_K_M?</summary>
              <p className="mt-2 text-muted-foreground">For chat and general Q&amp;A, no — Q4_K_M is indistinguishable. For coding, math, long-context reasoning, or JSON-mode structured output, yes — Q5_K_M&apos;s extra precision reduces artifacts. Test on your specific workload.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">How much VRAM does Llama 3 8B need?</summary>
              <p className="mt-2 text-muted-foreground">At Q4_K_M: ~4.4 GB weights + KV cache (varies with context). At 8K context: ~6 GB total. At 32K context: ~10 GB. At 64K context: ~14 GB. A 16 GB GPU (RTX 4060 Ti) comfortably handles 32K; a 24 GB GPU (RTX 4090) handles 64K+.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">Can I switch between Q4 and Q5 at runtime?</summary>
              <p className="mt-2 text-muted-foreground">Yes — both are different .gguf files. llama.cpp loads whichever file you point it at. You can keep both on disk (~10 GB total) and swap based on workload: Q4 for chat, Q5 for code review.</p>
            </details>
            <details className="group">
              <summary className="font-medium cursor-pointer">What about Q3_K_M, Q6_K, or Q8_0?</summary>
              <p className="mt-2 text-muted-foreground">Q3_K_M (~3.9 bpw) is too lossy for production — only use when VRAM is critically short. Q6_K (~6.6 bpw) and Q8_0 (~8.5 bpw) are near-lossless but 60-100% larger than Q4_K_M with diminishing quality returns. Stick to Q4_K_M or Q5_K_M for the sweet spot.</p>
            </details>
          </div>
        </div>

        {/* Internal links */}
        <div className="border-t border-border/60 pt-6 mt-8">
          <p className="text-xs text-muted-foreground mb-3">Related comparisons:</p>
          <div className="flex flex-wrap gap-2">
            <Link href="/compare/h100-vs-h200" className="text-xs text-emerald-500 hover:underline">H100 vs H200 (datacenter) →</Link>
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
                { "@type": "Question", name: "Which GGUF quant should I pick for Llama 3 8B?", acceptedAnswer: { "@type": "Answer", text: "Start with Q4_K_M — it's the llama.cpp default and works on virtually all consumer GPUs. Upgrade to Q5_K_M for code/math/structured output if you have VRAM headroom." } },
                { "@type": "Question", name: "Is Q5_K_M worth the extra VRAM?", acceptedAnswer: { "@type": "Answer", text: "For chat, no. For code, math, JSON mode, or long-context reasoning, yes." } },
                { "@type": "Question", name: "How much VRAM does Llama 3 8B need?", acceptedAnswer: { "@type": "Answer", text: "Q4_K_M at 8K context: ~6 GB. At 32K: ~10 GB. At 64K: ~14 GB. 16GB GPU is fine for most use; 24GB allows 64K+ context." } },
              ],
            }),
          }}
        />
      </div>
    </div>
  );
}
