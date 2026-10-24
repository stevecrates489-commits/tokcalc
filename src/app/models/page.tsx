import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, Cpu, Layers, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ContentShell, ContentFooter } from "@/components/content/page-shell";
import {
  MODEL_PAGE_IDS,
  MODEL_CONTENT_MAP,
  defaultQuantFor,
} from "@/lib/content/model-content";
import { MODELS, MODEL_MAP, fmtMoney, fmtTokens, fmtContext } from "@/lib/token-calc";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://tokcalc.vercel.app";

export const metadata: Metadata = {
  title: "All LLMs: VRAM Requirements, GPU Sizing & Serving Cost",
  description:
    "Every model in tokcalc's catalog with its VRAM requirement, the GPUs that can run it, throughput, and cost per million tokens — computed from open-source formulas, no marketing multipliers.",
  keywords: [
    "LLM VRAM requirements",
    "LLM GPU requirements",
    "how many GPUs for LLM",
    "LLM serving cost",
    "model to GPU mapping",
    "open weights models",
  ],
  openGraph: {
    title: "All LLMs: VRAM Requirements, GPU Sizing & Serving Cost",
    description:
      "Every model in tokcalc's catalog with VRAM, GPU fit, throughput, and cost per million tokens.",
    type: "website",
    url: `${SITE_URL}/models`,
  },
  alternates: { canonical: `${SITE_URL}/models` },
};

export default function ModelsHubPage() {
  const rows = MODEL_PAGE_IDS.map((id) => MODEL_CONTENT_MAP[id])
    .filter(Boolean)
    .sort((a, b) => b.model.paramsB - a.model.paramsB);

  return (
    <ContentShell>
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
        Every LLM, sized for GPUs:{" "}
        <span className="text-emerald-500">VRAM, fit, throughput & cost</span>
      </h1>
      <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8 max-w-3xl">
        {rows.length} models from the tokcalc catalog. Each row shows the VRAM the weights need at
        the recommended quantization, whether it fits on a single GPU, and the cheapest configuration
        that serves it — all computed by the same formulas the calculator uses.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8">
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Layers className="size-3" />
              Models
            </div>
            <div className="text-xl font-bold font-mono">{rows.length}</div>
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Cpu className="size-3" />
              Fit on one GPU
            </div>
            <div className="text-xl font-bold font-mono">
              {rows.filter((r) => r.singleGpuOptions.length > 0).length}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">at the recommended quant</div>
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Zap className="size-3" />
              GPU types covered
            </div>
            <div className="text-xl font-bold font-mono">31</div>
            <div className="text-[11px] text-muted-foreground mt-1">
              NVIDIA, AMD, Intel, TPU, LPU, WSE, Apple
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base">All models</CardTitle>
          <CardDescription className="text-xs">
            Sorted by parameter count. &ldquo;Fits on 1 GPU&rdquo; is evaluated at the recommended
            quantization.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          <table className="w-full text-sm min-w-[720px]">
            <thead className="bg-muted/40 text-[10px] uppercase">
              <tr>
                <th className="text-left px-4 py-2">Model</th>
                <th className="text-right px-4 py-2">Params</th>
                <th className="text-right px-4 py-2">VRAM (weights)</th>
                <th className="text-left px-4 py-2">Quant</th>
                <th className="text-center px-4 py-2">Fits on 1 GPU</th>
                <th className="text-left px-4 py-2">Cheapest config</th>
                <th className="text-right px-4 py-2">Cost / 1M tok</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const defRow = row.quantRows.find(
                  (r) => r.quant.id === row.defaultQuant,
                )!;
                const cheapest = row.cheapest;
                return (
                  <tr key={row.model.id} className="border-t border-border/40">
                    <td className="px-4 py-2">
                      <Link
                        href={`/models/${row.model.id}`}
                        className="font-medium hover:underline underline-offset-2"
                      >
                        {row.model.name}
                      </Link>
                      {row.model.isMoE && (
                        <Badge variant="outline" className="ml-2 text-[9px]">
                          MoE
                        </Badge>
                      )}
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {row.model.paramsB.toFixed(row.model.paramsB % 1 === 0 ? 0 : 2)}B
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {defRow.weightsGb.toFixed(1)} GB
                    </td>
                    <td className="px-4 py-2 font-mono text-xs">{defRow.quant.label}</td>
                    <td className="px-4 py-2 text-center">
                      {row.singleGpuOptions.length > 0 ? (
                        <span className="text-emerald-500">✓</span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs">
                      {cheapest ? `${cheapest.gpuCount}× ${cheapest.gpu.name}` : "—"}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-emerald-500">
                      {cheapest ? fmtMoney(cheapest.costPerMillion) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground mb-8">
        Looking for a GPU-first view?{" "}
        <Link href="/gpus" className="underline underline-offset-2 hover:text-foreground">
          See all GPUs and what they can serve
        </Link>
        .
      </p>

      <ContentFooter />
    </ContentShell>
  );
}
