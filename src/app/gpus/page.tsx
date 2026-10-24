import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight, Cpu, MemoryStick, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ContentShell, ContentFooter } from "@/components/content/page-shell";
import { GPU_CONTENT_MAP, GPU_PAGE_IDS } from "@/lib/content/gpu-content";
import { GPUS, fmtMoney, fmtTokens } from "@/lib/token-calc";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://tokcalc.vercel.app";

export const metadata: Metadata = {
  title: "All GPUs for LLM Inference: VRAM, Bandwidth, Throughput & Cost",
  description:
    "Every accelerator in tokcalc's catalog — NVIDIA, AMD, Intel, TPU, LPU, WSE, and Apple Silicon — with VRAM, memory bandwidth, what it can serve, and cost per million tokens.",
  keywords: [
    "LLM inference GPU",
    "GPU comparison for LLM",
    "GPU VRAM for LLM",
    "H100 vs H200",
    "A100 vs H100",
    "MI300X for LLM",
    "GPU tokens per second",
    "cheapest GPU for LLM",
  ],
  openGraph: {
    title: "All GPUs for LLM Inference: VRAM, Bandwidth, Throughput & Cost",
    description:
      "Every accelerator in tokcalc's catalog with VRAM, bandwidth, what it can serve, and cost per million tokens.",
    type: "website",
    url: `${SITE_URL}/gpus`,
  },
  alternates: { canonical: `${SITE_URL}/gpus` },
};

/** Category order — mirrors how buyers think about accelerators. */
const CATEGORY_ORDER = [
  "datacenter",
  "workstation",
  "consumer",
  "mac",
  "tpu",
  "lpu",
  "wse",
  "legacy",
];

const CATEGORY_LABEL: Record<string, string> = {
  datacenter: "Datacenter",
  workstation: "Workstation",
  consumer: "Consumer",
  mac: "Apple Silicon",
  tpu: "TPU",
  lpu: "LPU",
  wse: "Wafer-scale",
  legacy: "Legacy",
  rdu: "RDM",
};

export default function GpusHubPage() {
  const grouped = CATEGORY_ORDER.map((cat) => ({
    cat,
    label: CATEGORY_LABEL[cat] ?? cat,
    gpus: GPUS.filter((g) => g.category === cat),
  })).filter((group) => group.gpus.length > 0);

  return (
    <ContentShell>
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
        Every accelerator for LLM serving:{" "}
        <span className="text-emerald-500">VRAM, bandwidth, throughput & cost</span>
      </h1>
      <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8 max-w-3xl">
        {GPUS.length} accelerators across NVIDIA, AMD, Intel, Google TPU, Groq LPU, Cerebras
        wafer-scale, and Apple Silicon. Each page shows what the card can actually serve, how fast,
        and what it costs per million tokens.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-8">
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Cpu className="size-3" />
              Accelerators
            </div>
            <div className="text-xl font-bold font-mono">{GPUS.length}</div>
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <MemoryStick className="size-3" />
              Largest VRAM
            </div>
            <div className="text-xl font-bold font-mono">
              {Math.max(...GPUS.map((g) => g.vramGb))} GB
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {GPUS.reduce((best, g) => (g.vramGb > best.vramGb ? g : best)).name}
            </div>
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Zap className="size-3" />
              Fastest bandwidth
            </div>
            <div className="text-xl font-bold font-mono">
              {Math.max(...GPUS.map((g) => g.memBandwidthGbps)).toLocaleString()}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">GB/s</div>
          </CardContent>
        </Card>
      </div>

      {grouped.map((group) => (
        <div key={group.cat} className="mb-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-3">
            {group.label} ({group.gpus.length})
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {group.gpus.map((gpu) => {
              const content = GPU_CONTENT_MAP[gpu.id];
              const fits = content?.singleGpuModels.length ?? 0;
              const ref = content?.reference;
              return (
                <Link
                  key={gpu.id}
                  href={`/gpus/${gpu.id}`}
                  className="group rounded-lg border border-border/60 p-4 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="font-semibold text-sm group-hover:underline underline-offset-2">
                      {gpu.name}
                    </span>
                    <ArrowRight className="size-3.5 text-muted-foreground shrink-0 mt-0.5" />
                  </div>
                  <div className="flex flex-wrap gap-1.5 mb-2.5">
                    <Badge variant="outline" className="text-[9px]">
                      {gpu.vramGb} GB
                    </Badge>
                    <Badge variant="outline" className="text-[9px]">
                      {gpu.memBandwidthGbps.toLocaleString()} GB/s
                    </Badge>
                    {gpu.usdPerHour != null && (
                      <Badge variant="outline" className="text-[9px]">
                        ${gpu.usdPerHour}/hr
                      </Badge>
                    )}
                  </div>
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    {fits > 0 ? (
                      <>
                        Runs {fits} model{fits === 1 ? "" : "s"} on one card
                        {ref?.fits && ref.result.aggregateTokensPerSec > 0
                          ? ` · ${fmtTokens(ref.result.aggregateTokensPerSec)} tok/s on ${ref.model.name}`
                          : ""}
                        {ref?.fits && ref.result.costPerMillionOutputTokens > 0
                          ? ` · ${fmtMoney(ref.result.costPerMillionOutputTokens)}/1M tok`
                          : ""}
                      </>
                    ) : (
                      <>Needs multi-GPU configurations for most models</>
                    )}
                  </p>
                </Link>
              );
            })}
          </div>
        </div>
      ))}

      <p className="text-xs text-muted-foreground mb-8">
        Looking for the model side of the catalog?{" "}
        <Link href="/models" className="underline underline-offset-2 hover:text-foreground">
          See all models and their VRAM requirements
        </Link>
        .
      </p>

      <ContentFooter />
    </ContentShell>
  );
}
