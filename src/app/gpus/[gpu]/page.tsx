import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowRight, Cpu, Gauge, MemoryStick, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ContentShell, ContentFooter, FaqJsonLd } from "@/components/content/page-shell";
import { GPU_CONTENT_MAP, GPU_PAGE_IDS } from "@/lib/content/gpu-content";
import { fmtTokens, fmtMoney, fmtBytes, fmtMs, fmtContext } from "@/lib/token-calc";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://tokcalc.vercel.app";

export function generateStaticParams() {
  return GPU_PAGE_IDS.map((gpu) => ({ gpu }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ gpu: string }>;
}): Promise<Metadata> {
  const { gpu: gpuId } = await params;
  const content = GPU_CONTENT_MAP[gpuId];
  if (!content) return {};

  const { gpu, singleGpuModels, reference } = content;
  const canonical = `${SITE_URL}/gpus/${gpu.id}`;

  const biggest = singleGpuModels[singleGpuModels.length - 1];
  const fitSentence = singleGpuModels.length
    ? `Runs ${singleGpuModels.length} models on a single card, up to ${biggest.model.name}.`
    : "Needs multi-GPU configurations for most models.";

  const title = `${gpu.name}: Specs, VRAM, Throughput & Cost for LLM Serving`;
  const description = `${gpu.name} — ${gpu.vramGb} GB VRAM, ${gpu.memBandwidthGbps.toLocaleString()} GB/s memory bandwidth, ${gpu.usdPerHour ? `$${gpu.usdPerHour}/hr` : "quote required"}. ${fitSentence} Full LLM serving benchmarks and cost per million tokens, computed from open-source formulas.`;

  return {
    title,
    description,
    keywords: [
      `${gpu.name} specs`,
      `${gpu.name} LLM`,
      `${gpu.name} VRAM`,
      `${gpu.name} inference`,
      `${gpu.name} tokens per second`,
      `${gpu.name} serving cost`,
      `${gpu.id}`,
      gpu.vendor === "NVIDIA" ? `${gpu.name} for LLM serving` : `${gpu.name} accelerator`,
    ],
    openGraph: {
      title: `${gpu.name}: Specs, VRAM, Throughput & Cost`,
      description,
      type: "article",
      url: canonical,
    },
    twitter: { card: "summary_large_image", title, description },
    alternates: { canonical },
  };
}

export default async function GpuPage({
  params,
}: {
  params: Promise<{ gpu: string }>;
}) {
  const { gpu: gpuId } = await params;
  const content = GPU_CONTENT_MAP[gpuId];
  if (!content) notFound();

  const { gpu, singleGpuModels, reference, comparison, faqs } = content;
  const biggest = singleGpuModels[singleGpuModels.length - 1];

  // Top comparison rows for the cross-GPU chart/table (this GPU highlighted).
  const comparisonRows = comparison.slice(0, 12);
  const gpuRank = comparison.findIndex((c) => c.gpu.id === gpu.id);

  return (
    <ContentShell breadcrumb={[{ label: "GPUs", href: "/gpus" }]}>
      <FaqJsonLd faqs={faqs} />

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge variant="outline" className="text-[10px]">
          {gpu.vendor}
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {gpu.category}
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {gpu.year}
        </Badge>
        {gpu.usdPerHour != null && (
          <Badge variant="outline" className="text-[10px]">
            ${gpu.usdPerHour}/hr
          </Badge>
        )}
      </div>

      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
        {gpu.name}:{" "}
        <span className="text-emerald-500">Specs, VRAM, Throughput & Cost</span>
      </h1>
      <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8 max-w-3xl">
        {gpu.name} has {gpu.vramGb} GB of VRAM and{" "}
        {gpu.memBandwidthGbps.toLocaleString()} GB/s of memory bandwidth
        {gpu.nvlinkGbps ? `, plus ${gpu.nvlinkGbps} GB/s of NVLink` : ""}.{" "}
        {singleGpuModels.length > 0 ? (
          <>
            It runs {singleGpuModels.length} models from the catalog on a single card — up to{" "}
            {biggest.model.name} — and serves{" "}
            {reference.fits ? (
              <>
                {reference.model.name} at {fmtTokens(reference.result.aggregateTokensPerSec)}{" "}
                tok/s aggregate
              </>
            ) : (
              <>reference workloads only in multi-GPU configurations</>
            )}
            .
          </>
        ) : (
          <>
            Most models in the catalog need multi-GPU configurations on this accelerator.
          </>
        )}{" "}
        All figures are computed by tokcalc&apos;s open-source capacity model.
      </p>

      {/* ---- Spec tiles ---- */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <MemoryStick className="size-3" />
              VRAM
            </div>
            <div className="text-xl font-bold font-mono">{gpu.vramGb} GB</div>
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Zap className="size-3" />
              Memory bandwidth
            </div>
            <div className="text-xl font-bold font-mono">
              {gpu.memBandwidthGbps.toLocaleString()}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">GB/s</div>
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Cpu className="size-3" />
              FP16/BF16
            </div>
            <div className="text-xl font-bold font-mono">
              {gpu.flopsTflops ? `${gpu.flopsTflops} TF` : "—"}
            </div>
            {gpu.flopsTflops == null && (
              <div className="text-[11px] text-muted-foreground mt-1">not publicly reported</div>
            )}
          </CardContent>
        </Card>
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Gauge className="size-3" />
              Price
            </div>
            <div className="text-xl font-bold font-mono">
              {gpu.usdPerHour != null ? `$${gpu.usdPerHour}` : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {gpu.usdPerHour != null ? "per GPU-hour (typical)" : "quote required"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ---- Spec table ---- */}
      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base">Hardware specifications</CardTitle>
          {gpu.sourceUrl && (
            <CardDescription className="text-xs">
              <a
                href={gpu.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2"
              >
                Vendor source →
              </a>
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <tbody>
              {[
                ["Vendor", gpu.vendor],
                ["Category", gpu.category],
                ["VRAM", `${gpu.vramGb} GB`],
                ["Memory bandwidth", `${gpu.memBandwidthGbps.toLocaleString()} GB/s`],
                [
                  "FP16/BF16 dense compute",
                  gpu.flopsTflops ? `${gpu.flopsTflops} TFLOPS` : "not publicly reported",
                ],
                ["NVLink bandwidth", gpu.nvlinkGbps ? `${gpu.nvlinkGbps} GB/s` : "none"],
                ["Release year", String(gpu.year)],
                [
                  "Typical cloud price",
                  gpu.usdPerHour != null ? `$${gpu.usdPerHour}/hr` : "quote required",
                ],
                ...(gpu.note ? [["Note", gpu.note] as [string, string]] : []),
              ].map(([label, value]) => (
                <tr key={label} className="border-b border-border/30 last:border-0">
                  <td className="px-4 py-2 text-muted-foreground">{label}</td>
                  <td className="px-4 py-2 font-mono text-xs">{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* ---- Models that fit ---- */}
      <Card className="border-emerald-500/30 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="size-4 text-emerald-500" />
            Models that fit on a single {gpu.name}
          </CardTitle>
          <CardDescription className="text-xs">
            At each model&apos;s recommended quantization, {fmtContext(8192)} context and 8
            concurrent users. Cost is per 1M output tokens.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          {singleGpuModels.length === 0 ? (
            <p className="p-4 text-sm text-muted-foreground">
              No model in the tokcalc catalog fits on a single {gpu.name} at its recommended
              quantization. Use multi-GPU tensor parallelism or context parallelism.
            </p>
          ) : gpu.category === "wse" || gpu.category === "lpu" ? (
            /* Wafer-scale and LPU parts are not per-card comparable: a CS-3 is a
               whole wafer, and an LPU is SRAM-based with a completely different
               memory hierarchy. Listing "models that fit on one card" for them
               would be a category error, so we say so instead. */
            <p className="p-4 text-sm text-muted-foreground">
              {gpu.name} is a {gpu.category === "wse" ? "wafer-scale" : "LPU"} system, so
              &ldquo;fits on one card&rdquo; is not a meaningful comparison — the whole system is
              the unit of deployment. Use the cited benchmarks for this platform rather than the
              per-GPU numbers on this page.
            </p>
          ) : (
            <table className="w-full text-sm min-w-[560px]">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Model</th>
                  <th className="text-right px-4 py-2">Params</th>
                  <th className="text-left px-4 py-2">Quant</th>
                  <th className="text-right px-4 py-2">VRAM needed</th>
                  <th className="text-right px-4 py-2">Aggregate tok/s</th>
                  <th className="text-right px-4 py-2">Cost / 1M tok</th>
                </tr>
              </thead>
              <tbody>
                {singleGpuModels.map((m) => (
                  <tr key={m.model.id} className="border-t border-border/40">
                    <td className="px-4 py-2">
                      <Link
                        href={`/models/${m.model.id}`}
                        className="font-medium hover:underline underline-offset-2"
                      >
                        {m.model.name}
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {m.model.paramsB.toFixed(m.model.paramsB % 1 === 0 ? 0 : 2)}B
                    </td>
                    <td className="px-4 py-2 font-mono text-xs">{m.quant}</td>
                    <td className="px-4 py-2 text-right font-mono">
                      {fmtBytes(m.result.totalVramNeededGb)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {fmtTokens(m.aggregateTokensPerSec)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-emerald-500">
                      {fmtMoney(m.costPerMillion)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>

      {/* ---- Cross-GPU comparison ---- */}
      {comparisonRows.length > 0 && (
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="size-4 text-emerald-500" />
              {reference.model.name} across every accelerator
            </CardTitle>
            <CardDescription className="text-xs">
              Same model ({reference.quant}), same workload, every GPU in the catalog.{" "}
              {gpuRank >= 0 && (
                <>
                  {gpu.name} ranks <strong className="text-foreground">#{gpuRank + 1}</strong> of{" "}
                  {comparison.length} by aggregate throughput.
                </>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0 overflow-x-auto">
            <table className="w-full text-sm min-w-[560px]">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">GPU</th>
                  <th className="text-right px-4 py-2">Aggregate tok/s</th>
                  <th className="text-right px-4 py-2">Cost / 1M tok</th>
                  <th className="text-right px-4 py-2">$/hr</th>
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map((row) => (
                  <tr
                    key={row.gpu.id}
                    className={`border-t border-border/40 ${
                      row.gpu.id === gpu.id ? "bg-emerald-500/5" : ""
                    }`}
                  >
                    <td className="px-4 py-2">
                      <Link
                        href={`/gpus/${row.gpu.id}`}
                        className={`hover:underline underline-offset-2 ${
                          row.gpu.id === gpu.id ? "font-semibold" : "font-medium"
                        }`}
                      >
                        {row.gpu.name}
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {fmtTokens(row.aggregateTokensPerSec)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-emerald-500">
                      {fmtMoney(row.costPerMillion)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      {row.gpu.usdPerHour != null ? `$${row.gpu.usdPerHour}` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}

      {/* ---- FAQ ---- */}
      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base">Frequently asked questions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {faqs.map((faq) => (
            <div key={faq.q}>
              <h3 className="font-semibold text-sm mb-1.5">{faq.q}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{faq.a}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* ---- CTA ---- */}
      <Card className="border-emerald-500/30 shadow-sm mb-8">
        <CardContent className="p-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h2 className="font-semibold mb-1">Size a real deployment on {gpu.name}</h2>
            <p className="text-xs text-muted-foreground">
              Pick your model, context length, and concurrency — the calculator computes the rest
              with the same formulas.
            </p>
          </div>
          <Button asChild className="shrink-0">
            <a
              href={`https://tokcalc.vercel.app/#t=calculator&g=${gpu.id}&q=${reference.quant}&n=${reference.gpuCount}&b=8&p=8192&o=500&cb=1&cbm=2`}
            >
              Open in calculator <ArrowRight className="size-4 ml-1" />
            </a>
          </Button>
        </CardContent>
      </Card>

      <ContentFooter />
    </ContentShell>
  );
}
