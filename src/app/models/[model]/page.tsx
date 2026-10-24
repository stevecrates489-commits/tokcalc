import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowRight, Cpu, Gauge, Info, MemoryStick, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ContentShell,
  ContentFooter,
  FaqJsonLd,
} from "@/components/content/page-shell";
import {
  MODEL_PAGE_IDS,
  MODEL_CONTENT_MAP,
  REF_BATCH,
  REF_CONTEXT,
  CONTEXT_LADDER,
  computeModelContent,
  concurrencyAtContext,
  multiGpuCaveat,
  hasModelGpuPage,
  attentionSummary,
  type ModelContent,
} from "@/lib/content/model-content";
import { fmtTokens, fmtMoney, fmtBytes, fmtKb, fmtContext } from "@/lib/token-calc";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://tokcalc.vercel.app";

/** Pre-render one page per model in the catalog. */
export function generateStaticParams() {
  return MODEL_PAGE_IDS.map((modelId) => ({ model: modelId }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ model: string }>;
}): Promise<Metadata> {
  const { model: modelId } = await params;
  const content = MODEL_CONTENT_MAP[modelId];
  if (!content) return {};

  const { model, defaultQuant, quantRows, cheapest, fastest, singleGpuOptions } = content;
  const defRow = quantRows.find((r) => r.quant.id === defaultQuant)!;
  const canonical = `${SITE_URL}/models/${model.id}`;

  const title = `${model.name} VRAM Requirements: How Many GPUs & What It Costs`;

  // Build the description from the computed numbers, not from a template that
  // could claim something the catalog no longer supports.
  const fitSentence = singleGpuOptions.length
    ? `Fits on 1× ${singleGpuOptions[0].gpu.name} at ${defRow.quant.label}.`
    : fastest
      ? `Needs ${fastest.gpuCount}× ${fastest.gpu.name} at ${defRow.quant.label}.`
      : `Does not fit in the catalog at ${defRow.quant.label}.`;
  const costSentence = cheapest
    ? `From ${fmtMoney(cheapest.costPerMillion)} per 1M output tokens.`
    : "";

  const description = `${model.name} needs ${defRow.weightsGb.toFixed(0)} GB of VRAM for its weights at ${defRow.quant.label} (${defRow.minVramGb.toFixed(1)} GB with a loaded KV cache). ${fitSentence} ${costSentence} Full VRAM, throughput, latency and cost table computed from tokcalc's open-source formulas.`;

  return {
    title,
    description,
    keywords: [
      `${model.name} VRAM`,
      `${model.name} GPU`,
      `${model.name} requirements`,
      `how many GPUs for ${model.name}`,
      `${model.name} inference`,
      `${model.name} self-host`,
      `${model.name} tokens per second`,
      `${model.name} serving cost`,
      `${model.id} VRAM`,
    ],
    openGraph: {
      title: `${model.name} VRAM Requirements & GPU Sizing`,
      description,
      type: "article",
      url: canonical,
    },
    twitter: {
      card: "summary_large_image",
      title: `${model.name} VRAM Requirements`,
      description,
    },
    alternates: { canonical },
  };
}

/** Shared calculator deep-link with the model's defaults prefilled. */
function calculatorLink(modelId: string, gpuId: string, quant: string, gpuCount: number) {
  return `https://tokcalc.vercel.app/#t=calculator&m=${modelId}&g=${gpuId}&q=${quant}&n=${gpuCount}&b=${REF_BATCH}&p=${REF_CONTEXT}&o=500&cb=1&cbm=2`;
}

export default async function ModelPage({
  params,
}: {
  params: Promise<{ model: string }>;
}) {
  const { model: modelId } = await params;
  const content = MODEL_CONTENT_MAP[modelId];
  if (!content) notFound();

  const { model, defaultQuant, quantRows, gpuFits, cheapest, fastest, singleGpuOptions, faqs } =
    content;

  const defRow = quantRows.find((r) => r.quant.id === defaultQuant)!;
  const fp16Row = quantRows.find((r) => r.quant.id === "fp16")!;
  const attention = attentionSummary(model);

  // Top GPU rows: everything that fits, capped so the table stays readable.
  const fittingRows = gpuFits.filter((g) => g.fits).slice(0, 12);
  const recommendation = cheapest ?? fastest;
  const caveat = multiGpuCaveat(cheapest) ?? multiGpuCaveat(fastest);

  // Long-context behaviour on the recommended configuration.
  const contextRows = recommendation
    ? CONTEXT_LADDER.map((ctx) => ({
        ctx,
        concurrency: concurrencyAtContext(
          model.id,
          recommendation.gpu.id,
          recommendation.gpuCount ?? 1,
          defaultQuant,
          ctx,
        ),
      }))
    : [];

  const relatedPairs = gpuFits
    .filter((g) => g.fits && (g.singleGpu || g.gpuCount! <= 4))
    .filter((g) => hasModelGpuPage(model.id, g.gpu.id))
    .slice(0, 8);

  const categoryLabel =
    model.category === "vlm"
      ? "Vision-language"
      : model.category === "text"
        ? "Text"
        : model.category.charAt(0).toUpperCase() + model.category.slice(1);

  return (
    <ContentShell breadcrumb={[{ label: "Models", href: "/models" }]}>
      <FaqJsonLd faqs={faqs} />

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge variant="outline" className="text-[10px]">
          {categoryLabel}
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {model.isMoE ? "Mixture-of-Experts" : "Dense"}
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {model.paramsB.toFixed(model.paramsB % 1 === 0 ? 0 : 2)}B params
        </Badge>
        {model.isMoE && (
          <Badge variant="outline" className="text-[10px]">
            {model.activeParamsB.toFixed(0)}B active
          </Badge>
        )}
        <Badge variant="outline" className="text-[10px]">
          {fmtContext(model.maxContext)} max context
        </Badge>
      </div>

      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
        {model.name} VRAM Requirements:{" "}
        <span className="text-emerald-500">How Many GPUs & What It Costs</span>
      </h1>
      <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8 max-w-3xl">
        {model.name} needs{" "}
        <strong className="text-foreground">{defRow.weightsGb.toFixed(1)} GB</strong> of VRAM for
        its weights at {defRow.quant.label} —{" "}
        {defRow.minVramGb.toFixed(1)} GB once a KV cache is loaded. Below is the full breakdown:
        which
        GPUs fit, how many you need, what it serves, and what it costs per million tokens. Every
        figure comes from tokcalc&apos;s open-source capacity model.
      </p>

      {/* ---- Quick answer ---- */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <MemoryStick className="size-3" />
              Weights ({defRow.quant.label})
            </div>
            <div className="text-xl font-bold font-mono">{defRow.weightsGb.toFixed(1)} GB</div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {defRow.minVramGb.toFixed(1)} GB with KV cache
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Cpu className="size-3" />
              Fits on 1 GPU?
            </div>
            <div className="text-xl font-bold">
              {singleGpuOptions.length ? (
                <span className="text-emerald-500">Yes</span>
              ) : (
                <span className="text-amber-500">No</span>
              )}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {singleGpuOptions.length
                ? `${singleGpuOptions[0].gpu.name} or larger`
                : `Needs ${fastest?.gpuCount ?? "8+"}× at ${defRow.quant.label}`}
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Gauge className="size-3" />
              Cheapest config
            </div>
            <div className="text-xl font-bold font-mono">
              {cheapest ? fmtMoney(cheapest.costPerMillion) : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {cheapest
                ? `${cheapest.gpuCount}× ${cheapest.gpu.name} / 1M tokens`
                : "No rental price published"}
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Zap className="size-3" />
              Fastest config
            </div>
            <div className="text-xl font-bold font-mono">
              {fastest ? fmtTokens(fastest.aggregateTokensPerSec) : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {fastest ? `tok/s on ${fastest.gpuCount}× ${fastest.gpu.name}` : "—"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ---- VRAM by quantization ---- */}
      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <MemoryStick className="size-4 text-emerald-500" />
            VRAM requirement by quantization
          </CardTitle>
          <CardDescription className="text-xs">
            Weight footprint = params × bytes per param. The &ldquo;with KV cache&rdquo; column adds one
            request at {fmtContext(4096)} context.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-[10px] uppercase">
              <tr>
                <th className="text-left px-4 py-2">Quantization</th>
                <th className="text-right px-4 py-2">Weights</th>
                <th className="text-right px-4 py-2">With KV cache</th>
                <th className="text-left px-4 py-2">Smallest single GPU</th>
              </tr>
            </thead>
            <tbody>
              {quantRows.map((row) => (
                <tr
                  key={row.quant.id}
                  className={`border-t border-border/40 ${
                    row.quant.id === defaultQuant ? "bg-emerald-500/5" : ""
                  }`}
                >
                  <td className="px-4 py-2 font-medium">
                    {row.quant.label}
                    {row.quant.id === defaultQuant && (
                      <Badge className="ml-2 text-[9px] bg-emerald-500">recommended</Badge>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">{row.weightsGb.toFixed(1)} GB</td>
                  <td className="px-4 py-2 text-right font-mono">
                    {row.minVramGb.toFixed(1)} GB
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">
                    {row.smallestGpuThatHolds
                      ? `${row.smallestGpuThatHolds.name} (${row.smallestGpuThatHolds.vramGb} GB)`
                      : "— needs multi-GPU"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* ---- GPU selection table ---- */}
      <Card className="border-emerald-500/30 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="size-4 text-emerald-500" />
            Which GPU should you use?
          </CardTitle>
          <CardDescription className="text-xs">
            Reference workload: {fmtContext(REF_CONTEXT)} context, {REF_BATCH} concurrent users,
            continuous batching, {defRow.quant.label}. Ranked by fitting first, then by tokens per
            dollar-hour.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          <table className="w-full text-sm min-w-[640px]">
            <thead className="bg-muted/40 text-[10px] uppercase">
              <tr>
                <th className="text-left px-4 py-2">GPU</th>
                <th className="text-right px-4 py-2">GPUs</th>
                <th className="text-right px-4 py-2">Total VRAM</th>
                <th className="text-right px-4 py-2">Decode tok/s</th>
                <th className="text-right px-4 py-2">Aggregate tok/s</th>
                <th className="text-right px-4 py-2">Cost / 1M tokens</th>
              </tr>
            </thead>
            <tbody>
              {fittingRows.map((fit) => (
                <tr key={fit.gpu.id} className="border-t border-border/40">
                  <td className="px-4 py-2">
                    {hasModelGpuPage(model.id, fit.gpu.id) ? (
                      <Link
                        href={`/models/${model.id}/on/${fit.gpu.id}`}
                        className="font-medium hover:underline underline-offset-2"
                      >
                        {fit.gpu.name}
                      </Link>
                    ) : (
                      <span className="font-medium">{fit.gpu.name}</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fit.gpuCount}×
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fit.capacityGb} GB
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fmtTokens(fit.result.decodeTokensPerSec)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fmtTokens(fit.aggregateTokensPerSec)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-emerald-500">
                    {fmtMoney(fit.costPerMillion)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* ---- Caveat on PCIe scale-out ---- */}
      {caveat && (
        <p className="text-xs text-muted-foreground border-l-2 border-amber-500/50 pl-3 mb-8 leading-relaxed">
          {caveat}
        </p>
      )}

      {/* ---- Long context ---- */}
      {recommendation && contextRows.length > 0 && (
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Gauge className="size-4 text-emerald-500" />
              How concurrency changes with context length
            </CardTitle>
            <CardDescription className="text-xs">
              On {recommendation.gpuCount}× {recommendation.gpu.name} at {defRow.quant.label}. KV
              cache scales linearly with context, so long contexts cut the number of users you can
              serve from the same GPUs.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Context</th>
                  <th className="text-right px-4 py-2">Concurrent users</th>
                  <th className="text-right px-4 py-2">KV per request</th>
                  <th className="text-left px-4 py-2">Fits?</th>
                </tr>
              </thead>
              <tbody>
                {contextRows.map((row) => (
                  <tr key={row.ctx} className="border-t border-border/40">
                    <td className="px-4 py-2 font-mono">{fmtContext(row.ctx)}</td>
                    <td className="px-4 py-2 text-right font-mono">
                      {row.concurrency ?? 0}
                    </td>
                    <td className="px-4 py-2 text-right font-mono">
                      {fmtBytes(
                        (2 * model.layers * model.kvHeads * model.headDim * 2 * row.ctx) / 1e9,
                      )}
                    </td>
                    <td className="px-4 py-2">
                      {row.concurrency ? (
                        <span className="text-emerald-500">✓ fits</span>
                      ) : (
                        <span className="text-red-500">✗ does not fit</span>
                      )}
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
            <h2 className="font-semibold mb-1">
              Try {model.name} with your own context and concurrency
            </h2>
            <p className="text-xs text-muted-foreground">
              The calculator uses the exact same formulas as this page — change the workload and
              watch every number update.
            </p>
          </div>
          <Button asChild className="shrink-0">
            <a
              href={calculatorLink(
                model.id,
                recommendation?.gpu.id ?? "h100-sxm",
                defaultQuant,
                recommendation?.gpuCount ?? 1,
              )}
            >
              Open in calculator <ArrowRight className="size-4 ml-1" />
            </a>
          </Button>
        </CardContent>
      </Card>

      {/* ---- Related pages ---- */}
      {relatedPairs.length > 0 && (
        <div className="mb-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground mb-3">
            {model.name} on specific GPUs
          </h2>
          <div className="flex flex-wrap gap-2">
            {relatedPairs.map((fit) => (
              <Link
                key={fit.gpu.id}
                href={`/models/${model.id}/on/${fit.gpu.id}`}
                className="text-xs border border-border/60 rounded-md px-2.5 py-1.5 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-colors"
              >
                {model.name} on {fit.gpu.name}
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* ---- Architecture facts ---- */}
      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base">Model architecture</CardTitle>
          <CardDescription className="text-xs">
            The numbers behind the KV-cache and throughput math.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <tbody>
              {[
                ["Total parameters", `${model.paramsB.toFixed(2)}B`],
                ["Active parameters per token", `${model.activeParamsB.toFixed(2)}B`],
                ["Architecture", model.isMoE ? "Mixture-of-Experts" : "Dense"],
                ["Attention layout", attention.label],
                ["Transformer layers", String(model.layers)],
                ["Hidden dimension", String(model.hiddenDim)],
                [
                  "Attention heads",
                  `${model.qHeads} query / ${model.kvHeads} KV (GQA)`,
                ],
                ["Head dimension", String(model.headDim)],
                ["Vocabulary size", model.vocabSize.toLocaleString()],
                ["Max context (catalog)", fmtContext(model.maxContext)],
                ...(model.nativeContext
                  ? ([["Native context", fmtContext(model.nativeContext)]] as [string, string][])
                  : []),
                ...(model.contextExtensionMethod
                  ? ([["Context extension", model.contextExtensionMethod]] as [string, string][])
                  : []),
                [
                  `KV cache per token @ ${fmtContext(REF_CONTEXT)}`,
                  fmtKb(attention.kvPerTokenKb),
                ],
                ...(attention.stateGb > 0
                  ? ([
                      ["Fixed recurrent state", `${fmtBytes(attention.stateGb)} per request`],
                    ] as [string, string][])
                  : []),
                ...(model.residentWeightBytes
                  ? ([
                      [
                        "Native checkpoint size",
                        fmtBytes(model.residentWeightBytes / 1e9),
                      ],
                    ] as [string, string][])
                  : []),
                ...(model.nativeWeightFormat
                  ? ([["Native weight format", model.nativeWeightFormat]] as [string, string][])
                  : []),
                ...(model.license
                  ? ([["License", model.license]] as [string, string][])
                  : []),
              ].map(([label, value]) => (
                <tr key={label} className="border-b border-border/30 last:border-0">
                  <td className="px-4 py-2 text-muted-foreground">{label}</td>
                  <td className="px-4 py-2 font-mono text-xs">{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-4 py-3 text-xs text-muted-foreground border-t border-border/30">
            {attention.detail}
            {model.architectureStatus === "memory-only" && (
              <>
                {" "}
                <span className="text-amber-600 dark:text-amber-400">
                  This model&apos;s architecture is only partially modelled
                </span>{" "}
                — the memory figure above is a lower bound, because one
                architecture-specific cache is not modelled. Throughput should
                be read as an upper bound.
              </>
            )}
          </p>
        </CardContent>
      </Card>

      {model.provenance.limitations && model.provenance.limitations.length > 0 && (
        <Card className="border-border/60 shadow-sm">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Info className="size-4 text-amber-500" />
              Limitations of these numbers
            </CardTitle>
            <CardDescription className="text-xs">
              Where the source does not fully establish the values above.
              {" "}
              {model.provenance.applicability === "lower-bound"
                ? "The figures are a lower bound — real usage is at least this."
                : model.provenance.applicability === "partial"
                  ? "Some of the figures hold only under conditions that may not apply to you."
                  : null}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1.5 text-xs text-muted-foreground">
              {model.provenance.limitations.map((limitation, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-amber-500 shrink-0 mt-0.5">•</span>
                  <span className="leading-relaxed">{limitation}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <ContentFooter />
    </ContentShell>
  );
}
