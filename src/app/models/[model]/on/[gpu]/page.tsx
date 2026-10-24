import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowRight, Cpu, Gauge, MemoryStick, Zap } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ContentShell, ContentFooter, FaqJsonLd } from "@/components/content/page-shell";
import {
  MODEL_CONTENT_MAP,
  REF_BATCH,
  REF_CONTEXT,
  CONTEXT_LADDER,
  computeModelContent,
  concurrencyAtContext,
  modelGpuPairs,
  multiGpuCaveat,
  hasModelGpuPage,
  type ModelContent,
} from "@/lib/content/model-content";
import {
  GPU_MAP,
  GPUS,
  calculate,
  fmtTokens,
  fmtMoney,
  fmtBytes,
  fmtMs,
  fmtContext,
  type Quantization,
} from "@/lib/token-calc";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://tokcalc.vercel.app";

/** Pre-render only the model×GPU pairs that are actually worth a page. */
export function generateStaticParams() {
  return modelGpuPairs().map(({ modelId, gpuId }) => ({
    model: modelId,
    gpu: gpuId,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ model: string; gpu: string }>;
}): Promise<Metadata> {
  const { model: modelId, gpu: gpuId } = await params;
  const content = MODEL_CONTENT_MAP[modelId];
  const gpu = GPU_MAP[gpuId];
  if (!content || !gpu) return {};

  const { model, defaultQuant, quantRows, cheapest } = content;
  const defRow = quantRows.find((r) => r.quant.id === defaultQuant)!;
  const fit = content.gpuFits.find((g) => g.gpu.id === gpuId);
  const canonical = `${SITE_URL}/models/${modelId}/on/${gpuId}`;

  const fitSentence =
    fit?.fits && fit.gpuCount === 1
      ? `Fits on a single ${gpu.name}.`
      : fit?.fits
        ? `Needs ${fit.gpuCount}× ${gpu.name}.`
        : `Does not fit on ${gpu.name} alone.`;

  const costSentence = fit?.fits
    ? `Serves at ${fmtTokens(fit.aggregateTokensPerSec)} tok/s for ${fmtMoney(fit.costPerMillion)} per 1M output tokens.`
    : "";

  const title = `${model.name} on ${gpu.name}: Does It Fit, How Fast, What It Costs`;
  const description = `${model.name} on ${gpu.name}: ${fitSentence} ${costSentence} ${defRow.weightsGb.toFixed(0)} GB of weights at ${defRow.quant.label}, full VRAM and concurrency math from tokcalc's open-source formulas.`;

  return {
    title,
    description,
    keywords: [
      `${model.name} on ${gpu.name}`,
      `${model.name} ${gpu.name} VRAM`,
      `can ${model.name} run on ${gpu.name}`,
      `${model.id} ${gpu.id}`,
      `${model.name} ${gpu.vramGb}GB`,
      `${model.name} inference ${gpu.name}`,
    ],
    openGraph: {
      title: `${model.name} on ${gpu.name}`,
      description,
      type: "article",
      url: canonical,
    },
    twitter: { card: "summary_large_image", title, description },
    alternates: { canonical },
  };
}

export default async function ModelOnGpuPage({
  params,
}: {
  params: Promise<{ model: string; gpu: string }>;
}) {
  const { model: modelId, gpu: gpuId } = await params;
  const content = MODEL_CONTENT_MAP[modelId];
  const gpu = GPU_MAP[gpuId];
  if (!content || !gpu) notFound();

  const { model, defaultQuant, quantRows, singleGpuOptions, faqs: modelFaqs } = content;
  const defRow = quantRows.find((r) => r.quant.id === defaultQuant)!;
  const fit = content.gpuFits.find((g) => g.gpu.id === gpuId)!;

  // ---- GPU-count ladder: what happens at 1× / 2× / 4× / 8× --------------
  const ladder = [1, 2, 4, 8].map((n) => {
    const r = calculate({
      modelId: model.id,
      gpuId: gpu.id,
      quantization: defaultQuant,
      numGpus: n,
      batchSize: REF_BATCH,
      promptTokens: REF_CONTEXT,
      outputTokens: 500,
      useContinuousBatching: true,
    });
    return { n, result: r, capacityGb: gpu.vramGb * n };
  });

  const minGpuCount = ladder.find((l) => l.result.vramFits)?.n ?? null;

  // ---- Context ladder on the minimum viable configuration ---------------
  const contextRows = CONTEXT_LADDER.map((ctx) => ({
    ctx,
    concurrency: concurrencyAtContext(
      model.id,
      gpu.id,
      minGpuCount ?? 8,
      defaultQuant,
      ctx,
    ),
  }));

  // ---- Other GPUs that also run this model ------------------------------
  const alternatives = content.gpuFits
    .filter((g) => g.fits && g.gpu.id !== gpuId)
    .filter((g) => hasModelGpuPage(model.id, g.gpu.id))
    .slice(0, 6);

  // PCIe scale-out caveat for this configuration.
  const caveat = multiGpuCaveat(fit);

  // ---- Page-level FAQ --------------------------------------------------
  const faqs = [
    {
      q: `Can ${model.name} run on ${gpu.name}?`,
      a: fit.fits
        ? minGpuCount === 1
          ? `Yes. ${model.name} at ${defRow.quant.label} needs ${fit.result.totalVramNeededGb.toFixed(1)} GB including the KV cache, and ${gpu.name} has ${gpu.vramGb} GB — so it fits on a single GPU.`
          : `Yes, but it needs ${minGpuCount}× ${gpu.name}. At ${defRow.quant.label} the total footprint is ${fit.result.totalVramNeededGb.toFixed(1)} GB (${defRow.weightsGb.toFixed(1)} GB of weights plus ${(fit.result.kvCacheTotalGb).toFixed(1)} GB of KV cache at ${fmtContext(REF_CONTEXT)} × ${REF_BATCH} concurrent users), against ${fit.capacityGb} GB of available VRAM across ${minGpuCount} GPUs.`
        : `Not at ${defRow.quant.label}. ${model.name} needs ${defRow.weightsGb.toFixed(1)} GB for its weights alone, before any KV cache, and ${gpu.name} only has ${gpu.vramGb} GB. Quantise further, use a smaller model, or move to a larger accelerator.`,
    },
    {
      q: `How fast is ${model.name} on ${gpu.name}?`,
      a: fit.fits
        ? `${fmtTokens(fit.result.decodeTokensPerSec)} tokens/sec per stream and ${fmtTokens(fit.aggregateTokensPerSec)} tok/s aggregate with continuous batching at ${REF_BATCH} concurrent users. Time to first token is ${fmtMs(fit.result.ttftMs)}, and inter-token latency is ${fmtMs(fit.result.itlMs)}.`
        : `It cannot be measured on ${gpu.name} at this quantization because the model does not fit in ${gpu.vramGb} GB of VRAM.`,
    },
    {
      q: `What does ${model.name} on ${gpu.name} cost per million tokens?`,
      a: fit.fits
        ? `${fmtMoney(fit.costPerMillion)} per 1M output tokens at ${fit.gpuCount}× ${gpu.name}${gpu.usdPerHour ? ` (${fmtMoney(gpu.usdPerHour)}/hr)` : ""}. That is the marginal hardware cost — add idle capacity, ops, and egress for the real figure.`
        : `Not applicable at this configuration — the model does not fit on ${gpu.name}.`,
    },
    {
      q: `How much VRAM does ${model.name} need on ${gpu.name}?`,
      a: `${defRow.weightsGb.toFixed(1)} GB for the weights at ${defRow.quant.label}, plus ${(fit.result.kvCacheTotalGb).toFixed(1)} GB of KV cache for ${REF_BATCH} concurrent requests at ${fmtContext(REF_CONTEXT)} context — ${fit.result.totalVramNeededGb.toFixed(1)} GB total. Every extra concurrent user at long context costs more, so leave headroom.`,
    },
    {
      q: `Is ${gpu.name} the right choice for ${model.name}?`,
      a: fit.fits
        ? `It works. Compare ${fmtMoney(fit.costPerMillion)} per 1M tokens against the alternatives below — the cheapest option that fits the same workload is ${content.cheapest ? `${content.cheapest.gpuCount}× ${content.cheapest.gpu.name} at ${fmtMoney(content.cheapest.costPerMillion)}` : "—"}, and the fastest is ${content.fastest ? `${content.fastest.gpuCount}× ${content.fastest.gpu.name} at ${fmtTokens(content.fastest.aggregateTokensPerSec)} tok/s` : "—"}.`
        : `No — ${gpu.name} is undersized for ${model.name}. Look at the alternatives that do fit.`,
    },
  ];

  const canonical = `${SITE_URL}/models/${modelId}/on/${gpuId}`;

  return (
    <ContentShell
      breadcrumb={[
        { label: "Models", href: "/models" },
        { label: model.name, href: `/models/${model.id}` },
      ]}
    >
      <FaqJsonLd faqs={faqs} />

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge variant="outline" className="text-[10px]">
          {gpu.vendor}
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {gpu.vramGb} GB VRAM
        </Badge>
        <Badge variant="outline" className="text-[10px]">
          {gpu.memBandwidthGbps.toLocaleString()} GB/s
        </Badge>
        {gpu.usdPerHour != null && (
          <Badge variant="outline" className="text-[10px]">
            ${gpu.usdPerHour}/hr
          </Badge>
        )}
      </div>

      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">
        {model.name} on {gpu.name}:{" "}
        <span className="text-emerald-500">Does It Fit, How Fast, What It Costs</span>
      </h1>
      <p className="text-muted-foreground text-sm sm:text-base leading-relaxed mb-8 max-w-3xl">
        {model.name} at {defRow.quant.label} needs{" "}
        <strong className="text-foreground">{defRow.weightsGb.toFixed(1)} GB</strong> for its
        weights and{" "}
        <strong className="text-foreground">
          {fit.result.totalVramNeededGb.toFixed(1)} GB
        </strong>{" "}
        in total at {fmtContext(REF_CONTEXT)} context with {REF_BATCH} concurrent users.{" "}
        {fit.fits ? (
          <>
            That means{" "}
            {minGpuCount === 1 ? (
              <>
                it fits on a <strong className="text-foreground">single {gpu.name}</strong>.
              </>
            ) : (
              <>
                it needs{" "}
                <strong className="text-foreground">
                  {minGpuCount}× {gpu.name}
                </strong>
                .
              </>
            )}
          </>
        ) : (
          <>
            That <strong className="text-foreground">does not fit</strong> on {gpu.name} — see
            the alternatives below.
          </>
        )}{" "}
        Every figure is computed by tokcalc&apos;s open-source capacity model.
      </p>

      {/* ---- Answer tiles ---- */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        <Card
          className={`shadow-sm ${
            fit.fits ? "border-emerald-500/40" : "border-red-500/40"
          }`}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Cpu className="size-3" />
              Fits?
            </div>
            <div
              className={`text-xl font-bold ${fit.fits ? "text-emerald-500" : "text-red-500"}`}
            >
              {fit.fits ? (minGpuCount === 1 ? "Yes, 1 GPU" : `Yes, ${minGpuCount}×`) : "No"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {fit.result.totalVramNeededGb.toFixed(1)} GB needed / {fit.capacityGb} GB available
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Zap className="size-3" />
              Decode speed
            </div>
            <div className="text-xl font-bold font-mono">
              {fit.fits ? fmtTokens(fit.result.decodeTokensPerSec) : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">tok/s per stream</div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <Gauge className="size-3" />
              Aggregate
            </div>
            <div className="text-xl font-bold font-mono">
              {fit.fits ? fmtTokens(fit.aggregateTokensPerSec) : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              tok/s @ {REF_BATCH} concurrent
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-sm">
          <CardContent className="p-4">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
              <MemoryStick className="size-3" />
              Cost / 1M tokens
            </div>
            <div className="text-xl font-bold font-mono text-emerald-500">
              {fit.fits ? (fit.rentable ? fmtMoney(fit.costPerMillion) : "—") : "—"}
            </div>
            <div className="text-[11px] text-muted-foreground mt-1">
              {fit.fits
                ? fit.rentable
                  ? `${fit.gpuCount}× ${gpu.name}`
                  : `${fit.gpuCount}× ${gpu.name} · quote required`
                : "does not fit"}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ---- Caveat on PCIe scale-out ---- */}
      {caveat && (
        <p className="text-xs text-muted-foreground border-l-2 border-amber-500/50 pl-3 mb-8 leading-relaxed">
          {caveat}
        </p>
      )}

      {/* ---- GPU count ladder ---- */}
      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="size-4 text-emerald-500" />
            What changes as you add {gpu.name}s
          </CardTitle>
          <CardDescription className="text-xs">
            Tensor-parallel degree vs VRAM, throughput, and cost. Cost scales linearly with GPU
            count; decode throughput scales near-linearly too, so cost per token stays roughly flat.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0 overflow-x-auto">
          <table className="w-full text-sm min-w-[640px]">
            <thead className="bg-muted/40 text-[10px] uppercase">
              <tr>
                <th className="text-left px-4 py-2">GPUs</th>
                <th className="text-right px-4 py-2">Total VRAM</th>
                <th className="text-right px-4 py-2">VRAM needed</th>
                <th className="text-right px-4 py-2">Decode tok/s</th>
                <th className="text-right px-4 py-2">Aggregate tok/s</th>
                <th className="text-right px-4 py-2">TTFT</th>
                <th className="text-right px-4 py-2">Cost / 1M tok</th>
                <th className="text-center px-4 py-2">Fits?</th>
              </tr>
            </thead>
            <tbody>
              {ladder.map((row) => (
                <tr
                  key={row.n}
                  className={`border-t border-border/40 ${
                    row.n === minGpuCount ? "bg-emerald-500/5" : ""
                  }`}
                >
                  <td className="px-4 py-2 font-medium">
                    {row.n}× {gpu.name.split(" ")[0]}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">{row.capacityGb} GB</td>
                  <td className="px-4 py-2 text-right font-mono">
                    {row.result.totalVramNeededGb.toFixed(1)} GB
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fmtTokens(row.result.decodeTokensPerSec)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fmtTokens(row.result.aggregateTokensPerSec)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fmtMs(row.result.ttftMs)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-emerald-500">
                    {fmtMoney(row.result.costPerMillionOutputTokens)}
                  </td>
                  <td className="px-4 py-2 text-center">
                    {row.result.vramFits ? (
                      <span className="text-emerald-500">✓</span>
                    ) : (
                      <span className="text-red-500">✗</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* ---- Context ladder ---- */}
      <Card className="border-border/60 shadow-sm mb-8">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Gauge className="size-4 text-emerald-500" />
            Concurrency vs context length
          </CardTitle>
          <CardDescription className="text-xs">
            On {minGpuCount ?? 8}× {gpu.name} at {defRow.quant.label}. KV cache scales linearly with
            context, so this is the number of users your GPUs can actually serve.
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
                  <td className="px-4 py-2 text-right font-mono">{row.concurrency ?? 0}</td>
                  <td className="px-4 py-2 text-right font-mono">
                    {fmtBytes((2 * model.layers * model.kvHeads * model.headDim * 2 * row.ctx) / 1e9)}
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

      {/* ---- Alternatives ---- */}
      {alternatives.length > 0 && (
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base">Other GPUs that run {model.name}</CardTitle>
            <CardDescription className="text-xs">
              Same workload, same quantization — ranked by value (tokens per dollar-hour).
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0 overflow-x-auto">
            <table className="w-full text-sm min-w-[560px]">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">GPU</th>
                  <th className="text-right px-4 py-2">GPUs</th>
                  <th className="text-right px-4 py-2">Aggregate tok/s</th>
                  <th className="text-right px-4 py-2">Cost / 1M tok</th>
                </tr>
              </thead>
              <tbody>
                {alternatives.map((alt) => (
                  <tr key={alt.gpu.id} className="border-t border-border/40">
                    <td className="px-4 py-2">
                      <Link
                        href={`/models/${model.id}/on/${alt.gpu.id}`}
                        className="font-medium hover:underline underline-offset-2"
                      >
                        {alt.gpu.name}
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-right font-mono">{alt.gpuCount}×</td>
                    <td className="px-4 py-2 text-right font-mono">
                      {fmtTokens(alt.aggregateTokensPerSec)}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-emerald-500">
                      {fmtMoney(alt.costPerMillion)}
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
            <h2 className="font-semibold mb-1">Adjust the workload yourself</h2>
            <p className="text-xs text-muted-foreground">
              Same formulas, your context length, concurrency, and quantisation.
            </p>
          </div>
          <Button asChild className="shrink-0">
            <a
              href={`https://tokcalc.vercel.app/#t=calculator&m=${model.id}&g=${gpu.id}&q=${defaultQuant}&n=${minGpuCount ?? 1}&b=${REF_BATCH}&p=${REF_CONTEXT}&o=500&cb=1&cbm=2`}
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
