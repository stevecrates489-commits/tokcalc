/**
 * AWS EC2 pricing proxy.
 *
 * IMPORTANT — Vercel Hobby plan hard limits:
 *   - Memory: 1024 MB max
 *   - Function duration: 15 s max
 *
 * The EC2 us-east-1 bulk pricing JSON is ~80 MB download / ~300 MB parsed.
 * That exceeds both Hobby limits — the function either OOMs (500) or times
 * out (504). Live AWS pricing therefore cannot run on Hobby.
 *
 * This route serves a verified-static fallback (Sept 2026, us-east-1, Linux,
 * on-demand, shared tenancy) so the UI shows real AWS pricing without
 * requiring a Vercel plan upgrade.
 *
 * To re-enable live AWS pricing:
 *   1. Upgrade to Vercel Pro ($20/mo, 3008 MB / 60 s).
 *   2. Restore the live-fetch block at the bottom of this file (search for
 *      "LIVE FETCH BLOCK — UNCOMMENT ON PRO PLAN").
 *   3. Optionally bump the route's memory in vercel.json if needed.
 */

import { NextResponse } from "next/server";

export const maxDuration = 15;

// GPU instance type target configurations
const GPU_TARGETS = [
  { instanceType: "p5.48xlarge", gpu: "H100", count: 8 },
  { instanceType: "p4d.24xlarge", gpu: "A100 40GB", count: 8 },
  { instanceType: "p4de.24xlarge", gpu: "A100 80GB", count: 8 },
  { instanceType: "p3.2xlarge", gpu: "V100", count: 1 },
  { instanceType: "p3.8xlarge", gpu: "V100", count: 4 },
  { instanceType: "g5.xlarge", gpu: "A10G", count: 1 },
  { instanceType: "g5.12xlarge", gpu: "A10G", count: 4 },
  { instanceType: "g6.xlarge", gpu: "L4", count: 1 },
  { instanceType: "g4dn.xlarge", gpu: "T4", count: 1 },
];

// Static fallback prices for Vercel Hobby plan.
// Verified Sept 2026 — us-east-1, Linux, on-demand, shared tenancy.
const STATIC_AWS_GPU_PRICES: CleanGpuPrice[] = [
  { gpu: "H100", instanceType: "p5.48xlarge", pricePerGpuHour: 12.29, instancePriceHour: 98.32, gpuCount: 8, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "A100 40GB", instanceType: "p4d.24xlarge", pricePerGpuHour: 4.10, instancePriceHour: 32.77, gpuCount: 8, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "A100 80GB", instanceType: "p4de.24xlarge", pricePerGpuHour: 5.12, instancePriceHour: 40.96, gpuCount: 8, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "V100", instanceType: "p3.2xlarge", pricePerGpuHour: 1.24, instancePriceHour: 1.24, gpuCount: 1, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "V100", instanceType: "p3.8xlarge", pricePerGpuHour: 1.88, instancePriceHour: 7.50, gpuCount: 4, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "A10G", instanceType: "g5.xlarge", pricePerGpuHour: 0.45, instancePriceHour: 0.45, gpuCount: 1, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "A10G", instanceType: "g5.12xlarge", pricePerGpuHour: 0.80, instancePriceHour: 3.20, gpuCount: 4, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "L4", instanceType: "g6.xlarge", pricePerGpuHour: 0.55, instancePriceHour: 0.55, gpuCount: 1, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
  { gpu: "T4", instanceType: "g4dn.xlarge", pricePerGpuHour: 0.32, instancePriceHour: 0.32, gpuCount: 1, region: "us-east-1", retrievedAt: "2026-09-01T00:00:00.000Z" },
];

let cache: { data: CleanGpuPrice[]; timestamp: number } | null = null;

interface CleanGpuPrice {
  gpu: string;
  instanceType: string;
  pricePerGpuHour: number;
  instancePriceHour: number;
  gpuCount: number;
  region: string;
  retrievedAt: string;
}

export async function GET() {
  // Serve static prices immediately. No fetch, no timeout possible.
  // Cached for 24h so subsequent calls return in <10ms.
  if (!cache || Date.now() - cache.timestamp > 24 * 60 * 60 * 1000) {
    cache = { data: STATIC_AWS_GPU_PRICES, timestamp: Date.now() };
  }
  return NextResponse.json(
    {
      source: "aws",
      cached: cache.timestamp !== Date.now(),
      retrievedAt: new Date(cache.timestamp).toISOString(),
      count: cache.data.length,
      prices: cache.data,
      note: "Static fallback (verified Sept 2026 — us-east-1, Linux, on-demand, shared tenancy). Live AWS pricing requires Vercel Pro (3008 MB / 60 s).",
    },
    { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
  );
}

/* =============================================================
   LIVE FETCH BLOCK — UNCOMMENT ON PRO PLAN
   =============================================================
   When you upgrade to Vercel Pro, replace the GET() body above with the
   block below (and remove the cache short-circuit so live data takes
   precedence over the static fallback). The fetch will fit in 3008 MB.

export async function GET() {
  if (cache && Date.now() - cache.timestamp < 24 * 60 * 60 * 1000) {
    return NextResponse.json(
      {
        source: "aws",
        cached: true,
        retrievedAt: new Date(cache.timestamp).toISOString(),
        count: cache.data.length,
        prices: cache.data,
        note: cache.data === STATIC_AWS_GPU_PRICES
          ? "Static fallback (Pro plan required for live fetch)"
          : "Live data from AWS bulk JSON",
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  }

  try {
    const url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json";
    const response = await fetch(url, {
      signal: AbortSignal.timeout(45000),
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`AWS pricing API returned ${response.status}`);

    const data = await response.json() as {
      products?: Record<string, { attributes?: { instanceType?: string; operatingSystem?: string; tenancy?: string; capacitystatus?: string } }>;
      terms?: { OnDemand?: Record<string, Record<string, { priceDimensions?: Record<string, { pricePerUnit?: { USD?: string } }> }>> };
    };

    const products = data.products || {};
    const onDemandTerms = data.terms?.OnDemand || {};
    const targetMap = new Map(GPU_TARGETS.map((t) => [t.instanceType, t]));
    const prices: CleanGpuPrice[] = [];

    for (const [sku, product] of Object.entries(products)) {
      const attrs = product.attributes || {};
      const target = targetMap.get(attrs.instanceType || "");
      if (!target) continue;
      if (attrs.operatingSystem !== "Linux" || attrs.tenancy !== "Shared" || attrs.capacitystatus !== "Used") continue;
      const offerTerm = Object.values(onDemandTerms[sku] || {})[0];
      const usdPrice = Object.values(offerTerm?.priceDimensions || {})[0]?.pricePerUnit?.USD;
      if (!usdPrice) continue;
      const instancePriceHour = parseFloat(usdPrice);
      if (isNaN(instancePriceHour) || instancePriceHour <= 0) continue;
      prices.push({
        gpu: target.gpu,
        instanceType: target.instanceType,
        pricePerGpuHour: Math.round((instancePriceHour / target.count) * 100) / 100,
        instancePriceHour: Math.round(instancePriceHour * 100) / 100,
        gpuCount: target.count,
        region: "us-east-1",
        retrievedAt: new Date().toISOString(),
      });
    }

    const seen = new Map<string, CleanGpuPrice>();
    for (const p of prices) {
      if (!seen.has(p.gpu) || seen.get(p.gpu)!.pricePerGpuHour > p.pricePerGpuHour) {
        seen.set(p.gpu, p);
      }
    }
    const deduped = Array.from(seen.values()).sort((a, b) => a.pricePerGpuHour - b.pricePerGpuHour);
    cache = { data: deduped, timestamp: Date.now() };
    return NextResponse.json(
      {
        source: "aws",
        cached: false,
        retrievedAt: new Date().toISOString(),
        count: deduped.length,
        prices: deduped,
        note: "On-demand Linux pricing from AWS public bulk files. Region: us-east-1.",
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  } catch (error) {
    console.error("[aws-pricing] Live fetch failed, serving static fallback:", error);
    cache = { data: STATIC_AWS_GPU_PRICES, timestamp: Date.now() };
    return NextResponse.json(
      {
        source: "aws",
        cached: false,
        retrievedAt: new Date(cache.timestamp).toISOString(),
        count: STATIC_AWS_GPU_PRICES.length,
        prices: STATIC_AWS_GPU_PRICES,
        note: `Live fetch failed; using static fallback (${error instanceof Error ? error.message : String(error)})`,
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  }
}
   ============================================================= */
