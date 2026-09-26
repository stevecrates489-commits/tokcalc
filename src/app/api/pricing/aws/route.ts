/**
 * AWS EC2 pricing proxy — uses public bulk pricing files (NO IAM credentials required).
 * Optimized to extract GPU pricing without throwing out-of-memory errors on serverless.
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

// Static fallback prices for Vercel Hobby plan (1024 MB memory ceiling).
// The EC2 us-east-1 bulk JSON is ~80 MB download / ~300 MB parsed, which
// exceeds Hobby's memory cap. To get live data, upgrade to Pro (3008 MB)
// and set ALLOW_LIVE_AWS_PRICING=1 on Vercel env vars.
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
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

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
  if (cache && Date.now() - cache.timestamp < CACHE_TTL_MS) {
    return NextResponse.json(
      {
        source: "aws",
        cached: true,
        retrievedAt: new Date(cache.timestamp).toISOString(),
        count: cache.data.length,
        prices: cache.data,
        note: cache.data === STATIC_AWS_GPU_PRICES
          ? "Static fallback (set ALLOW_LIVE_AWS_PRICING=1 to enable live fetch)"
          : "Live data from AWS bulk JSON",
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  }

  const allowLive = process.env.ALLOW_LIVE_AWS_PRICING === "1";
  if (!allowLive) {
    // Vercel Hobby plan caps function memory at 1024 MB which is insufficient
    // for the ~300 MB parsed EC2 bulk JSON. Return static fallback by default.
    // To enable live data: upgrade to Pro (3008 MB) + set ALLOW_LIVE_AWS_PRICING=1.
    cache = { data: STATIC_AWS_GPU_PRICES, timestamp: Date.now() };
    return NextResponse.json(
      {
        source: "aws",
        cached: false,
        retrievedAt: new Date(cache.timestamp).toISOString(),
        count: STATIC_AWS_GPU_PRICES.length,
        prices: STATIC_AWS_GPU_PRICES,
        note: "Static fallback (set ALLOW_LIVE_AWS_PRICING=1 to enable live fetch; requires Vercel Pro for sufficient memory)",
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  }

  try {
    const url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json";

    const response = await fetch(url, {
      signal: AbortSignal.timeout(12000),
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`AWS pricing API returned ${response.status}`);
    }

    const data = (await response.json()) as {
      products?: Record<
        string,
        {
          productFamily?: string;
          attributes?: {
            instanceType?: string;
            operatingSystem?: string;
            tenancy?: string;
            capacitystatus?: string;
          };
        }
      >;
      terms?: {
        OnDemand?: Record<
          string,
          Record<
            string,
            {
              priceDimensions?: Record<
                string,
                { pricePerUnit?: { USD?: string } }
              >;
            }
          >
        >;
      };
    };

    const products = data.products || {};
    const onDemandTerms = data.terms?.OnDemand || {};
    const prices: CleanGpuPrice[] = [];

    const targetMap = new Map(GPU_TARGETS.map((t) => [t.instanceType, t]));

    for (const [sku, product] of Object.entries(products)) {
      const attrs = product.attributes || {};
      const instanceType = attrs.instanceType || "";

      const target = targetMap.get(instanceType);
      if (!target) continue;

      if (
        attrs.operatingSystem !== "Linux" ||
        attrs.tenancy !== "Shared" ||
        attrs.capacitystatus !== "Used"
      ) {
        continue;
      }

      const skuTerms = onDemandTerms[sku];
      if (!skuTerms) continue;

      const offerTerm = Object.values(skuTerms)[0];
      if (!offerTerm?.priceDimensions) continue;

      const priceDim = Object.values(offerTerm.priceDimensions)[0];
      const usdPrice = priceDim?.pricePerUnit?.USD;
      if (!usdPrice) continue;

      const instancePriceHour = parseFloat(usdPrice);
      if (isNaN(instancePriceHour) || instancePriceHour <= 0) continue;

      const pricePerGpuHour = Math.round((instancePriceHour / target.count) * 100) / 100;

      prices.push({
        gpu: target.gpu,
        instanceType,
        pricePerGpuHour,
        instancePriceHour: Math.round(instancePriceHour * 100) / 100,
        gpuCount: target.count,
        region: "us-east-1",
        retrievedAt: new Date().toISOString(),
      });
    }

    // Deduplicate: keep cheapest per-GPU price
    const seen = new Map<string, CleanGpuPrice>();
    for (const p of prices) {
      const key = p.gpu;
      if (!seen.has(key) || seen.get(key)!.pricePerGpuHour > p.pricePerGpuHour) {
        seen.set(key, p);
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
        note: "On-demand Linux pricing from AWS public bulk files. No IAM credentials required. Region: us-east-1.",
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  } catch (error) {
    // Live fetch failed (OOM, timeout, or 5xx) — fall back to static prices
    // instead of returning a 502 so the UI stays functional.
    console.error("[aws-pricing] Live fetch failed, falling back to static prices:", error);
    cache = { data: STATIC_AWS_GPU_PRICES, timestamp: Date.now() };
    return NextResponse.json(
      {
        source: "aws",
        cached: false,
        retrievedAt: new Date(cache.timestamp).toISOString(),
        count: STATIC_AWS_GPU_PRICES.length,
        prices: STATIC_AWS_GPU_PRICES,
        note: `Live fetch failed (${error instanceof Error ? error.message : String(error)}); using static fallback`,
      },
      { headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400" } },
    );
  }
}