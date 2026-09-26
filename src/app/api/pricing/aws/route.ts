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
    return NextResponse.json({
      source: "aws",
      cached: true,
      retrievedAt: new Date(cache.timestamp).toISOString(),
      count: cache.data.length,
      prices: cache.data,
    });
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

    return NextResponse.json({
      source: "aws",
      cached: false,
      retrievedAt: new Date().toISOString(),
      count: deduped.length,
      prices: deduped,
      note: "On-demand Linux pricing from AWS public bulk files. No IAM credentials required. Region: us-east-1.",
    });
  } catch (error) {
    console.error("AWS pricing fetch error:", error);
    return NextResponse.json(
      {
        source: "aws",
        error: "Failed to fetch AWS bulk pricing due to response size limits or connection timeouts.",
        fallback: "Use static pricing from tokcalc's GPU catalog or Azure/Vast.ai live pricing.",
      },
      { status: 502 },
    );
  }
}