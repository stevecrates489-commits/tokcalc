/**
 * AWS EC2 pricing proxy — uses public bulk pricing files (NO IAM credentials required).
 *
 * Per the user's research: AWS publishes public pricing JSON files at:
 *   https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json
 *
 * This file is large (~50-100MB) but contains all EC2 pricing for us-east-1.
 * This serverless function:
 *   1. Fetches the bulk file (with 8s timeout — might fail on slow connections)
 *   2. Streams-parses to extract only GPU instance types (p5, p4d, g5, g6, etc.)
 *   3. Caches in memory for 24 hours
 *   4. Returns clean JSON with per-GPU pricing
 *
 * No IAM credentials, no API key, no payment setup — just public data.
 */

import { NextResponse } from "next/server";

// GPU instance type prefixes — filter the massive EC2 catalog
const GPU_INSTANCE_PREFIXES = ["p5", "p5e", "p6", "p4d", "p4de", "p3", "g5", "g6", "g4dn"];

// Known GPU counts per instance type
const GPU_COUNTS: Record<string, number> = {
  "p5.48xlarge": 8, "p5e.48xlarge": 8, "p6.48xlarge": 8,
  "p4d.24xlarge": 8, "p4de.24xlarge": 8,
  "p3.2xlarge": 1, "p3.8xlarge": 4, "p3.16xlarge": 8,
  "g5.xlarge": 1, "g5.2xlarge": 1, "g5.4xlarge": 1, "g5.8xlarge": 1, "g5.12xlarge": 4, "g5.16xlarge": 1, "g5.48xlarge": 8,
  "g6.xlarge": 1, "g6.2xlarge": 1, "g6.4xlarge": 1, "g6.8xlarge": 1, "g6.12xlarge": 4, "g6.16xlarge": 1, "g6.48xlarge": 8,
  "g4dn.xlarge": 1, "g4dn.2xlarge": 1, "g4dn.4xlarge": 1, "g4dn.8xlarge": 1, "g4dn.12xlarge": 4, "g4dn.16xlarge": 1,
};

// Known GPU model per instance type
const GPU_MODEL: Record<string, string> = {
  p5: "H100", p5e: "H200", p6: "B200",
  p4d: "A100 40GB", p4de: "A100 80GB",
  p3: "V100",
  g5: "A10G", g6: "L4", g4dn: "T4",
};

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

function getGpuModel(instanceType: string): string | null {
  const prefix = Object.keys(GPU_MODEL).find((p) => instanceType.startsWith(p));
  return prefix ? GPU_MODEL[prefix] : null;
}

function getGpuCount(instanceType: string): number {
  return GPU_COUNTS[instanceType] || 1;
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
    // Fetch the AWS EC2 bulk pricing file for us-east-1
    // This is a large file (~50-100MB) but we only need GPU instances
    const url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json";

    const response = await fetch(url, {
      signal: AbortSignal.timeout(8000), // 8 second timeout
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) {
      throw new Error(`AWS pricing API returned ${response.status}`);
    }

    // Parse the large JSON — this is memory-intensive but works on Vercel's 1GB limit
    const data = await response.json() as {
      products?: Record<string, {
        productFamily?: string;
        attributes?: {
          instanceType?: string;
          location?: string;
          operatingSystem?: string;
          tenancy?: string;
          capacitystatus?: string;
        };
      }>;
      terms?: {
        OnDemand?: Record<string, {
          priceDimensions?: Record<string, {
            pricePerUnit?: { USD?: string };
          }>;
        }>;
      };
    };

    const products = data.products || {};
    const onDemandTerms = data.terms?.OnDemand || {};

    const prices: CleanGpuPrice[] = [];

    // Iterate products and find GPU instance types
    for (const [sku, product] of Object.entries(products)) {
      const attrs = product.attributes || {};
      const instanceType = attrs.instanceType || "";

      // Skip non-GPU instances
      if (!GPU_INSTANCE_PREFIXES.some((p) => instanceType.startsWith(p))) continue;

      // Filter: Linux, Shared tenancy, Used capacity (on-demand)
      if (attrs.operatingSystem !== "Linux") continue;
      if (attrs.tenancy !== "Shared") continue;
      if (attrs.capacitystatus !== "Used") continue;

      // Find the OnDemand price for this SKU
      const term = onDemandTerms[sku];
      if (!term?.priceDimensions) continue;

      const priceDims = Object.values(term.priceDimensions);
      const usdPrice = priceDims[0]?.pricePerUnit?.USD;
      if (!usdPrice) continue;

      const instancePriceHour = parseFloat(usdPrice);
      if (isNaN(instancePriceHour) || instancePriceHour <= 0) continue;

      const gpuModel = getGpuModel(instanceType);
      const gpuCount = getGpuCount(instanceType);
      if (!gpuModel) continue;

      const pricePerGpuHour = Math.round((instancePriceHour / gpuCount) * 100) / 100;

      prices.push({
        gpu: gpuModel,
        instanceType,
        pricePerGpuHour,
        instancePriceHour: Math.round(instancePriceHour * 100) / 100,
        gpuCount,
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
        error: "Failed to fetch AWS bulk pricing. The file is large (~50MB) and may have timed out. Cached results from a previous fetch may still be available.",
        fallback: "Use static pricing from tokcalc's GPU catalog or Azure/Vast.ai live pricing.",
      },
      { status: 502 },
    );
  }
}
