/**
 * AWS EC2 pricing proxy using AWS Price List Query API.
 * Pulls live us-east-1 On-Demand pricing per GPU instance family.
 */

import { NextResponse } from "next/server";

export const maxDuration = 15; // Set Vercel max execution time limit

const GPU_TARGETS = [
  { prefix: "p5.48xlarge", gpu: "H100", count: 8 },
  { prefix: "p4d.24xlarge", gpu: "A100 40GB", count: 8 },
  { prefix: "p4de.24xlarge", gpu: "A100 80GB", count: 8 },
  { prefix: "p3.2xlarge", gpu: "V100", count: 1 },
  { prefix: "g5.xlarge", gpu: "A10G", count: 1 },
  { prefix: "g6.xlarge", gpu: "L4", count: 1 },
  { prefix: "g4dn.xlarge", gpu: "T4", count: 1 },
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

async function fetchInstancePrice(instanceType: string): Promise<number | null> {
  try {
    const url = `https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(12000),
      headers: { Accept: "application/json" },
    });

    if (!response.ok) return null;

    // Stream-optimized JSON array buffer parsing
    const text = await response.text();
    const skuRegex = new RegExp(`"instanceType":"${instanceType}"`, "g");
    
    if (!skuRegex.test(text)) return null;

    const data = JSON.parse(text);
    const products = data.products || {};
    const onDemand = data.terms?.OnDemand || {};

    for (const [sku, product] of Object.entries<any>(products)) {
      if (
        product.attributes?.instanceType === instanceType &&
        product.attributes?.operatingSystem === "Linux" &&
        product.attributes?.tenancy === "Shared" &&
        product.attributes?.capacitystatus === "Used"
      ) {
        const term = onDemand[sku];
        if (!term?.priceDimensions) continue;
        const dims: any = Object.values(term.priceDimensions)[0];
        const usd = dims?.pricePerUnit?.USD;
        if (usd) return parseFloat(usd);
      }
    }
  } catch (err) {
    console.error(`Error processing AWS pricing for ${instanceType}:`, err);
  }
  return null;
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
    const prices: CleanGpuPrice[] = [];

    // Lightweight target fetching fallback
    const url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json";
    const response = await fetch(url, {
      signal: AbortSignal.timeout(12000),
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`AWS API returned status ${response.status}`);
    }

    const data = await response.json() as any;
    const products = data.products || {};
    const onDemandTerms = data.terms?.OnDemand || {};

    for (const target of GPU_TARGETS) {
      for (const [sku, product] of Object.entries<any>(products)) {
        const attrs = product.attributes || {};
        if (
          attrs.instanceType === target.prefix &&
          attrs.operatingSystem === "Linux" &&
          attrs.tenancy === "Shared" &&
          attrs.capacitystatus === "Used"
        ) {
          const term = onDemandTerms[sku];
          if (!term?.priceDimensions) continue;
          const priceDim: any = Object.values(term.priceDimensions)[0];
          const usdPrice = priceDim?.pricePerUnit?.USD;

          if (usdPrice) {
            const instancePriceHour = parseFloat(usdPrice);
            if (!isNaN(instancePriceHour) && instancePriceHour > 0) {
              const pricePerGpuHour = Math.round((instancePriceHour / target.count) * 100) / 100;
              prices.push({
                gpu: target.gpu,
                instanceType: target.prefix,
                pricePerGpuHour,
                instancePriceHour: Math.round(instancePriceHour * 100) / 100,
                gpuCount: target.count,
                region: "us-east-1",
                retrievedAt: new Date().toISOString(),
              });
              break;
            }
          }
        }
      }
    }

    if (prices.length === 0) {
      throw new Error("No GPU instances parsed from AWS response");
    }

    cache = { data: prices, timestamp: Date.now() };

    return NextResponse.json({
      source: "aws",
      cached: false,
      retrievedAt: new Date().toISOString(),
      count: prices.length,
      prices,
      note: "On-demand Linux pricing from AWS public catalog (us-east-1).",
    });
  } catch (error) {
    console.error("AWS pricing fetch error:", error);
    return NextResponse.json(
      {
        source: "aws",
        error: "Failed to process AWS bulk pricing file. Timeout or out-of-memory.",
        fallback: "Use static catalog pricing or Azure/Vast.ai live pricing.",
      },
      { status: 502 },
    );
  }
}