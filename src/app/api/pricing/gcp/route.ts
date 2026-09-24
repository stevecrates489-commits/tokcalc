/**
 * Google Cloud GPU pricing proxy.
 *
 * Uses the Google Cloud Billing Catalog API:
 *   GET https://cloudbilling.googleapis.com/v1/services/{SERVICE_ID}/skus?key=API_KEY
 *
 * Requires GCP_API_KEY environment variable.
 * Set on Vercel: Settings → Environment Variables → GCP_API_KEY
 *
 * This serverless function:
 *   1. Fetches Compute Engine SKUs from the Cloud Billing Catalog API
 *   2. Filters for GPU instance types (a3, a4 — H100, H200, B200)
 *   3. Extracts on-demand pricing
 *   4. Caches in memory for 24 hours
 *
 * The API key is stored as an env var — NOT hardcoded in source code.
 * This is important because tokcalc is open source on GitHub.
 */

import { NextResponse } from "next/server";

// GCP GPU instance types to look for in SKU descriptions
const GPU_PATTERNS = [
  { pattern: "H100", gpu: "H100" },
  { pattern: "H200", gpu: "H200" },
  { pattern: "B200", gpu: "B200" },
  { pattern: "A100", gpu: "A100" },
  { pattern: "L4", gpu: "L4" },
  { pattern: "T4", gpu: "T4" },
  { pattern: "L40", gpu: "L40S" },
  { pattern: "V100", gpu: "V100" },
];

// GCP instance type → GPU count mapping
const GCP_GPU_COUNTS: Record<string, number> = {
  "a3-highgpu-8g": 8, "a3-highgpu-2g": 2,
  "a3-edgegpu-8g": 8, "a3-edgegpu-2g": 2,
  "a4-highgpu-8g": 8,
  "a2-highgpu-1g": 1, "a2-highgpu-2g": 2, "a2-highgpu-4g": 4, "a2-highgpu-8g": 8,
  "n1-standard-4": 1, // T4 usually 1 per instance
  "g2-standard-4": 1, "g2-standard-8": 1, "g2-standard-12": 1, "g2-standard-16": 1, "g2-standard-24": 4, "g2-standard-32": 1, "g2-standard-48": 8,
};

let cache: { data: CleanGcpPrice[]; timestamp: number } | null = null;
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

interface CleanGcpPrice {
  gpu: string;
  instanceType: string;
  pricePerGpuHour: number;
  instancePriceHour: number;
  gpuCount: number;
  region: string;
  retrievedAt: string;
}

export async function GET() {
  const apiKey = process.env.GCP_API_KEY;

  if (!apiKey) {
    return NextResponse.json(
      {
        source: "gcp",
        error: "GCP_API_KEY environment variable is not set. Add it on Vercel: Settings → Environment Variables → GCP_API_KEY",
        fallback: "Use Azure or Vast.ai live pricing instead.",
      },
      { status: 503 },
    );
  }

  if (cache && Date.now() - cache.timestamp < CACHE_TTL_MS) {
    return NextResponse.json({
      source: "gcp",
      cached: true,
      retrievedAt: new Date(cache.timestamp).toISOString(),
      count: cache.data.length,
      prices: cache.data,
    });
  }

  try {
    // Step 1: Get the list of services to find Compute Engine's service ID
    const servicesUrl = `https://cloudbilling.googleapis.com/v1/services?key=${apiKey}&pageSize=100`;
    const servicesResponse = await fetch(servicesUrl, {
      signal: AbortSignal.timeout(10000),
    });

    if (!servicesResponse.ok) {
      throw new Error(`GCP Services API returned ${servicesResponse.status}`);
    }

    const servicesData = await servicesResponse.json() as {
      services?: Array<{ serviceId: string; displayName: string }>;
    };

    // Find Compute Engine service
    const computeService = servicesData.services?.find(
      (s) => s.displayName?.toLowerCase().includes("compute engine"),
    );

    if (!computeService) {
      throw new Error("Compute Engine service not found in GCP billing catalog");
    }

    // Step 2: Fetch SKUs for Compute Engine
    const skusUrl = `https://cloudbilling.googleapis.com/v1/services/${computeService.serviceId}/skus?key=${apiKey}&pageSize=5000`;
    const skusResponse = await fetch(skusUrl, {
      signal: AbortSignal.timeout(15000),
    });

    if (!skusResponse.ok) {
      throw new Error(`GCP SKUs API returned ${skusResponse.status}`);
    }

    const skusData = await skusResponse.json() as {
      skus?: Array<{
        skuId: string;
        description: string;
        pricingInfo?: Array<{
          pricingExpression?: {
            tieredRates?: Array<{
              unitPrice?: { units?: string; nanos?: number };
            }>;
          };
        }>;
        serviceRegions?: string[];
        geoTaxonomy?: { regions?: string[] };
      }>;
      nextPagePageToken?: string;
    };

    const skus = skusData.skus || [];

    const prices: CleanGcpPrice[] = [];

    for (const sku of skus) {
      const description = sku.description || "";

      // Check if this SKU matches any GPU pattern
      const gpuMatch = GPU_PATTERNS.find((p) => description.includes(p.pattern));
      if (!gpuMatch) continue;

      // Skip preemptible/spot/reserved
      if (description.toLowerCase().includes("preemptible")) continue;
      if (description.toLowerCase().includes("spot")) continue;
      if (description.toLowerCase().includes("reserved")) continue;
      if (description.toLowerCase().includes("commitment")) continue;

      // Skip non-GPU SKUs that happen to mention GPU in text
      if (!description.toLowerCase().includes("gpu") && !description.toLowerCase().includes("premium")) {
        // Only include if the instance type is in our known GPU instances
        const instanceMatch = Object.keys(GCP_GPU_COUNTS).find(
          (it) => description.toLowerCase().includes(it.toLowerCase()),
        );
        if (!instanceMatch) continue;
      }

      // Extract price
      const pricingInfo = sku.pricingInfo?.[0];
      const tieredRates = pricingInfo?.pricingExpression?.tieredRates;
      const firstRate = tieredRates?.[0];
      const unitPrice = firstRate?.unitPrice;

      if (!unitPrice) continue;

      // GCP prices are in USD with nanos precision
      const dollars = parseFloat(unitPrice.units || "0");
      const nanos = (unitPrice.nanos || 0) / 1e9;
      const pricePerHour = dollars + nanos;

      if (isNaN(pricePerHour) || pricePerHour <= 0) continue;

      // Try to determine instance type and GPU count
      let instanceType = "unknown";
      let gpuCount = 1;

      for (const [it, count] of Object.entries(GCP_GPU_COUNTS)) {
        if (description.toLowerCase().includes(it.toLowerCase())) {
          instanceType = it;
          gpuCount = count;
          break;
        }
      }

      const pricePerGpuHour = Math.round((pricePerHour / gpuCount) * 100) / 100;

      prices.push({
        gpu: gpuMatch.gpu,
        instanceType,
        pricePerGpuHour,
        instancePriceHour: Math.round(pricePerHour * 100) / 100,
        gpuCount,
        region: sku.serviceRegions?.[0] || sku.geoTaxonomy?.regions?.[0] || "global",
        retrievedAt: new Date().toISOString(),
      });
    }

    // Deduplicate: keep cheapest per-GPU price per GPU model
    const seen = new Map<string, CleanGcpPrice>();
    for (const p of prices) {
      const key = p.gpu;
      if (!seen.has(key) || seen.get(key)!.pricePerGpuHour > p.pricePerGpuHour) {
        seen.set(key, p);
      }
    }

    const deduped = Array.from(seen.values()).sort((a, b) => a.pricePerGpuHour - b.pricePerGpuHour);

    cache = { data: deduped, timestamp: Date.now() };

    return NextResponse.json({
      source: "gcp",
      cached: false,
      retrievedAt: new Date().toISOString(),
      count: deduped.length,
      prices: deduped,
      note: "On-demand pricing from Google Cloud Billing Catalog API. Region varies by SKU. Requires GCP_API_KEY env var.",
    });
  } catch (error) {
    console.error("GCP pricing fetch error:", error);
    return NextResponse.json(
      {
        source: "gcp",
        error: "Failed to fetch GCP pricing. Check that GCP_API_KEY is set and valid.",
        fallback: "Use Azure or Vast.ai live pricing instead.",
      },
      { status: 502 },
    );
  }
}
