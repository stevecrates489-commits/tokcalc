/**
 * Google Cloud GPU pricing proxy.
 *
 * Uses the Google Cloud Billing Catalog API:
 *   GET https://cloudbilling.googleapis.com/v1/services/{SERVICE_ID}/skus?key=API_KEY
 *
 * Requires GCP_API_KEY environment variable.
 * Set on Vercel: Settings → Environment Variables → GCP_API_KEY
 */

import { NextResponse } from "next/server";

// Direct Service ID for Google Compute Engine in GCP Cloud Billing API
const COMPUTE_ENGINE_SERVICE_ID = "6F81-5844-456A";

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
  "n1-standard-4": 1,
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
    // Resolve the Compute Engine service ID. Prefer the static well-known ID
    // for speed; if it 404s (e.g. Google changed IDs, or the API key's project
    // doesn't expose this service), fall back to listing services and picking
    // the one whose displayName is "Compute Engine".
    let serviceId = COMPUTE_ENGINE_SERVICE_ID;
    let skusResponse = await fetch(
      `https://cloudbilling.googleapis.com/v1/services/${serviceId}/skus?key=${apiKey}&pageSize=5000`,
      { signal: AbortSignal.timeout(15000) },
    );

    if (skusResponse.status === 404) {
      console.warn(`[gcp-pricing] Static service ID ${serviceId} returned 404, attempting dynamic lookup…`);
      const listRes = await fetch(
        `https://cloudbilling.googleapis.com/v1/services?key=${apiKey}&pageSize=200`,
        { signal: AbortSignal.timeout(15000) },
      );
      if (listRes.ok) {
        const list = (await listRes.json()) as { services?: Array<{ serviceId: string; displayName: string }> };
        const ce = (list.services || []).find((s) => /compute engine/i.test(s.displayName));
        if (ce) {
          serviceId = ce.serviceId;
          console.warn(`[gcp-pricing] Resolved Compute Engine service ID dynamically: ${serviceId}`);
          skusResponse = await fetch(
            `https://cloudbilling.googleapis.com/v1/services/${serviceId}/skus?key=${apiKey}&pageSize=5000`,
            { signal: AbortSignal.timeout(15000) },
          );
        }
      }
    }

    if (!skusResponse.ok) {
      const errBody = await skusResponse.text().catch(() => "");
      throw new Error(
        `GCP SKUs API returned ${skusResponse.status} for service ${serviceId}. ` +
          `This usually means the GCP_API_KEY is missing the Cloud Billing API scope, ` +
          `the API isn't enabled on the key's GCP project, or the key has IP restrictions. ` +
          `Raw response: ${errBody.slice(0, 200)}`,
      );
    }

    const skusData = (await skusResponse.json()) as {
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
    };

    const skus = skusData.skus || [];
    const prices: CleanGcpPrice[] = [];

    for (const sku of skus) {
      const description = sku.description || "";

      const gpuMatch = GPU_PATTERNS.find((p) => description.includes(p.pattern));
      if (!gpuMatch) continue;

      // Skip non-on-demand SKUs
      const lowerDesc = description.toLowerCase();
      if (
        lowerDesc.includes("preemptible") ||
        lowerDesc.includes("spot") ||
        lowerDesc.includes("reserved") ||
        lowerDesc.includes("commitment")
      ) {
        continue;
      }

      if (!lowerDesc.includes("gpu") && !lowerDesc.includes("premium")) {
        const instanceMatch = Object.keys(GCP_GPU_COUNTS).find((it) =>
          lowerDesc.includes(it.toLowerCase()),
        );
        if (!instanceMatch) continue;
      }

      const pricingInfo = sku.pricingInfo?.[0];
      const tieredRates = pricingInfo?.pricingExpression?.tieredRates;
      const firstRate = tieredRates?.[0];
      const unitPrice = firstRate?.unitPrice;

      if (!unitPrice) continue;

      const dollars = parseFloat(unitPrice.units || "0");
      const nanos = (unitPrice.nanos || 0) / 1e9;
      const pricePerHour = dollars + nanos;

      if (isNaN(pricePerHour) || pricePerHour <= 0) continue;

      let instanceType = "unknown";
      let gpuCount = 1;

      for (const [it, count] of Object.entries(GCP_GPU_COUNTS)) {
        if (lowerDesc.includes(it.toLowerCase())) {
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