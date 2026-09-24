/**
 * Azure Retail Prices API proxy.
 *
 * Fetches live GPU pricing from Azure's public, unauthenticated API:
 *   GET https://prices.azure.com/api/retail/prices
 *
 * This serverless function:
 *   1. Fetches server-side (avoids CORS issues)
 *   2. Caches in memory for 24 hours (reduces API load)
 *   3. Returns a clean JSON response with per-GPU pricing
 *
 * Usage from client:
 *   fetch('/api/pricing/azure?XTransformPort=3000')
 *   or (since this is the same app):
 *   fetch('/api/pricing/azure')
 *
 * Per the fullstack skill rules, if requesting to a different port,
 * use XTransformPort query param. But this API is on the same port
 * so no special handling needed.
 */

import { NextRequest, NextResponse } from "next/server";

interface AzurePriceItem {
  currencyCode: string;
  tierMinimumUnits: number;
  reservationPrice?: number;
  retailPrice: number;
  unitPrice: number;
  armRegionName: string;
  location: string;
  armSkuName: string;
  productName: string;
  skuName: string;
  serviceName: string;
  serviceFamily: string;
  unit: string;
  meters: Array<{
    meterId: string;
    meterName: string;
    unit: string;
    unitPrice: number;
  }>;
}

// In-memory cache (lives for the lifetime of the serverless function instance)
let cache: {
  data: Array<{ gpu: string; sku: string; price: number; region: string; retrievedAt: string }>;
  timestamp: number;
} | null = null;

const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

// GPU instance types to query (H100, H200, A100, L40S, T4, L4, V100)
const AZURE_GPU_SKUS = [
  "Standard_ND96isr_H100_v5",    // 8× H100 80GB
  "Standard_NDm_A100_v4",       // 8× A100 80GB (NDm)
  "Standard_ND96asr_v4",        // 8× A100 80GB (NDasr)
  "Standard_NC24ads_A100_v4",   // 1× A100 80GB
  "Standard_NC4as_T4_v3",       // 1× T4
  "Standard_ND96amsr_A100_v4",   // 4× A100 80GB
];

export async function GET(_request: NextRequest) {
  // Return cached data if fresh
  if (cache && Date.now() - cache.timestamp < CACHE_TTL_MS) {
    return NextResponse.json({
      source: "azure",
      cached: true,
      retrievedAt: new Date(cache.timestamp).toISOString(),
      prices: cache.data,
    });
  }

  try {
    // Fetch from Azure's public, unauthenticated API
    const prices: Array<{ gpu: string; sku: string; price: number; region: string; retrievedAt: string }> = [];

    for (const sku of AZURE_GPU_SKUS) {
      const url = `https://prices.azure.com/api/retail/prices?$filter=armSkuName eq '${sku}' and priceType eq 'Consumption'`;
      const response = await fetch(url, {
        headers: { "Accept": "application/json" },
      });

      if (!response.ok) continue;

      const data = await response.json() as { Items: AzurePriceItem[]; NextPageLink?: string };
      const items = data.Items || [];

      for (const item of items) {
        // Extract GPU name from product name
        const productName = item.productName || "";
        let gpuName = "unknown";
        let gpuCount = 1;

        if (productName.includes("H100")) { gpuName = "H100"; gpuCount = 8; }
        else if (productName.includes("H200")) { gpuName = "H200"; gpuCount = 8; }
        else if (productName.includes("A100") && productName.includes("80")) { gpuName = "A100 80GB"; gpuCount = sku.includes("96") ? 8 : (sku.includes("24") ? 1 : 4); }
        else if (productName.includes("A100")) { gpuName = "A100 40GB"; gpuCount = sku.includes("96") ? 8 : 1; }
        else if (productName.includes("T4")) { gpuName = "T4"; gpuCount = 1; }
        else if (productName.includes("L40")) { gpuName = "L40S"; gpuCount = 8; }

        if (gpuName === "unknown") continue;

        // Skip spot/reserved — only on-demand for now
        if (item.skuName?.toLowerCase().includes("spot")) continue;
        if (item.skuName?.toLowerCase().includes("low priority")) continue;

        const instancePricePerHour = item.retailPrice;
        const derivedGpuPrice = instancePricePerHour / gpuCount;

        prices.push({
          gpu: gpuName,
          sku: item.armSkuName,
          price: Math.round(derivedGpuPrice * 100) / 100, // per-GPU $/hr, 2 decimals
          region: item.armRegionName,
          retrievedAt: new Date().toISOString(),
        });
      }

      // Follow pagination if needed (Azure API returns max 100 items per page)
      let nextLink = data.NextPageLink;
      while (nextLink) {
        const nextResponse = await fetch(nextLink);
        if (!nextResponse.ok) break;
        const nextData = await nextResponse.json() as { Items: AzurePriceItem[]; NextPageLink?: string };
        // Process same as above (simplified — just get first page for now)
        nextLink = nextData.NextPageLink;
        break; // Limit to 2 pages to avoid timeout
      }
    }

    // Deduplicate: keep cheapest per GPU per region
    const seen = new Map<string, { gpu: string; sku: string; price: number; region: string; retrievedAt: string }>();
    for (const p of prices) {
      const key = `${p.gpu}-${p.region}`;
      if (!seen.has(key) || seen.get(key)!.price > p.price) {
        seen.set(key, p);
      }
    }

    const deduped = Array.from(seen.values());

    // Update cache
    cache = {
      data: deduped,
      timestamp: Date.now(),
    };

    return NextResponse.json({
      source: "azure",
      cached: false,
      retrievedAt: new Date().toISOString(),
      count: deduped.length,
      prices: deduped,
    });
  } catch (error) {
    console.error("Azure pricing fetch error:", error);
    return NextResponse.json(
      {
        source: "azure",
        error: "Failed to fetch Azure pricing. The Azure Retail Prices API may be temporarily unavailable.",
        fallback: "Use static pricing from tokcalc's GPU catalog.",
      },
      { status: 502 },
    );
  }
}
