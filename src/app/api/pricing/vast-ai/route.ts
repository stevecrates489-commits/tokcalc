/**
 * Vast.ai marketplace pricing proxy.
 *
 * Fetches live spot-market GPU offers from Vast.ai's public API:
 *   GET https://cloud.vast.ai/api/v0/bundles/
 *
 * This serverless function:
 *   1. Fetches server-side (avoids CORS issues)
 *   2. Filters for GPU-only offers
 *   3. Deduplicates by GPU model (keeps cheapest available)
 *   4. Caches in memory for 5 minutes (spot prices change rapidly)
 *
 * Per Perplexity research (Prompt #3):
 *   "Vast.ai is the best fit for dynamic spot/marketplace pricing"
 *   "Official offer-search REST API / OpenAPI"
 *   "Treat every returned offer as temporary market data"
 */

import { NextRequest, NextResponse } from "next/server";

interface VastAiBundle {
  id: number;
  gpu_name?: string;
  num_gpus?: number;
  gpu_ram?: number;
  dph_total?: number;     // total $/hour
  reliability?: number;
  geolocation?: string;
  inetname?: string;
  dlperf?: number;       // download performance
  cuda_max_good?: number; // CUDA version
  total_flops?: number;
  cpu_ram?: number;
  cpu_cores?: number;
}

interface CleanOffer {
  gpu: string;
  gpuCount: number;
  vramGb: number;
  pricePerGpuHour: number;
  totalPriceHour: number;
  reliability: number;
  region: string;
  bundleId: number;
}

// In-memory cache (5 minute TTL — spot prices change rapidly)
let cache: { data: CleanOffer[]; timestamp: number } | null = null;
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

// GPUs we care about — use short names for flexible matching
// (e.g., "H100" matches both "H100 SXM" and "H100 NVL")
const TARGET_GPUS = [
  "H100", "H200", "B200",
  "A100", "A6000",
  "RTX 4090", "RTX 3090", "RTX 4080", "RTX 3080",
  "RTX 5090", "RTX 5080", "RTX 5070",
  "L40", "L4",
  "V100",
  "T4",
];

export async function GET(_request: NextRequest) {
  // Return cached data if fresh
  if (cache && Date.now() - cache.timestamp < CACHE_TTL_MS) {
    return NextResponse.json({
      source: "vast_ai",
      cached: true,
      retrievedAt: new Date(cache.timestamp).toISOString(),
      count: cache.data.length,
      offers: cache.data,
      note: "Spot-market prices. Updated every 5 minutes. Individual offers may be gone by the time you try to rent.",
    });
  }

  try {
    // Vast.ai API: search for GPU offers, sorted by cheapest first
    // The q parameter filters by GPU name, order sorts by price ascending
    const url = "https://cloud.vast.ai/api/v0/bundles/";
    const response = await fetch(url, {
      headers: { "Accept": "application/json" },
      // Timeout after 10 seconds
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`Vast.ai API returned ${response.status}`);
    }

    const data = await response.json() as { offers?: VastAiBundle[] };
    const bundles = data.offers || [];

    // Filter and clean offers
    const offers: CleanOffer[] = [];

    for (const bundle of bundles) {
      if (!bundle.gpu_name || !bundle.dph_total || bundle.dph_total <= 0) continue;
      if (!bundle.num_gpus || bundle.num_gpus <= 0) continue;

      const gpuName = bundle.gpu_name.trim();
      const gpuCount = bundle.num_gpus;
      const totalPriceHour = bundle.dph_total;
      const pricePerGpuHour = Math.round((totalPriceHour / gpuCount) * 100) / 100;

      // Check if this GPU is in our target list (bidirectional matching)
      const isTarget = TARGET_GPUS.some(
        (target) =>
          gpuName.toLowerCase().includes(target.toLowerCase()) ||
          target.toLowerCase().includes(gpuName.toLowerCase()),
      );
      if (!isTarget) continue;

      offers.push({
        gpu: gpuName,
        gpuCount,
        vramGb: bundle.gpu_ram || 0,
        pricePerGpuHour,
        totalPriceHour: Math.round(totalPriceHour * 100) / 100,
        reliability: bundle.reliability || 0,
        region: bundle.geolocation || "unknown",
        bundleId: bundle.id,
      });
    }

    // Deduplicate: keep cheapest per-GPU price for each GPU model
    const cheapestPerGpu = new Map<string, CleanOffer>();
    for (const offer of offers) {
      const key = offer.gpu;
      if (!cheapestPerGpu.has(key) || cheapestPerGpu.get(key)!.pricePerGpuHour > offer.pricePerGpuHour) {
        cheapestPerGpu.set(key, offer);
      }
    }

    const deduped = Array.from(cheapestPerGpu.values()).sort(
      (a, b) => a.pricePerGpuHour - b.pricePerGpuHour,
    );

    // Update cache
    cache = {
      data: deduped,
      timestamp: Date.now(),
    };

    return NextResponse.json({
      source: "vast_ai",
      cached: false,
      retrievedAt: new Date().toISOString(),
      count: deduped.length,
      offers: deduped,
      note: "Spot-market prices from Vast.ai marketplace. Individual offers may be gone by the time you try to rent. Reliability score indicates host trustworthiness (0-1).",
    });
  } catch (error) {
    console.error("Vast.ai pricing fetch error:", error);
    return NextResponse.json(
      {
        source: "vast_ai",
        error: "Failed to fetch Vast.ai marketplace pricing. The API may be temporarily unavailable or rate-limited.",
        fallback: "Use static pricing from tokcalc's GPU catalog.",
      },
      { status: 502 },
    );
  }
}
