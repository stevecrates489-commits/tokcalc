"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Zap, TrendingDown, Cloud } from "lucide-react";
import { track } from "@/lib/track";

interface UnifiedPrice {
  gpu: string;
  pricePerGpuHour: number;
  provider: string;
  region: string;
  retrievedAt: string;
}

interface ProviderResponse {
  source: string;
  cached: boolean;
  retrievedAt?: string;
  count?: number;
  prices?: Array<{ gpu: string; pricePerGpuHour?: number; price?: number; region?: string; retrievedAt?: string }>;
  error?: string;
}

export function LivePricingComparison() {
  const [prices, setPrices] = useState<UnifiedPrice[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  const fetchAll = async () => {
    setLoading(true);
    setErrors([]);
    const allPrices: UnifiedPrice[] = [];
    const newErrors: string[] = [];
    let latestTimestamp: string | null = null;

    // Fetch all providers in parallel
    const providers = [
      { url: "/api/pricing/azure", name: "Azure" },
      { url: "/api/pricing/vast-ai", name: "Vast.ai" },
      { url: "/api/pricing/aws", name: "AWS" },
      { url: "/api/pricing/gcp", name: "GCP" },
    ];

    const results = await Promise.allSettled(
      providers.map(async (p) => {
        const res = await fetch(p.url);
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${res.status}`);
        }
        return { provider: p.name, data: (await res.json()) as ProviderResponse };
      }),
    );

    for (const result of results) {
      if (result.status === "fulfilled") {
        const { provider, data } = result.value;
        const providerPrices = data.prices || [];

        for (const p of providerPrices) {
          const pricePerGpuHour = p.pricePerGpuHour ?? p.price ?? 0;
          if (pricePerGpuHour > 0) {
            allPrices.push({
              gpu: p.gpu,
              pricePerGpuHour,
              provider,
              region: p.region || "—",
              retrievedAt: p.retrievedAt || data.retrievedAt || new Date().toISOString(),
            });
          }
        }

        if (data.retrievedAt && (!latestTimestamp || new Date(data.retrievedAt) > new Date(latestTimestamp))) {
          latestTimestamp = data.retrievedAt;
        }

        track("fetched_live_pricing", { provider, count: providerPrices.length });
      } else if (result.status === "rejected") {
        const providerName = providers[results.indexOf(result)].name;
        newErrors.push(`${providerName}: ${result.reason?.message || "fetch failed"}`);
      }
    }

    setPrices(allPrices);
    setLastUpdated(latestTimestamp);
    setErrors(newErrors);
    setLoading(false);
  };

  useEffect(() => {
    fetchAll();
  }, []);

  // Deduplicate: keep cheapest per GPU per provider
  const cheapestPerGpu = (() => {
    const seen = new Map<string, UnifiedPrice>();
    for (const p of prices) {
      const key = `${p.gpu}-${p.provider}`;
      if (!seen.has(key) || seen.get(key)!.pricePerGpuHour > p.pricePerGpuHour) {
        seen.set(key, p);
      }
    }

    // Group by GPU, sort providers by price
    const byGpu = new Map<string, UnifiedPrice[]>();
    for (const p of Array.from(seen.values())) {
      if (!byGpu.has(p.gpu)) byGpu.set(p.gpu, []);
      byGpu.get(p.gpu)!.push(p);
    }

    // Sort each GPU's providers by price
    for (const arr of byGpu.values()) {
      arr.sort((a, b) => a.pricePerGpuHour - b.pricePerGpuHour);
    }

    // Sort GPUs by their cheapest provider
    return Array.from(byGpu.entries())
      .map(([gpu, providers]) => ({ gpu, providers }))
      .sort((a, b) => a.providers[0].pricePerGpuHour - b.providers[0].pricePerGpuHour);
  })();

  const asOfLabel = (() => {
    if (!lastUpdated) return null;
    const ageMs = Date.now() - new Date(lastUpdated).getTime();
    if (ageMs < 60000) return "just now";
    if (ageMs < 3600000) return `${Math.floor(ageMs / 60000)}m ago`;
    return `${Math.floor(ageMs / 3600000)}h ago`;
  })();

  const providerColors: Record<string, string> = {
    Azure: "text-blue-500",
    "Vast.ai": "text-amber-500",
    AWS: "text-orange-500",
    GCP: "text-red-500",
  };

  return (
    <Card className="border-emerald-500/30 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-sm flex items-center gap-2">
              <Cloud className="size-4 text-emerald-500" />
              Live GPU price comparison — all providers
              <Badge variant="outline" className="text-[9px] gap-1 ml-1">
                <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
                LIVE
              </Badge>
            </CardTitle>
            <CardDescription className="text-xs mt-1">
              Cheapest available GPU price across Azure, AWS, GCP, and Vast.ai marketplace.
              {asOfLabel && ` Last updated ${asOfLabel}.`}
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchAll}
            disabled={loading}
            className="gap-1.5 h-8"
          >
            <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
            <span className="text-xs">Refresh</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading && prices.length === 0 && (
          <div className="flex items-center justify-center py-8 text-xs text-muted-foreground">
            <RefreshCw className="size-4 animate-spin mr-2" />
            Fetching live prices from Azure, Vast.ai, AWS, and GCP...
          </div>
        )}

        {errors.length > 0 && (
          <div className="mb-3 p-2 rounded-md border border-amber-500/30 bg-amber-500/5 text-[10px] text-amber-600 dark:text-amber-400">
            ⚠ {errors.join(" · ")}
          </div>
        )}

        {cheapestPerGpu.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-3 py-1.5">GPU</th>
                  <th className="text-right px-3 py-1.5">Cheapest $/GPU-hr</th>
                  <th className="text-left px-3 py-1.5">Provider</th>
                  <th className="text-left px-3 py-1.5">All providers</th>
                </tr>
              </thead>
              <tbody>
                {cheapestPerGpu.map(({ gpu, providers }) => {
                  const cheapest = providers[0];
                  return (
                    <tr key={gpu} className="border-t border-border/40 hover:bg-muted/20">
                      <td className="px-3 py-1.5 font-medium">{gpu}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-emerald-500 font-semibold">
                        ${cheapest.pricePerGpuHour}/hr
                      </td>
                      <td className="px-3 py-1.5">
                        <span className={providerColors[cheapest.provider] || "text-muted-foreground"}>
                          {cheapest.provider}
                        </span>
                        <span className="text-muted-foreground text-[10px] ml-1">{cheapest.region}</span>
                      </td>
                      <td className="px-3 py-1.5 text-[10px] text-muted-foreground">
                        {providers.map((p, i) => (
                          <span key={i}>
                            {i > 0 && " · "}
                            <span className={providerColors[p.provider] || "text-muted-foreground"}>
                              {p.provider} ${p.pricePerGpuHour}
                            </span>
                          </span>
                        ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed">
              On-demand prices (spot/reserved excluded). Vast.ai shows spot-market prices which change rapidly.
              GCP requires <code>GCP_API_KEY</code> env var on Vercel. AWS uses public bulk pricing files (no auth).
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
