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
import { RefreshCw, Zap, AlertTriangle } from "lucide-react";
import { track } from "@/lib/track";

interface AzurePrice {
  gpu: string;
  sku: string;
  price: number;
  region: string;
  retrievedAt: string;
}

interface AzureApiResponse {
  source: string;
  cached: boolean;
  retrievedAt?: string;
  count?: number;
  prices?: AzurePrice[];
  error?: string;
}

export function AzureLivePricing() {
  const [data, setData] = useState<AzureApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPrices = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/pricing/azure");
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
      const json = (await res.json()) as AzureApiResponse;
      setData(json);
      track("fetched_live_pricing", { provider: "azure", count: json.count });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch Azure pricing");
    } finally {
      setLoading(false);
    }
  };

  // Auto-fetch on mount
  useEffect(() => {
    fetchPrices();
  }, []);

  // Find cheapest price per GPU
  const cheapestPerGpu = (() => {
    if (!data?.prices) return [];
    const seen = new Map<string, AzurePrice>();
    for (const p of data.prices) {
      if (!seen.has(p.gpu) || seen.get(p.gpu)!.price > p.price) {
        seen.set(p.gpu, p);
      }
    }
    return Array.from(seen.values()).sort((a, b) => a.price - b.price);
  })();

  // Calculate "as of" label
  const asOfLabel = (() => {
    if (!data?.retrievedAt) return null;
    const ageMs = Date.now() - new Date(data.retrievedAt).getTime();
    if (ageMs < 60000) return "just now";
    if (ageMs < 3600000) return `${Math.floor(ageMs / 60000)}m ago`;
    return `${Math.floor(ageMs / 3600000)}h ago`;
  })();

  return (
    <Card className="border-emerald-500/30 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-sm flex items-center gap-2">
              <Zap className="size-4 text-emerald-500" />
              Live Azure GPU pricing
              <Badge variant="outline" className="text-[9px] gap-1 ml-1">
                <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse" />
                LIVE
              </Badge>
            </CardTitle>
            <CardDescription className="text-xs mt-1">
              Fetched from Azure Retail Prices API (public, unauthenticated).
              {asOfLabel && ` Last updated ${asOfLabel}.`}
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchPrices}
            disabled={loading}
            className="gap-1.5 h-8"
          >
            <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
            <span className="text-xs">Refresh</span>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading && !data && (
          <div className="flex items-center justify-center py-8 text-xs text-muted-foreground">
            <RefreshCw className="size-4 animate-spin mr-2" />
            Fetching live Azure prices...
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 p-3 rounded-md border border-amber-500/30 bg-amber-500/5 text-xs">
            <AlertTriangle className="size-3.5 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-amber-600 dark:text-amber-400">
                Couldn&apos;t fetch live prices
              </p>
              <p className="text-muted-foreground mt-0.5">{error}</p>
              <p className="text-muted-foreground mt-1">
                Static estimates from tokcalc&apos;s catalog are shown below.
              </p>
            </div>
          </div>
        )}

        {data?.error && (
          <div className="flex items-start gap-2 p-3 rounded-md border border-amber-500/30 bg-amber-500/5 text-xs">
            <AlertTriangle className="size-3.5 text-amber-500 shrink-0 mt-0.5" />
            <p className="text-muted-foreground">{data.error}</p>
          </div>
        )}

        {cheapestPerGpu.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-3 py-1.5">GPU</th>
                  <th className="text-left px-3 py-1.5">Azure SKU</th>
                  <th className="text-right px-3 py-1.5">$/GPU-hr (cheapest)</th>
                  <th className="text-left px-3 py-1.5">Region</th>
                  <th className="text-left px-3 py-1.5">As of</th>
                </tr>
              </thead>
              <tbody>
                {cheapestPerGpu.map((p, i) => (
                  <tr key={i} className="border-t border-border/40 hover:bg-muted/20">
                    <td className="px-3 py-1.5 font-medium">{p.gpu}</td>
                    <td className="px-3 py-1.5 text-muted-foreground font-mono text-[10px]">{p.sku}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-emerald-500 font-semibold">
                      ${p.price}/hr
                    </td>
                    <td className="px-3 py-1.5 text-muted-foreground">{p.region}</td>
                    <td className="px-3 py-1.5 text-muted-foreground text-[10px]">{asOfLabel}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed">
              Per-GPU price derived from Azure instance price ÷ GPU count.
              On-demand rates only (spot/reserved excluded).
              Prices may vary by region, availability zone, and commitment.
              Source: <a
                href="https://prices.azure.com/api/retail/prices"
                target="_blank"
                rel="noopener noreferrer"
                className="text-emerald-500 hover:underline"
              >Azure Retail Prices API ↗</a>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
