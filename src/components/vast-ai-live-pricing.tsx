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
import { RefreshCw, Zap, AlertTriangle, TrendingDown } from "lucide-react";
import { track } from "@/lib/track";

interface VastAiOffer {
  gpu: string;
  gpuCount: number;
  vramGb: number;
  pricePerGpuHour: number;
  totalPriceHour: number;
  reliability: number;
  region: string;
  bundleId: number;
}

interface VastAiApiResponse {
  source: string;
  cached: boolean;
  retrievedAt?: string;
  count?: number;
  offers?: VastAiOffer[];
  note?: string;
  error?: string;
}

export function VastAiLivePricing() {
  const [data, setData] = useState<VastAiApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPrices = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/pricing/vast-ai");
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
      const json = (await res.json()) as VastAiApiResponse;
      setData(json);
      track("fetched_live_pricing", { provider: "vast_ai", count: json.count });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch Vast.ai pricing");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrices();
  }, []);

  const asOfLabel = (() => {
    if (!data?.retrievedAt) return null;
    const ageMs = Date.now() - new Date(data.retrievedAt).getTime();
    if (ageMs < 60000) return "just now";
    if (ageMs < 3600000) return `${Math.floor(ageMs / 60000)}m ago`;
    return `${Math.floor(ageMs / 3600000)}h ago`;
  })();

  const offers = data?.offers || [];

  return (
    <Card className="border-amber-500/30 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingDown className="size-4 text-amber-500" />
              Vast.ai marketplace — spot prices
              <Badge variant="outline" className="text-[9px] gap-1 ml-1">
                <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
                LIVE
              </Badge>
            </CardTitle>
            <CardDescription className="text-xs mt-1">
              Cheapest available GPU offers on Vast.ai marketplace right now.
              {asOfLabel && ` Last updated ${asOfLabel}.`}
              {" "}Spot prices change rapidly — individual offers may be gone by the time you rent.
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
            Fetching Vast.ai marketplace offers...
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 p-3 rounded-md border border-amber-500/30 bg-amber-500/5 text-xs">
            <AlertTriangle className="size-3.5 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-amber-600 dark:text-amber-400">
                Couldn&apos;t fetch marketplace prices
              </p>
              <p className="text-muted-foreground mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {offers.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-3 py-1.5">GPU</th>
                  <th className="text-right px-3 py-1.5">$/GPU-hr</th>
                  <th className="text-right px-3 py-1.5">GPU count</th>
                  <th className="text-right px-3 py-1.5">Total $/hr</th>
                  <th className="text-right px-3 py-1.5">VRAM</th>
                  <th className="text-right px-3 py-1.5">Reliability</th>
                  <th className="text-left px-3 py-1.5">Region</th>
                </tr>
              </thead>
              <tbody>
                {offers.slice(0, 15).map((o, i) => (
                  <tr key={i} className="border-t border-border/40 hover:bg-muted/20">
                    <td className="px-3 py-1.5 font-medium">{o.gpu}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-amber-500 font-semibold">
                      ${o.pricePerGpuHour}/hr
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">{o.gpuCount}×</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">${o.totalPriceHour}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">{o.vramGb}GB</td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      <span className={o.reliability > 0.9 ? "text-emerald-500" : o.reliability > 0.7 ? "text-amber-500" : "text-red-500"}>
                        {(o.reliability * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-muted-foreground text-[10px]">{o.region || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed">
              Spot-market prices from Vast.ai marketplace.
              Reliability score (0-100%) indicates host trustworthiness.
              Total $/hr is the full bundle price (all GPUs + CPU + RAM).
              Source: <a
                href="https://vast.ai/pricing"
                target="_blank"
                rel="noopener noreferrer"
                className="text-amber-500 hover:underline"
              >Vast.ai marketplace API ↗</a>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
