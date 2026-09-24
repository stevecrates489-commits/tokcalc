/**
 * Price record type for live cloud GPU pricing.
 *
 * Based on the Perplexity research brief (Prompt #3):
 * "Use an append-only historical model. A price should never be overwritten
 * without retaining the prior record."
 *
 * Phase 1: Azure Retail Prices API (unauthenticated, easiest)
 * Phase 2: Vast.ai marketplace API (dynamic spot pricing)
 * Phase 3: AWS / GCP / OCI official APIs (authenticated)
 */

export type PriceProvider =
  | "azure"
  | "aws"
  | "gcp"
  | "oci"
  | "vast_ai"
  | "runpod"
  | "lambda"
  | "modal"
  | "tensor"
  | "default";

export type PurchaseOption =
  | "on_demand"
  | "spot"
  | "reserved"
  | "savings_plan"
  | "marketplace"
  | "serverless";

export type BillingUnit =
  | "instance_hour"
  | "gpu_hour"
  | "gpu_second";

export interface PriceRecord {
  id: string;
  provider: PriceProvider;
  productFamily: "cloud_vm" | "gpu_marketplace";

  /** Cloud SKU identifier (e.g., "Standard_ND96isr_H100_v5") */
  providerSku: string;
  providerSkuName: string;

  /** GPU info extracted from the SKU */
  acceleratorVendor?: string;
  acceleratorModel?: string;
  acceleratorCount?: number;
  acceleratorMemoryGb?: number;

  /** Region/zone */
  region?: string;
  locationLabel?: string;

  purchaseOption: PurchaseOption;
  billingUnit: BillingUnit;

  /** Price in USD */
  amount: number;
  currency: "USD";

  /** Per-GPU derived price (if instance-level, divide by GPU count) */
  derivedPricePerGpuHour?: number;
  isDerived: boolean;
  derivationFormula?: string;

  /** Validity window */
  effectiveFrom?: string;
  effectiveTo?: string;
  retrievedAt: string;

  /** Provenance */
  sourceType: "official_api" | "official_price_page" | "manual_verified";
  sourceUrl: string;

  /** Display confidence */
  confidence: "high" | "medium" | "low";
}

/**
 * Match a GPU from tokcalc's catalog to a cloud provider's SKU.
 * Returns the best guess based on GPU name + VRAM.
 */
export function matchGpuToCloudSku(
  gpuName: string,
  gpuVramGb: number,
  provider: PriceProvider,
): { sku: string; skuName: string } | null {
  const name = gpuName.toLowerCase();

  // Azure SKU patterns
  if (provider === "azure") {
    if (name.includes("h100")) return { sku: "Standard_ND96isr_H100_v5", skuName: "ND H100 v5 (96 GPU = 8× H100)" };
    if (name.includes("h200")) return { sku: "Standard_ND-H200-v5", skuName: "ND H200 v5" };
    if (name.includes("a100") && gpuVramGb >= 80) return { sku: "Standard_ND96asr_v4", skuName: "ND A100 v4 (8× A100 80GB)" };
    if (name.includes("a100") && gpuVramGb < 80) return { sku: "Standard_ND96amsr_A100_v4", skuName: "ND A100 v4 (40GB)" };
    if (name.includes("v100")) return { sku: "Standard_NC24ads_A100_v4", skuName: "NC A100 v4" };
    if (name.includes("t4")) return { sku: "Standard_NC4as_T4_v3", skuName: "NC T4 v3" };
    if (name.includes("l40s") || name.includes("l40")) return { sku: "Standard_NDm_A100_v4", skuName: "NDm L40S" };
  }

  // AWS SKU patterns
  if (provider === "aws") {
    if (name.includes("h100")) return { sku: "p5.48xlarge", skuName: "p5.48xlarge (8× H100)" };
    if (name.includes("h200")) return { sku: "p5e.48xlarge", skuName: "p5e.48xlarge (8× H200)" };
    if (name.includes("b200")) return { sku: "p6.48xlarge", skuName: "p6.48xlarge (8× B200)" };
    if (name.includes("a100") && gpuVramGb >= 80) return { sku: "p4de.24xlarge", skuName: "p4de.24xlarge (8× A100 80GB)" };
    if (name.includes("a100") && gpuVramGb < 80) return { sku: "p4d.24xlarge", skuName: "p4d.24xlarge (8× A100 40GB)" };
    if (name.includes("l40s")) return { sku: "g6.48xlarge", skuName: "g6.48xlarge (8× L40S)" };
    if (name.includes("l4")) return { sku: "g6.xlarge", skuName: "g6.xlarge (1× L4)" };
    if (name.includes("t4")) return { sku: "g4dn.12xlarge", skuName: "g4dn.12xlarge (4× T4)" };
  }

  return null;
}

/**
 * Format a price record for display in the UI.
 */
export function formatPriceRecord(r: PriceRecord): {
  displayPrice: string;
  isLive: boolean;
  ageLabel: string;
} {
  const perGpu = r.derivedPricePerGpuHour ?? r.amount;
  const displayPrice = `$${perGpu.toFixed(2)}/hr`;

  // "Live" if retrieved within last 24 hours
  const ageMs = Date.now() - new Date(r.retrievedAt).getTime();
  const isLive = ageMs < 24 * 60 * 60 * 1000;

  let ageLabel: string;
  if (ageMs < 60 * 1000) ageLabel = "just now";
  else if (ageMs < 60 * 60 * 1000) ageLabel = `${Math.floor(ageMs / 60000)}m ago`;
  else if (ageMs < 24 * 60 * 60 * 1000) ageLabel = `${Math.floor(ageMs / 3600000)}h ago`;
  else ageLabel = `${Math.floor(ageMs / 86400000)}d ago`;

  return { displayPrice, isLive, ageLabel };
}
