"use client";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Confidence } from "@/lib/token-calc";

const CONFIDENCE_CONFIG: Record<
  Confidence,
  { color: string; label: string; description: string }
> = {
  "official_spec": {
    color: "bg-emerald-500",
    label: "Official spec",
    description:
      "Read directly from a vendor spec sheet, model card, or a HuggingFace config.json / safetensors index. Verify at the cited source URL.",
  },
  "published_price": {
    color: "bg-sky-500",
    label: "Published price",
    description:
      "A published price — cloud GPU $/hr or API token pricing. Prices move, so treat as current-at-publish, not a quote.",
  },
  "observed_run": {
    color: "bg-violet-500",
    label: "Observed run",
    description:
      "An actual measurement: a benchmark run, or a reading you recorded through record_measured. Real data beats any model of it.",
  },
  modeled: {
    color: "bg-emerald-400",
    label: "Modeled",
    description:
      "Derived from a physics-based formula with cited sources (e.g., decode tok/s = HBM_BW × η_mem × quant_eff / model_size). Conditional on assumptions — see formula docs.",
  },
  heuristic: {
    color: "bg-amber-500",
    label: "Heuristic",
    description:
      "Derived from a rule of thumb with known error bars (e.g., continuous batching multiplier default 1.5×, long-context attention O(N²) correction). Treat as ballpark, not exact.",
  },
  user_supplied: {
    color: "bg-zinc-400",
    label: "User-supplied",
    description: "You provided this value directly. We trust your input.",
  },
};

export function ConfidenceBadge({
  confidence,
  className,
}: {
  confidence: Confidence;
  className?: string;
}) {
  const config = CONFIDENCE_CONFIG[confidence];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`Confidence: ${config.label}`}
          className={`inline-flex items-center gap-1 text-[9px] font-medium uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors ${className ?? ""}`}
        >
          <span className={`size-1.5 rounded-full ${config.color}`} />
          {config.label}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[280px] text-xs leading-relaxed">
        <div className="font-semibold mb-1">{config.label}</div>
        <div className="text-muted-foreground">{config.description}</div>
      </TooltipContent>
    </Tooltip>
  );
}

/** Compact version — just the dot, no label. Use in tight spaces like headline cards. */
export function ConfidenceDot({
  confidence,
  className,
}: {
  confidence: Confidence;
  className?: string;
}) {
  const config = CONFIDENCE_CONFIG[confidence];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`Confidence: ${config.label}`}
          className={`inline-flex items-center justify-center ${className ?? ""}`}
        >
          <span className={`size-1.5 rounded-full ${config.color} hover:scale-125 transition-transform`} />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[280px] text-xs leading-relaxed">
        <div className="font-semibold mb-1">{config.label}</div>
        <div className="text-muted-foreground">{config.description}</div>
      </TooltipContent>
    </Tooltip>
  );
}

/** Legend block — show at the bottom of the Calculator tab + in the glossary. */
export function ConfidenceLegend() {
  return (
    <div className="rounded-md border border-border/60 bg-muted/20 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">
        Confidence levels
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px]">
        {(Object.keys(CONFIDENCE_CONFIG) as Confidence[]).map((key) => {
          const config = CONFIDENCE_CONFIG[key];
          return (
            <div key={key} className="flex items-start gap-1.5">
              <span className={`size-1.5 rounded-full ${config.color} mt-1 shrink-0`} />
              <div>
                <div className="font-medium text-foreground">{config.label}</div>
                <div className="text-muted-foreground text-[10px] leading-snug mt-0.5">
                  {config.description.split(". ")[0]}.
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
