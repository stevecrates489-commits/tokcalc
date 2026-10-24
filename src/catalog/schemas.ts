/**
 * Zod schemas for the catalog records.
 *
 * These are the AUTHORITATIVE definition of what a record may contain.
 * `scripts/catalog-validate.ts` runs them in CI, so an invalid record fails
 * the build instead of shipping.
 *
 * This module is intentionally NOT imported by the runtime loader: zod would
 * then be bundled into the browser for every visitor. The loader trusts the
 * records because CI enforces these schemas, and does its own cheap
 * integrity checks (unique ids, referential sanity) at import time.
 *
 * The literal unions come from `./types`, so the schema and the TypeScript
 * type can never disagree about what a legal value is.
 */

import { z } from "zod";
import {
  APPLICABILITIES,
  ARCHITECTURE_STATUSES,
  ATTENTION_KINDS,
  GPU_CATEGORIES,
  LIFECYCLES,
  MODEL_CATEGORIES,
  PROVENANCE_SOURCES,
  QUANT_FAMILIES,
  QUANTIZATION_IDS,
} from "./types";

/**
 * Stable ids are lowercase, kebab-case, and may contain dots.
 *
 * Dots are allowed because ids are already published in URLs
 * (/models/smollm2-1.7b) — an id is permanent, so the rule has to accept
 * what already exists rather than force a breaking rename.
 */
const STABLE_ID = /^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$/;

/** ISO calendar date, e.g. "2026-10-05". */
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

export const ProvenanceSchema = z
  .object({
    source: z.enum(PROVENANCE_SOURCES),
    sourceUrl: z.string().url().optional(),
    retrievedAt: z.string().regex(ISO_DATE, "expected YYYY-MM-DD").optional(),
    notes: z.string().optional(),
    applicability: z.enum(APPLICABILITIES).optional(),
    limitations: z.array(z.string().min(1)).optional(),
  })
  .refine(
    (p) => p.source === "unspecified" || p.sourceUrl !== undefined,
    {
      message:
        "provenance.sourceUrl is required for every source except \"unspecified\" — a record must not claim a source it cannot point at",
    },
  )
  .refine(
    (p) => !p.limitations || p.limitations.length > 0,
    { message: "provenance.limitations must not be an empty array — omit it instead" },
  );

export const LayerAttentionSchema = z
  .object({
    kind: z.enum(ATTENTION_KINDS),
    windowTokens: z.number().int().positive().optional(),
    kvLoraRank: z.number().int().positive().optional(),
    mlaRopeHeadDim: z.number().int().positive().optional(),
    stateBytes: z.number().int().positive().optional(),
  })
  .superRefine((layer, ctx) => {
    switch (layer.kind) {
      case "sliding":
        if (layer.windowTokens === undefined) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["windowTokens"],
            message: "a sliding layer needs windowTokens — its cache cost depends on it",
          });
        }
        break;
      case "mla":
        if (layer.kvLoraRank === undefined) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["kvLoraRank"],
            message: "an mla layer needs kvLoraRank — the compressed latent size",
          });
        }
        if (layer.mlaRopeHeadDim === undefined) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["mlaRopeHeadDim"],
            message: "an mla layer needs mlaRopeHeadDim — the decoupled RoPE key size",
          });
        }
        break;
      case "linear":
      case "kda":
        if (layer.stateBytes === undefined) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["stateBytes"],
            message:
              "a linear/kda layer needs stateBytes — the fixed recurrent state it keeps instead of a KV cache",
          });
        }
        break;
      case "full":
        break;
    }
  });

export const ModelRecordSchema = z
  .object({
    id: z.string().regex(STABLE_ID, "ids are lowercase kebab-case and never change"),
    name: z.string().min(1),
    family: z.string().min(1),
    category: z.enum(MODEL_CATEGORIES),
    paramsB: z.number().positive(),
    activeParamsB: z.number().positive(),
    layers: z.number().int().positive(),
    hiddenDim: z.number().int().positive(),
    qHeads: z.number().int().positive(),
    kvHeads: z.number().int().positive(),
    headDim: z.number().int().positive(),
    vocabSize: z.number().int().positive(),
    maxContext: z.number().int().positive(),
    isMoE: z.boolean(),

    attentionLayers: z.array(LayerAttentionSchema).optional(),
    architectureStatus: z.enum(ARCHITECTURE_STATUSES).optional(),
    nativeWeightFormat: z.string().optional(),
    repository: z.string().optional(),
    revision: z.string().optional(),
    license: z.string().optional(),
    residentWeightBytes: z.number().int().positive().optional(),
    nativeContext: z.number().int().positive().optional(),
    contextExtensionMethod: z.string().optional(),
    runtimeContextLimit: z.number().int().positive().optional(),

    lifecycle: z.enum(LIFECYCLES),
    provenance: ProvenanceSchema,
  })
  .superRefine((m, ctx) => {
    if (m.activeParamsB > m.paramsB) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["activeParamsB"],
        message: "activeParamsB cannot exceed paramsB — every active parameter is also a resident one",
      });
    }
    if (m.isMoE !== m.activeParamsB < m.paramsB) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["isMoE"],
        message:
          "isMoE must agree with the parameter split: MoE iff activeParamsB < paramsB",
      });
    }
    if (m.attentionLayers && m.attentionLayers.length !== m.layers) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["attentionLayers"],
        message: `attentionLayers has ${m.attentionLayers.length} entries but the model has ${m.layers} layers — the layout must describe every layer`,
      });
    }
    if (m.attentionLayers && !m.architectureStatus) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["architectureStatus"],
        message:
          "a model with an explicit attention layout must state architectureStatus — if the layout is not fully modelled, say \"memory-only\" instead of leaving it implied",
      });
    }
    if (m.nativeContext !== undefined && m.nativeContext > m.maxContext) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["nativeContext"],
        message: "nativeContext cannot exceed maxContext",
      });
    }
    if (m.runtimeContextLimit !== undefined && m.runtimeContextLimit > m.maxContext) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["runtimeContextLimit"],
        message: "runtimeContextLimit cannot exceed maxContext",
      });
    }
  });

export const AcceleratorRecordSchema = z
  .object({
    id: z.string().regex(STABLE_ID, "ids are lowercase kebab-case and never change"),
    name: z.string().min(1),
    vendor: z.string().min(1),
    category: z.enum(GPU_CATEGORIES),
    memBandwidthGbps: z.number().positive(),
    flopsTflops: z.number().positive().nullable(),
    vramGb: z.number().positive(),
    nvlinkGbps: z.number().min(0),
    usdPerHour: z.number().min(0).nullable(),
    year: z.number().int().min(2010).max(2100),
    note: z.string().optional(),
    sourceUrl: z.string().url().optional(),
    lifecycle: z.enum(LIFECYCLES),
    provenance: ProvenanceSchema,
  })
  .superRefine((a, ctx) => {
    if (a.nvlinkGbps > 0 && a.category === "consumer") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["nvlinkGbps"],
        message:
          "consumer parts do not ship NVLink — if this is a workstation part, use the \"workstation\" category",
      });
    }
  });

export const QuantizationRecordSchema = z
  .object({
    id: z.enum(QUANTIZATION_IDS),
    label: z.string().min(1),
    bytesPerParam: z.number().positive().max(8),
    efficiency: z.number().positive(),
    description: z.string().min(1),
    family: z.enum(QUANT_FAMILIES),
    useCase: z.string().optional(),
    lifecycle: z.enum(LIFECYCLES),
    provenance: ProvenanceSchema,
  });

export const ModelsFileSchema = z.object({ records: z.array(ModelRecordSchema) });
export const AcceleratorsFileSchema = z.object({
  records: z.array(AcceleratorRecordSchema),
});
export const QuantizationsFileSchema = z.object({
  records: z.array(QuantizationRecordSchema),
});

export const CatalogManifestSchema = z.object({
  catalogVersion: z.string().regex(/^\d+\.\d+\.\d+$/),
  description: z.string().min(1),
  updatedAt: z.string().regex(ISO_DATE, "expected YYYY-MM-DD"),
  files: z.object({
    models: z.string(),
    accelerators: z.string(),
    quantizations: z.string(),
  }),
  counts: z.object({
    models: z.number().int().nonnegative(),
    accelerators: z.number().int().nonnegative(),
    quantizations: z.number().int().nonnegative(),
  }),
});

export type ModelRecordInput = z.input<typeof ModelRecordSchema>;
export type AcceleratorRecordInput = z.input<typeof AcceleratorRecordSchema>;
export type QuantizationRecordInput = z.input<typeof QuantizationRecordSchema>;
