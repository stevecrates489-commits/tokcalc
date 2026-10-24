/**
 * Public surface of the catalog module.
 *
 * Import from here rather than reaching into `loader.ts` / `types.ts`
 * directly, so the internal layout can change without touching call sites.
 */

export {
  CATALOG_MANIFEST,
  CATALOG_VERSION,
  MODELS,
  MODEL_MAP,
  ACCELERATORS,
  ACCELERATOR_MAP,
  QUANTIZATIONS,
  QUANT_MAP,
  getModel,
  getAccelerator,
  getQuantization,
  activeModels,
  activeAccelerators,
  modelFamilies,
  acceleratorVendors,
} from "./loader";

export type {
  AcceleratorRecord,
  ArchitectureStatus,
  AttentionKind,
  GpuCategory,
  GpuSpec,
  LayerAttention,
  Lifecycle,
  ModelCategory,
  ModelRecord,
  ModelSpec,
  Provenance,
  ProvenanceSource,
  QuantFamily,
  QuantMeta,
  Quantization,
  QuantizationRecord,
} from "./types";
