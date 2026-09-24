/**
 * URL-hash state serialization for tokcalc.
 *
 * Format: `#t=<tab>&<key>=<value>&...`
 * - Compact keys to keep URLs short
 * - Empty/optional values omitted entirely (not encoded as `&=`
 * - Booleans encoded as `1`/`0` (omitted when false)
 * - All numbers as-is (NaN → omitted)
 *
 * Two state namespaces:
 * - Calculator tab:  m/g/q/n/b/p/o/c/s/k/cb/cbm/r/cp/ch/ct/cpr
 * - Build vs Buy tab: sm/sg/sq/sn/sp/u/b/i/o/r/ap/am
 *
 * The `t` key (tab) determines which namespace is active.
 */

export type Tab = "calculator" | "build-vs-buy" | "reference";

/** Calculator tab state — shape mirrors the useState vars in page.tsx */
export interface CalcTabState {
  modelId: string;
  gpuId: string;
  quantization: string;
  numGpus: number;
  batchSize: number;
  promptTokens: number;
  outputTokens: number;
  gpuHourlyCost: number | "";
  useSpeculative: boolean;
  speculativeBoost: number;
  useContinuousBatching: boolean;
  continuousBatchingMultiplier: number;
  reasoningTokens: number;
  cachePrefixTokens: number;
  cacheHitRate: number;
  cacheTTL: string;
  cacheProvider: string;
}

/** Build-vs-Buy tab state — shape mirrors BuildVsBuyTab useState vars */
export interface BvbTabState {
  shModel: string;
  shGpu: string;
  shQuant: string;
  shNumGpus: number;
  shGpuPrice: number | "";
  utilization: number;
  batchSize: number;
  inputTokens: number;
  outputTokens: number;
  reqsPerDay: number;
  apiProvider: string;
  apiModel: string;
}

/** Parse the URL hash into a structured state object. */
export function parseUrlHash(hash: string): {
  tab: Tab;
  calc?: Partial<CalcTabState>;
  bvb?: Partial<BvbTabState>;
} {
  // Strip leading `#`
  const stripped = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!stripped) return { tab: "calculator" };

  const params = new URLSearchParams(stripped);
  const tabParam = params.get("t");
  const tab: Tab =
    tabParam === "build-vs-buy" ? "build-vs-buy" :
    tabParam === "reference" ? "reference" :
    "calculator";

  if (tab === "calculator") {
    return {
      tab,
      calc: {
        modelId: params.get("m") ?? undefined,
        gpuId: params.get("g") ?? undefined,
        quantization: params.get("q") ?? undefined,
        numGpus: numOrUndef(params.get("n")),
        batchSize: numOrUndef(params.get("b")),
        promptTokens: numOrUndef(params.get("p")),
        outputTokens: numOrUndef(params.get("o")),
        gpuHourlyCost: numOrUndefOrEmpty(params.get("c")),
        useSpeculative: boolOrUndef(params.get("s")),
        speculativeBoost: numOrUndef(params.get("k")),
        useContinuousBatching: boolOrUndef(params.get("cb")),
        continuousBatchingMultiplier: numOrUndef(params.get("cbm")),
        reasoningTokens: numOrUndef(params.get("r")),
        cachePrefixTokens: numOrUndef(params.get("cp")),
        cacheHitRate: numOrUndef(params.get("ch")),
        cacheTTL: params.get("ct") ?? undefined,
        cacheProvider: params.get("cpr") ?? undefined,
      } as Partial<CalcTabState>,
    };
  }

  if (tab === "build-vs-buy") {
    return {
      tab,
      bvb: {
        shModel: params.get("sm") ?? undefined,
        shGpu: params.get("sg") ?? undefined,
        shQuant: params.get("sq") ?? undefined,
        shNumGpus: numOrUndef(params.get("sn")),
        shGpuPrice: numOrUndefOrEmpty(params.get("sp")),
        utilization: numOrUndef(params.get("u")),
        batchSize: numOrUndef(params.get("b")),
        inputTokens: numOrUndef(params.get("i")),
        outputTokens: numOrUndef(params.get("o")),
        reqsPerDay: numOrUndef(params.get("r")),
        apiProvider: params.get("ap") ?? undefined,
        apiModel: params.get("am") ?? undefined,
      } as Partial<BvbTabState>,
    };
  }

  return { tab };
}

/** Serialize Calculator tab state to a URL hash (without leading `#`). */
export function serializeCalcState(s: CalcTabState): string {
  const p = new URLSearchParams();
  p.set("t", "calculator");
  if (s.modelId) p.set("m", s.modelId);
  if (s.gpuId) p.set("g", s.gpuId);
  if (s.quantization) p.set("q", s.quantization);
  if (s.numGpus !== 1) p.set("n", String(s.numGpus));
  if (s.batchSize !== 1) p.set("b", String(s.batchSize));
  if (s.promptTokens !== 500) p.set("p", String(s.promptTokens));
  if (s.outputTokens !== 200) p.set("o", String(s.outputTokens));
  if (s.gpuHourlyCost !== "") p.set("c", String(s.gpuHourlyCost));
  if (s.useSpeculative) { p.set("s", "1"); p.set("k", String(s.speculativeBoost)); }
  if (s.useContinuousBatching) { p.set("cb", "1"); p.set("cbm", String(s.continuousBatchingMultiplier)); }
  if (s.reasoningTokens > 0) p.set("r", String(s.reasoningTokens));
  if (s.cachePrefixTokens > 0) {
    p.set("cp", String(s.cachePrefixTokens));
    p.set("ch", String(s.cacheHitRate));
    if (s.cacheTTL !== "5m") p.set("ct", s.cacheTTL);
    if (s.cacheProvider !== "self-hosted") p.set("cpr", s.cacheProvider);
  }
  return p.toString();
}

/** Serialize Build-vs-Buy tab state to a URL hash (without leading `#`). */
export function serializeBvbState(s: BvbTabState): string {
  const p = new URLSearchParams();
  p.set("t", "build-vs-buy");
  if (s.shModel) p.set("sm", s.shModel);
  if (s.shGpu) p.set("sg", s.shGpu);
  if (s.shQuant) p.set("sq", s.shQuant);
  if (s.shNumGpus !== 1) p.set("sn", String(s.shNumGpus));
  if (s.shGpuPrice !== "") p.set("sp", String(s.shGpuPrice));
  if (s.utilization !== 50) p.set("u", String(s.utilization));
  if (s.batchSize !== 8) p.set("b", String(s.batchSize));
  if (s.inputTokens !== 500) p.set("i", String(s.inputTokens));
  if (s.outputTokens !== 200) p.set("o", String(s.outputTokens));
  if (s.reqsPerDay !== 1000) p.set("r", String(s.reqsPerDay));
  if (s.apiProvider !== "openai") p.set("ap", s.apiProvider);
  if (s.apiModel !== "gpt-4o-mini") p.set("am", s.apiModel);
  return p.toString();
}

/** Write a hash to the URL without adding a history entry. */
export function writeUrlHash(hash: string): void {
  if (typeof window === "undefined") return;
  const newUrl = `${window.location.pathname}${window.location.search}#${hash}`;
  window.history.replaceState(null, "", newUrl);
}

/** Copy current URL to clipboard. Returns true on success. */
export async function copyCurrentUrlToClipboard(): Promise<boolean> {
  if (typeof navigator === "undefined" || !navigator.clipboard) return false;
  try {
    await navigator.clipboard.writeText(window.location.href);
    return true;
  } catch {
    return false;
  }
}

/* ============================================================
   LOCALSTORAGE PERSISTENCE (Phase A bonus)
   ============================================================
   On unmount (page unload), write the current state to localStorage.
   On mount, if no URL hash exists, restore from localStorage.
   URL hash always wins — explicit shares override persistence.
*/

const STORAGE_KEY_CALC = "tokcalc:calc-state:v1";
const STORAGE_KEY_BVB = "tokcalc:bvb-state:v1";

/** Save Calculator tab state to localStorage. Silently fails on error. */
export function saveCalcToStorage(s: CalcTabState): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY_CALC, JSON.stringify(s));
  } catch {
    // Quota exceeded or disabled — non-fatal
  }
}

/** Save Build-vs-Buy tab state to localStorage. Silently fails on error. */
export function saveBvbToStorage(s: BvbTabState): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY_BVB, JSON.stringify(s));
  } catch {
    // Quota exceeded or disabled — non-fatal
  }
}

/** Read saved Calculator state from localStorage (or null if none). */
export function loadCalcFromStorage(): Partial<CalcTabState> | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY_CALC);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/** Read saved Build-vs-Buy state from localStorage (or null if none). */
export function loadBvbFromStorage(): Partial<BvbTabState> | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY_BVB);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/* ---------- internal helpers ---------- */

function numOrUndef(v: string | null): number | undefined {
  if (v === null || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function numOrUndefOrEmpty(v: string | null): number | "" | undefined {
  if (v === null) return undefined;
  if (v === "") return "";
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function boolOrUndef(v: string | null): boolean | undefined {
  if (v === null) return undefined;
  return v === "1" || v === "true";
}
