/**
 * Per-install local storage for MCP state that must survive between calls:
 *   - ~/.tokcalc/model-specs/   → fetch_model_spec cache (24h TTL)
 *   - ~/.tokcalc/calibration.json → record_measured measurements
 *
 * Design notes:
 *  - The npm stdio server runs on the user's machine, so per-install storage
 *    is private and persistent. The hosted HTTP endpoint runs on Vercel where
 *    the filesystem is read-only (except /tmp) and shared across tenants —
 *    there we degrade gracefully: writes fall back to a temp dir (ephemeral)
 *    and every response discloses where data actually went.
 *  - All helpers are best-effort: they never throw into a tool handler; they
 *    return null/[] on failure so tools keep working without persistence.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const MAX_CALIBRATION_RECORDS = 500;

function storeRoot(): string | null {
  try {
    const base = process.env.TOKCALC_STORE_DIR || os.homedir() || "/tmp";
    const dir = path.join(base, ".tokcalc");
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  } catch {
    return null;
  }
}

/** True when the store is genuinely persistent (user home), not a temp fallback. */
export function storeIsPersistent(): boolean {
  try {
    const base = process.env.TOKCALC_STORE_DIR || os.homedir();
    return base !== "/tmp" && base !== undefined;
  } catch {
    return false;
  }
}

// ---------- model spec cache ----------

export interface CachedModelSpec {
  modelId: string;
  hfRepo: string;
  fetchedAt: string;
  config: Record<string, unknown>;
}

export function readSpecCache(modelId: string): CachedModelSpec | null {
  const root = storeRoot();
  if (!root) return null;
  try {
    const file = path.join(root, "model-specs", `${modelId.replace(/[^a-zA-Z0-9._-]/g, "_")}.json`);
    if (!fs.existsSync(file)) return null;
    const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as CachedModelSpec;
    const ageMs = Date.now() - new Date(parsed.fetchedAt).getTime();
    if (!Number.isFinite(ageMs) || ageMs > 24 * 3600 * 1000) return null; // 24h TTL
    return parsed;
  } catch {
    return null;
  }
}

export function writeSpecCache(entry: CachedModelSpec): boolean {
  const root = storeRoot();
  if (!root) return false;
  try {
    const dir = path.join(root, "model-specs");
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `${entry.modelId.replace(/[^a-zA-Z0-9._-]/g, "_")}.json`);
    fs.writeFileSync(file, JSON.stringify(entry, null, 2));
    return true;
  } catch {
    return false;
  }
}

// ---------- calibration records ----------

export interface CalibrationRecord {
  model: string;
  gpu: string;
  quantization: string;
  gpuCount: number;
  observed: {
    decodeTps?: number;
    aggregateTps?: number;
    ttftMs?: number;
    itlMs?: number;
  };
  ratios: {
    decodeTps?: number;
    aggregateTps?: number;
    ttftMs?: number;
    itlMs?: number;
  };
  source?: string;
  notes?: string;
  recordedAt: string;
}

function calibrationFile(): string | null {
  const root = storeRoot();
  if (!root) return null;
  return path.join(root, "calibration.json");
}

export function readCalibration(): CalibrationRecord[] {
  const file = calibrationFile();
  if (!file) return [];
  try {
    if (!fs.existsSync(file)) return [];
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    return Array.isArray(parsed) ? (parsed as CalibrationRecord[]) : [];
  } catch {
    return [];
  }
}

export function writeCalibrationRecord(record: CalibrationRecord): { ok: boolean; path: string | null } {
  const file = calibrationFile();
  if (!file) return { ok: false, path: null };
  try {
    const all = readCalibration();
    all.push(record);
    while (all.length > MAX_CALIBRATION_RECORDS) all.shift();
    fs.writeFileSync(file, JSON.stringify(all, null, 2));
    return { ok: true, path: file };
  } catch {
    return { ok: false, path: file };
  }
}

/** Most-recent-first records matching a model/gpu/quant cell. */
export function latestCalibrationFor(
  model: string,
  gpu: string,
  quantization: string,
): CalibrationRecord[] {
  return readCalibration()
    .filter(
      (r) =>
        r.model === model &&
        r.gpu === gpu &&
        r.quantization === quantization,
    )
    .sort((a, b) => (a.recordedAt < b.recordedAt ? 1 : -1));
}
