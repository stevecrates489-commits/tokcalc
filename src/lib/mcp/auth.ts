/**
 * API key validation utility for tokcalc MCP HTTP server.
 *
 * Used by:
 *   - src/app/api/mcp/route.ts (public hosted endpoint)
 *   - mini-services/mcp-server/http.ts (standalone self-hosted HTTP server)
 *
 * v0.2.0 design:
 *   1. If UPSTASH_REDIS_REST_URL is set, check Redis for user-generated keys
 *      (created via /api/keys endpoint — self-serve key request form).
 *   2. Fall back to MCP_API_KEY env var (master/admin key — set on Vercel).
 *   3. If neither matches, return invalid.
 *
 * Key format: tokcalc_live_<32-char-hex> (e.g., tokcalc_live_9f82a17b3c4e5d6a7b8c9d0e1f2a3b4c)
 * Storage: SHA-256 hash of the key is stored in Redis (never the raw key).
 *
 * Roadmap:
 *   v0.2.0 (this file): static env var master key + Redis-stored user keys
 *   v0.3.0:              OAuth 2.1 with PKCE for multi-user auth
 */

import crypto from "node:crypto";
import { Redis } from "@upstash/redis";

// ============================================================
// CONSTANTS
// ============================================================

const KEY_PREFIX = "tokcalc_live_";
const KEY_TTL_SECONDS = 90 * 24 * 60 * 60; // 90 days

// ============================================================
// SINGLETON REDIS CLIENT
// ============================================================

let redisClient: Redis | null = null;

function getRedis(): Redis | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  if (!redisClient) {
    redisClient = new Redis({ url, token });
  }
  return redisClient;
}

// ============================================================
// HELPERS
// ============================================================

/**
 * Constant-time string comparison to prevent timing attacks.
 */
function safeEqual(a: string, b: string): boolean {
  const aa = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  if (aa.length !== bb.length) return false;
  return crypto.timingSafeEqual(aa, bb);
}

/**
 * SHA-256 hash of an API key (for safe storage + lookup in Redis).
 * Never log/store the raw key — only its hash.
 */
export function hashKey(key: string): string {
  return crypto.createHash("sha256").update(key).digest("hex");
}

/**
 * Generate a new random API key.
 * Format: tokcalc_live_<32-char-hex>
 * Uses crypto.randomBytes for cryptographic randomness.
 */
export function generateApiKey(): string {
  const random = crypto.randomBytes(16).toString("hex");
  return `${KEY_PREFIX}${random}`;
}

/**
 * Validate the format of an API key (prefix + hex suffix).
 * Doesn't check if it's actually valid — just format.
 */
export function isValidKeyFormat(key: string): boolean {
  return /^tokcalc_live_[a-f0-9]{32}$/i.test(key);
}

/**
 * Extract the bearer token from an Authorization header value.
 * Returns null if missing or malformed.
 */
export function extractBearer(authHeader: string | null): string | null {
  if (!authHeader) return null;
  const match = /^Bearer\s+(.+)$/i.exec(authHeader.trim());
  return match ? match[1].trim() : null;
}

// ============================================================
// VALIDATION RESULT
// ============================================================

export interface ApiKeyValidation {
  valid: boolean;
  /** Email associated with the key (only for Redis-stored user keys) */
  email?: string;
  /** Where the key was validated: 'redis' (user key), 'env' (master key), or 'invalid' */
  source: "redis" | "env" | "invalid";
  /** Identifier for rate-limit bucketing (key hash if valid, IP-based if invalid) */
  rateLimitId: string;
}

// ============================================================
// MAIN VALIDATION FUNCTION
// ============================================================

/**
 * Validate an API key against Redis (user keys) and env var (master key).
 *
 * Behavior:
 *   1. If no Authorization header / no key provided → invalid (caller sends 401)
 *   2. If Redis is configured:
 *      - Hash the provided key
 *      - Look up `tokcalc:apikey:<hash>` in Redis
 *      - If found → valid (source: 'redis', return stored email)
 *   3. If MCP_API_KEY env var is set:
 *      - Constant-time compare against env var
 *      - If match → valid (source: 'env')
 *   4. Else → invalid
 *
 * @param authHeader The raw Authorization header value (e.g., "Bearer tokcalc_live_...")
 * @param clientIp The client IP (for rate-limit bucketing when invalid)
 */
export async function validateApiKey(
  authHeader: string | null,
  clientIp: string = "unknown",
): Promise<ApiKeyValidation> {
  const providedKey = extractBearer(authHeader);

  // No key provided
  if (!providedKey) {
    return {
      valid: false,
      source: "invalid",
      rateLimitId: `anon-ip:${clientIp}`,
    };
  }

  // Bad format → don't even bother checking Redis
  if (!isValidKeyFormat(providedKey)) {
    return {
      valid: false,
      source: "invalid",
      rateLimitId: `bad-key:${hashKey(providedKey).slice(0, 16)}`,
    };
  }

  // Step 1: Check Redis for user-generated key
  const redis = getRedis();
  if (redis) {
    try {
      const keyHash = hashKey(providedKey);
      const redisKey = `tokcalc:apikey:${keyHash}`;
      const stored = await redis.get<{ email: string; createdAt: string }>(redisKey);
      if (stored) {
        return {
          valid: true,
          email: stored.email,
          source: "redis",
          rateLimitId: `key:${keyHash.slice(0, 16)}`,
        };
      }
    } catch (err) {
      // Redis error — fall through to env var check
      console.error("[tokcalc/auth] Redis lookup error:", err);
    }
  }

  // Step 2: Check MCP_API_KEY env var (master/admin key)
  const masterKey = process.env.MCP_API_KEY;
  if (masterKey && safeEqual(providedKey, masterKey)) {
    return {
      valid: true,
      source: "env",
      rateLimitId: `key:${hashKey(providedKey).slice(0, 16)}`,
    };
  }

  // Step 3: Invalid
  return {
    valid: false,
    source: "invalid",
    rateLimitId: `bad-key:${hashKey(providedKey).slice(0, 16)}`,
  };
}

/**
 * Store a newly-generated API key in Redis.
 * Called by /api/keys POST endpoint.
 *
 * Stores two Redis entries:
 *   1. `tokcalc:apikey:<hash>` = {email, createdAt, key} (for validation lookup)
 *   2. `tokcalc:email:<email>` = <hash> (for "get my key" by email)
 *
 * Both entries have 90-day TTL (auto-expire unused keys).
 */
export async function storeApiKey(key: string, email: string): Promise<void> {
  const redis = getRedis();
  if (!redis) {
    throw new Error("UPSTASH_REDIS_REST_URL is not set — cannot store API key");
  }

  const keyHash = hashKey(key);
  const apiKey = `tokcalc:apikey:${keyHash}`;
  const emailKey = `tokcalc:email:${email.toLowerCase()}`;

  // Store key → {email, createdAt}
  await redis.set(apiKey, { email, createdAt: new Date().toISOString(), key }, { ex: KEY_TTL_SECONDS });
  // Store email → hash (for "get existing key by email" lookup)
  await redis.set(emailKey, keyHash, { ex: KEY_TTL_SECONDS });
}

/**
 * Look up an existing API key by email.
 * If the email already has a key, return it (so we don't create duplicates).
 * Returns null if no key exists or Redis isn't configured.
 */
export async function getApiKeyByEmail(email: string): Promise<string | null> {
  const redis = getRedis();
  if (!redis) return null;

  const emailKey = `tokcalc:email:${email.toLowerCase()}`;
  const keyHash = await redis.get<string>(emailKey);
  if (!keyHash) return null;

  const apiKey = `tokcalc:apikey:${keyHash}`;
  const stored = await redis.get<{ email: string; createdAt: string; key: string }>(apiKey);
  return stored?.key || null;
}

/**
 * Is auth enabled at all?
 * (Either MCP_API_KEY env var is set, OR Redis is configured for key storage)
 */
export function isAuthEnabled(): boolean {
  return !!process.env.MCP_API_KEY || !!process.env.UPSTASH_REDIS_REST_URL;
}

/**
 * Is Redis-backed key storage configured?
 */
export function isRedisConfigured(): boolean {
  return !!process.env.UPSTASH_REDIS_REST_URL && !!process.env.UPSTASH_REDIS_REST_TOKEN;
}
