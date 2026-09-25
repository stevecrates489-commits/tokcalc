/**
 * KV-backed rate limiting middleware for tokcalc MCP HTTP server.
 *
 * Per v0.2.0 research (Perplexity brief, 2025-09-25):
 *   "For a free public service, start conservatively:
 *      - unauthenticated IP: 10 MCP requests/minute
 *      - authenticated API key: 60 requests/minute
 *      - burst allowance: 5 requests
 *      - daily per-key budget: 1,000-5,000 calls
 *      - maximum concurrent tool calls per key: 2-5
 *      - maximum request body: 64-256 KB
 *      - maximum tool execution time: 10-15 seconds
 *    For a calculator-style server, 30 requests/minute per IP and 120 requests/minute
 *    per API key would also be reasonable if the work is cheap."
 *
 * Implementation:
 *   - Uses @upstash/ratelimit + @upstash/redis (works on Vercel, Cloudflare, Node)
 *   - Sliding window algorithm (more accurate than fixed window)
 *   - Two buckets per request:
 *       1. Per-IP (always) — 30 req/min
 *       2. Per-API-key (when authenticated) — 120 req/min
 *   - Graceful degradation: if UPSTASH_REDIS_REST_URL is not set, rate limiting
 *     is DISABLED (useful for local dev — DO NOT deploy to prod without it)
 *
 * Env vars (set on Vercel: Settings → Environment Variables):
 *   UPSTASH_REDIS_REST_URL       — your Upstash Redis REST URL
 *   UPSTASH_REDIS_REST_TOKEN     — your Upstash Redis REST token
 *
 * Optional overrides (with sensible defaults):
 *   MCP_RATE_LIMIT_ANON_IP_PER_MIN   (default: 30)  — limit for unauthenticated IP
 *   MCP_RATE_LIMIT_AUTHED_KEY_PER_MIN (default: 120) — limit for authenticated API key
 *
 * Get free Upstash Redis: https://upstash.com/ (10K commands/month free tier)
 */

import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import type { ServerResponse } from "node:http";

// === Singleton Redis client (lazy-initialized on first use) ===
let redisClient: Redis | null = null;
let anonIpLimiter: Ratelimit | null = null;
let authedKeyLimiter: Ratelimit | null = null;

const ANON_IP_LIMIT = Number(process.env.MCP_RATE_LIMIT_ANON_IP_PER_MIN ?? 30);
const AUTHED_KEY_LIMIT = Number(process.env.MCP_RATE_LIMIT_AUTHED_KEY_PER_MIN ?? 120);

/**
 * Initialize the Redis client + rate limiters.
 * Returns null if UPSTASH_REDIS_REST_URL is not set (rate limiting disabled).
 */
function getLimiters(): { anonIp: Ratelimit; authedKey: Ratelimit } | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;

  if (!url || !token) {
    return null; // Rate limiting disabled — local dev mode
  }

  if (!redisClient) {
    redisClient = new Redis({ url, token });
  }

  if (!anonIpLimiter) {
    anonIpLimiter = new Ratelimit({
      redis: redisClient,
      limiter: Ratelimit.slidingWindow(ANON_IP_LIMIT, "1 m"),
      analytics: true,
      prefix: "tokcalc-mcp:anon-ip",
      ephemeralCache: undefined, // we're a stateless server, no in-process cache
    });
  }

  if (!authedKeyLimiter) {
    authedKeyLimiter = new Ratelimit({
      redis: redisClient,
      limiter: Ratelimit.slidingWindow(AUTHED_KEY_LIMIT, "1 m"),
      analytics: true,
      prefix: "tokcalc-mcp:authed-key",
      ephemeralCache: undefined,
    });
  }

  return { anonIp: anonIpLimiter, authedKey: authedKeyLimiter };
}

export interface RateLimitResult {
  /** Whether the request should be allowed through */
  success: boolean;
  /** The limit that was applied (per minute) */
  limit: number;
  /** Remaining requests in the current window */
  remaining: number;
  /** Unix timestamp (seconds) when the limit resets */
  reset: number;
  /** Which bucket was checked — useful for the 429 response */
  bucket: "anon-ip" | "authed-key" | "none";
}

/**
 * Check the rate limit for a request.
 *
 * Behavior:
 *   - If KV is not configured (UPSTASH_REDIS_REST_URL unset) → always success (no limit)
 *   - If authenticated → check both per-IP AND per-key buckets, return the stricter result
 *   - If anonymous → check only per-IP bucket
 *
 * @param rateLimitId  The identifier from auth.ts (e.g., "key:abc123" or "anon-ip:1.2.3.4")
 * @param authenticated  Whether the request was authenticated
 */
export async function checkRateLimit(
  rateLimitId: string,
  authenticated: boolean,
): Promise<RateLimitResult> {
  const limiters = getLimiters();

  // No KV configured — rate limiting disabled
  if (!limiters) {
    return {
      success: true,
      limit: Infinity,
      remaining: Infinity,
      reset: 0,
      bucket: "none",
    };
  }

  // Always check the per-IP bucket (anonymous AND authenticated requests both count)
  const ipResult = await limiters.anonIp.limit(rateLimitId);

  // If authenticated, ALSO check the per-key bucket (looser limit)
  if (authenticated && rateLimitId.startsWith("key:")) {
    const keyResult = await limiters.authedKey.limit(rateLimitId);

    // Return the stricter of the two (the one that's closer to being exhausted)
    if (!keyResult.success) {
      return {
        success: false,
        limit: AUTHED_KEY_LIMIT,
        remaining: keyResult.remaining,
        reset: keyResult.reset,
        bucket: "authed-key",
      };
    }
  }

  return {
    success: ipResult.success,
    limit: ANON_IP_LIMIT,
    remaining: ipResult.remaining,
    reset: ipResult.reset,
    bucket: "anon-ip",
  };
}

/**
 * Send a proper 429 Too Many Requests response with RateLimit-* headers.
 * Per draft IETF RateLimit header spec + research recommendations.
 */
export function sendTooManyRequests(
  res: ServerResponse,
  result: RateLimitResult,
): void {
  const retryAfterSec = Math.max(1, Math.ceil((result.reset * 1000 - Date.now()) / 1000));

  res.writeHead(429, {
    "Content-Type": "application/json",
    "Retry-After": String(retryAfterSec),
    "RateLimit-Limit": String(result.limit === Infinity ? 0 : result.limit),
    "RateLimit-Remaining": String(result.remaining === Infinity ? 0 : result.remaining),
    "RateLimit-Reset": String(result.reset),
    "X-RateLimit-Bucket": result.bucket,
  });

  res.end(JSON.stringify({
    jsonrpc: "2.0",
    error: {
      code: -32002,
      message: "Too Many Requests",
      data: {
        detail: `Rate limit exceeded on bucket "${result.bucket}". Try again in ${retryAfterSec} second${retryAfterSec === 1 ? "" : "s"}.`,
        retryAfter: retryAfterSec,
        limit: result.limit === Infinity ? "unlimited" : result.limit,
        reset: result.reset,
      },
    },
    id: null,
  }));
}

/**
 * Is rate limiting enabled? (i.e., is KV configured?)
 * Useful for startup banners + logging.
 */
export function isRateLimitEnabled(): boolean {
  return !!process.env.UPSTASH_REDIS_REST_URL && !!process.env.UPSTASH_REDIS_REST_TOKEN;
}
