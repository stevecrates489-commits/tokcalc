/**
 * Self-serve API key generation endpoint.
 *
 * POST /api/keys  { email: "user@example.com" }
 *   → { key: "tokcalc_live_...", expiresAt, usage: {curl example} }
 *
 * Flow:
 *   1. Validate email format (simple regex — no SMTP verification)
 *   2. Rate limit: 5 key requests per IP per day (prevents abuse)
 *   3. Check if email already has a key in Redis → return existing key
 *   4. Generate new key: tokcalc_live_<32-char-hex>
 *   5. Store hash + email in Redis (90-day TTL)
 *   6. Return key + curl usage example
 *
 * No actual email verification (would need SMTP — deferred to v0.3.0).
 * For beta, we trust the email format + rate limit per IP.
 *
 * Required env vars:
 *   UPSTASH_REDIS_REST_URL       — Upstash Redis REST URL
 *   UPSTASH_REDIS_REST_TOKEN     — Upstash Redis REST token
 *
 * If Redis isn't configured, returns 503 Service Unavailable.
 */

import { NextRequest } from "next/server";
import {
  generateApiKey,
  storeApiKey,
  getApiKeyByEmail,
  isRedisConfigured,
} from "@/lib/mcp/auth";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

// Force Node.js runtime (we need crypto + Upstash client)
export const runtime = "nodejs";
// Never cache key generation responses
export const dynamic = "force-dynamic";
// Quick response — no heavy compute
export const maxDuration = 10;

// ============================================================
// RATE LIMITING (separate from MCP tool call rate limit)
// ============================================================

let redisClient: Redis | null = null;
let keyGenLimiter: Ratelimit | null = null;

const KEY_GEN_LIMIT_PER_DAY = 5; // 5 key requests per IP per day

function getKeyGenLimiter(): Ratelimit | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;

  if (!redisClient) {
    redisClient = new Redis({ url, token });
  }
  if (!keyGenLimiter) {
    keyGenLimiter = new Ratelimit({
      redis: redisClient,
      limiter: Ratelimit.slidingWindow(KEY_GEN_LIMIT_PER_DAY, "1 d"),
      prefix: "tokcalc:keys:ratelimit",
    });
  }
  return keyGenLimiter;
}

// ============================================================
// HELPERS
// ============================================================

const EMAIL_REGEX = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;

function getClientIp(req: NextRequest): string {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  const cfIp = req.headers.get("cf-connecting-ip");
  if (cfIp) return cfIp.trim();
  return "unknown";
}

function jsonError(statusCode: number, error: string, detail?: string): Response {
  return new Response(
    JSON.stringify({ error, detail: detail ?? null }),
    { status: statusCode, headers: { "Content-Type": "application/json" } },
  );
}

// ============================================================
// POST HANDLER
// ============================================================

export async function POST(req: NextRequest): Promise<Response> {
  // Pre-flight: Redis must be configured for key storage
  if (!isRedisConfigured()) {
    return jsonError(503,
      "Key generation unavailable",
      "Server is not configured for key storage (UPSTASH_REDIS_REST_URL not set). Email hello@tokcalc.vercel.app for an API key.",
    );
  }

  // ── RATE LIMIT ──
  // 5 key requests per IP per day (prevents abuse / spam)
  const limiter = getKeyGenLimiter();
  if (limiter) {
    const clientIp = getClientIp(req);
    const rateLimitId = `ip:${clientIp}`;
    const result = await limiter.limit(rateLimitId);
    if (!result.success) {
      const retryAfterSec = Math.max(1, Math.ceil((result.reset * 1000 - Date.now()) / 1000));
      return new Response(
        JSON.stringify({
          error: "Too Many Requests",
          detail: `You've requested ${KEY_GEN_LIMIT_PER_DAY} API keys today (limit per IP). Try again in ${Math.ceil(retryAfterSec / 3600)} hour${Math.ceil(retryAfterSec / 3600) === 1 ? "" : "s"}.`,
        }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "Retry-After": String(retryAfterSec),
            "RateLimit-Limit": String(KEY_GEN_LIMIT_PER_DAY),
            "RateLimit-Remaining": String(result.remaining),
            "RateLimit-Reset": String(result.reset),
          },
        },
      );
    }
  }

  // ── PARSE BODY ──
  let body: { email?: string };
  try {
    const text = await req.text();
    body = JSON.parse(text);
  } catch {
    return jsonError(400, "Invalid JSON body", "Body must be a JSON object with an 'email' field.");
  }

  const email = body.email?.trim().toLowerCase();

  // ── VALIDATE EMAIL ──
  if (!email) {
    return jsonError(400, "Missing email", "Body must include an 'email' field.");
  }
  if (!EMAIL_REGEX.test(email)) {
    return jsonError(400, "Invalid email format", `"${email}" is not a valid email address.`);
  }
  if (email.length > 254) {
    return jsonError(400, "Email too long", "Email must be 254 characters or less.");
  }

  // ── CHECK FOR EXISTING KEY ──
  // If the email already has a key, return it (so user doesn't accumulate duplicate keys)
  try {
    const existingKey = await getApiKeyByEmail(email);
    if (existingKey) {
      return Response.json({
        key: existingKey,
        alreadyExisted: true,
        message: "You already have an API key. Reusing the existing one.",
        expiresAt: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString(),
        usage: {
          curl: `curl -X POST https://tokcalc.vercel.app/api/mcp \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json, text/event-stream" \\
  -H "MCP-Protocol-Version: 2025-03-26" \\
  -H "Authorization: Bearer ${existingKey}" \\
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'`,
          cursor_config: `{
  "mcpServers": {
    "tokcalc": {
      "url": "https://tokcalc.vercel.app/api/mcp",
      "headers": { "Authorization": "Bearer ${existingKey}" }
    }
  }
}`,
        },
      });
    }
  } catch (err) {
    console.error("[tokcalc/keys] Error checking existing key:", err);
    // Non-fatal — proceed to generate a new key
  }

  // ── GENERATE NEW KEY ──
  const newKey = generateApiKey();

  // ── STORE IN REDIS ──
  try {
    await storeApiKey(newKey, email);
  } catch (err) {
    console.error("[tokcalc/keys] Error storing new key:", err);
    return jsonError(500, "Failed to store API key", "Internal server error. Please try again.");
  }

  // ── RETURN SUCCESS ──
  const expiresAt = new Date(Date.now() + 90 * 24 * 60 * 60 * 1000);
  return Response.json({
    key: newKey,
    alreadyExisted: false,
    message: "API key generated. Store it securely — you won't see this again.",
    expiresAt: expiresAt.toISOString(),
    usage: {
      curl: `curl -X POST https://tokcalc.vercel.app/api/mcp \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json, text/event-stream" \\
  -H "MCP-Protocol-Version: 2025-03-26" \\
  -H "Authorization: Bearer ${newKey}" \\
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'`,
      cursor_config: `{
  "mcpServers": {
    "tokcalc": {
      "url": "https://tokcalc.vercel.app/api/mcp",
      "headers": { "Authorization": "Bearer ${newKey}" }
    }
  }
}`,
    },
  });
}

// ============================================================
// GET / DELETE — not supported (don't allow listing/deleting keys via API)
// ============================================================

export async function GET(): Promise<Response> {
  return new Response(
    JSON.stringify({
      error: "Method Not Allowed",
      detail: "Use POST to request a key. GET is not supported (keys cannot be listed).",
    }),
    { status: 405, headers: { "Content-Type": "application/json", "Allow": "POST" } },
  );
}

export async function DELETE(): Promise<Response> {
  return new Response(
    JSON.stringify({
      error: "Method Not Allowed",
      detail: "Key revocation is not yet implemented. Email hello@tokcalc.vercel.app to revoke a key.",
    }),
    { status: 405, headers: { "Content-Type": "application/json", "Allow": "POST" } },
  );
}
