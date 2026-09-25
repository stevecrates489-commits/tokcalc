/**
 * Bearer API key authentication middleware for tokcalc MCP HTTP server.
 *
 * Per v0.2.0 research (Perplexity brief, 2025-09-25):
 *   "For a small public read-only service, a static API key is operationally
 *    acceptable. Use a static key if the key identifies a customer or deployment
 *    rather than an individual user, the tools expose no private user data,
 *    you can revoke and rotate keys, you document that clients must send it
 *    as a bearer credential, you return proper 401 responses, you do not put
 *    the key in a URL, and you are prepared for clients that only understand
 *    OAuth."
 *
 * Design:
 *   1. Read MCP_API_KEY from env var (set on Vercel: Settings → Environment Variables)
 *   2. If env var is NOT set → allow anonymous access (with stricter rate limits in rate-limit.ts)
 *   3. If env var IS set → require Authorization: Bearer <key> on every request
 *   4. Use crypto.timingSafeEqual for constant-time comparison (prevents timing attacks)
 *   5. NEVER log the raw token — only its SHA-256 hash for debugging
 *
 * Roadmap:
 *   v0.2.0-beta.1 (this file): static env var key (Option C)
 *   v0.2.0 final:              upgrade to DB-backed keys (Option B) using Prisma + SQLite
 *   v0.3.0:                     OAuth 2.1 with PKCE
 */

import crypto from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

/**
 * Determine if the request is authenticated.
 *
 * Returns:
 *   - { authenticated: true, keyHash: string } if valid bearer token
 *   - { authenticated: false, reason: string } if not authenticated
 *
 * "Not authenticated" is NOT an error if MCP_API_KEY is unset — in that case
 * we allow anonymous access (rate-limited more strictly in rate-limit.ts).
 * "Not authenticated" IS an error (401) if MCP_API_KEY IS set but the client
 * didn't send a matching bearer token.
 */
export interface AuthResult {
  authenticated: boolean;
  /** SHA-256 hash of the provided key (for logging/debugging only — never log raw) */
  keyHash?: string;
  /** Identifier for rate-limit bucketing (keyHash if authenticated, IP if anonymous) */
  rateLimitId: string;
  /** Reason for auth failure (only set when authenticated === false AND MCP_API_KEY is set) */
  reason?: string;
}

/**
 * Constant-time string comparison to prevent timing attacks.
 * Returns true if strings are equal length AND content matches.
 */
function safeEqual(a: string, b: string): boolean {
  const aa = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  if (aa.length !== bb.length) return false;
  return crypto.timingSafeEqual(aa, bb);
}

/**
 * SHA-256 hash of an API key (for safe logging / rate-limit bucketing).
 * Never log or store the raw key — only its hash.
 */
export function hashKey(key: string): string {
  return crypto.createHash("sha256").update(key).digest("hex").slice(0, 16);
}

/**
 * Extract the bearer token from the Authorization header.
 * Returns null if missing or malformed.
 */
function extractBearer(req: IncomingMessage): string | null {
  const authHeader = req.headers["authorization"];
  if (!authHeader || typeof authHeader !== "string") return null;

  // Match "Bearer <token>" case-insensitively
  const match = /^Bearer\s+(.+)$/i.exec(authHeader.trim());
  return match ? match[1].trim() : null;
}

/**
 * Get the client IP address from the request.
 * Handles common proxy headers (X-Forwarded-For, X-Real-IP) for Vercel/Cloudflare.
 */
export function getClientIp(req: IncomingMessage): string {
  // Vercel/Cloudflare proxy chain (take the first IP in the list)
  const xff = req.headers["x-forwarded-for"];
  if (typeof xff === "string") {
    return xff.split(",")[0].trim();
  }
  if (Array.isArray(xff) && xff.length > 0) {
    return xff[0].split(",")[0].trim();
  }

  // Cloudflare-specific
  const cfIp = req.headers["cf-connecting-ip"];
  if (typeof cfIp === "string") return cfIp.trim();

  // X-Real-IP (Nginx)
  const realIp = req.headers["x-real-ip"];
  if (typeof realIp === "string") return realIp.trim();

  // Direct connection (local dev)
  return req.socket.remoteAddress || "unknown";
}

/**
 * Validate the request's authentication.
 *
 * Behavior:
 *   - If MCP_API_KEY env var is unset → anonymous access allowed (rate-limited per IP)
 *   - If MCP_API_KEY is set:
 *     - If Authorization header matches → authenticated (rate-limited per key, looser)
 *     - If header missing or wrong → return 401 (will be enforced by caller)
 *
 * @returns AuthResult — caller decides whether to send 401 or proceed
 */
export function authenticateRequest(req: IncomingMessage): AuthResult {
  const expectedKey = process.env.MCP_API_KEY;
  const clientIp = getClientIp(req);

  // === Anonymous mode (MCP_API_KEY not set) ===
  // Allow anonymous access; rate-limit per IP more strictly.
  if (!expectedKey) {
    return {
      authenticated: false,
      rateLimitId: `anon-ip:${clientIp}`,
    };
  }

  // === Authenticated mode (MCP_API_KEY is set) ===
  const providedKey = extractBearer(req);

  if (!providedKey) {
    return {
      authenticated: false,
      reason: "Missing Authorization header. Expected: Authorization: Bearer <your-api-key>",
      rateLimitId: `anon-ip:${clientIp}`,
    };
  }

  if (!safeEqual(providedKey, expectedKey)) {
    return {
      authenticated: false,
      reason: "Invalid API key.",
      rateLimitId: `bad-key:${hashKey(providedKey)}`, // bucket by hash so attacker can't pollute legit buckets
    };
  }

  // Valid bearer token — return key hash for rate-limit bucketing
  return {
    authenticated: true,
    keyHash: hashKey(providedKey),
    rateLimitId: `key:${hashKey(providedKey)}`,
  };
}

/**
 * Send a proper 401 Unauthorized response with WWW-Authenticate header.
 * Per HTTP spec + MCP spec: bearer auth should return WWW-Authenticate on failure.
 */
export function sendUnauthorized(res: ServerResponse, reason: string): void {
  res.writeHead(401, {
    "Content-Type": "application/json",
    "WWW-Authenticate": 'Bearer realm="tokcalc-mcp"',
  });
  res.end(JSON.stringify({
    jsonrpc: "2.0",
    error: {
      code: -32001,
      message: "Unauthorized",
      data: { detail: reason },
    },
    id: null,
  }));
}

/**
 * Determine whether to enforce auth (i.e., is MCP_API_KEY set?).
 * Useful for logging / startup banners.
 */
export function isAuthEnabled(): boolean {
  return !!process.env.MCP_API_KEY;
}
