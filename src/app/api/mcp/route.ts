/**
 * tokcalc MCP Server — Public hosted endpoint (v0.2.0 final).
 *
 * URL: https://tokcalc.vercel.app/api/mcp
 *
 * Implements the MCP Streamable HTTP transport (protocol version 2025-03-26)
 * in STATELESS mode with BEARER API KEY auth + KV-BACKED RATE LIMITING.
 *
 * Uses the Web-standard transport (`WebStandardStreamableHTTPServerTransport`)
 * because Next.js Route Handlers use the Web Fetch API (Request/Response),
 * not Node's IncomingMessage/ServerResponse.
 *
 * ─────────────────────────────────────────────────────────────────────
 * Required env vars (set on Vercel: Settings → Environment Variables):
 *   MCP_API_KEY                  — bearer API key clients must send
 *   UPSTASH_REDIS_REST_URL       — Upstash Redis REST URL (for rate limiting)
 *   UPSTASH_REDIS_REST_TOKEN     — Upstash Redis REST token
 *
 * Optional overrides:
 *   MCP_RATE_LIMIT_ANON_IP_PER_MIN      (default: 30)
 *   MCP_RATE_LIMIT_AUTHED_KEY_PER_MIN   (default: 120)
 * ─────────────────────────────────────────────────────────────────────
 *
 * Test with curl (after deploy):
 *   curl -X POST https://tokcalc.vercel.app/api/mcp \
 *     -H "Content-Type: application/json" \
 *     -H "Accept: application/json, text/event-stream" \
 *     -H "MCP-Protocol-Version: 2025-03-26" \
 *     -H "Authorization: Bearer $MCP_API_KEY" \
 *     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
 *
 * Cursor config (for hosted HTTP):
 *   {
 *     "mcpServers": {
 *       "tokcalc": {
 *         "url": "https://tokcalc.vercel.app/api/mcp",
 *         "headers": { "Authorization": "Bearer <your-api-key>" }
 *       }
 *     }
 *   }
 *
 * Claude Desktop config (for hosted HTTP — newer Claude versions):
 *   {
 *     "mcpServers": {
 *       "tokcalc": {
 *         "type": "http",
 *         "url": "https://tokcalc.vercel.app/api/mcp",
 *         "headers": { "Authorization": "Bearer <your-api-key>" }
 *       }
 *     }
 *   }
 */

import { NextRequest } from "next/server";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { createMcpServer } from "@/lib/mcp/server";
import { validateApiKey, isAuthEnabled } from "@/lib/mcp/auth";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

// Force Node.js runtime (we need crypto module + Upstash client works on Node)
export const runtime = "nodejs";
// Never cache MCP responses — each request needs fresh transport + server
export const dynamic = "force-dynamic";
// Allow up to 30s for tool calls (some calculate() calls are compute-heavy)
export const maxDuration = 30;

// ============================================================
// CONSTANTS
// ============================================================

const ANON_IP_LIMIT = Number(process.env.MCP_RATE_LIMIT_ANON_IP_PER_MIN ?? 30);
const AUTHED_KEY_LIMIT = Number(process.env.MCP_RATE_LIMIT_AUTHED_KEY_PER_MIN ?? 120);
const MAX_BODY_BYTES = 256 * 1024; // 256KB

// ============================================================
// SINGLETONS (lazy-initialized)
// ============================================================

let redisClient: Redis | null = null;
let anonIpLimiter: Ratelimit | null = null;
let authedKeyLimiter: Ratelimit | null = null;

function getLimiters(): { anonIp: Ratelimit; authedKey: Ratelimit } | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;

  if (!redisClient) {
    redisClient = new Redis({ url, token });
  }
  if (!anonIpLimiter) {
    anonIpLimiter = new Ratelimit({
      redis: redisClient,
      limiter: Ratelimit.slidingWindow(ANON_IP_LIMIT, "1 m"),
      prefix: "tokcalc-mcp:anon-ip",
    });
  }
  if (!authedKeyLimiter) {
    authedKeyLimiter = new Ratelimit({
      redis: redisClient,
      limiter: Ratelimit.slidingWindow(AUTHED_KEY_LIMIT, "1 m"),
      prefix: "tokcalc-mcp:authed-key",
    });
  }
  return { anonIp: anonIpLimiter, authedKey: authedKeyLimiter };
}

// ============================================================
// RATE LIMIT HELPERS (auth is now in @/lib/mcp/auth.ts)
// ============================================================

function getClientIp(req: NextRequest): string {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  const cfIp = req.headers.get("cf-connecting-ip");
  if (cfIp) return cfIp.trim();
  return "unknown";
}

interface RateLimitResult {
  success: boolean;
  limit: number;
  remaining: number;
  reset: number;
  bucket: "anon-ip" | "authed-key" | "none";
}

async function checkRateLimit(rateLimitId: string, authenticated: boolean): Promise<RateLimitResult> {
  const limiters = getLimiters();
  if (!limiters) {
    return { success: true, limit: Infinity, remaining: Infinity, reset: 0, bucket: "none" };
  }

  const ipResult = await limiters.anonIp.limit(rateLimitId);

  if (authenticated && rateLimitId.startsWith("key:")) {
    const keyResult = await limiters.authedKey.limit(rateLimitId);
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

// ============================================================
// ERROR RESPONSE HELPERS
// ============================================================

function mcpError(statusCode: number, code: number, message: string, detail?: string, extraHeaders?: Record<string, string>): Response {
  return new Response(
    JSON.stringify({
      jsonrpc: "2.0",
      error: {
        code,
        message,
        data: detail ? { detail } : undefined,
      },
      id: null,
    }),
    {
      status: statusCode,
      headers: {
        "Content-Type": "application/json",
        ...extraHeaders,
      },
    },
  );
}

function unauthorized(reason: string): Response {
  return mcpError(401, -32001, "Unauthorized", reason, {
    "WWW-Authenticate": 'Bearer realm="tokcalc-mcp"',
  });
}

function tooManyRequests(result: RateLimitResult): Response {
  const retryAfterSec = Math.max(1, Math.ceil((result.reset * 1000 - Date.now()) / 1000));
  return mcpError(429, -32002, "Too Many Requests",
    `Rate limit exceeded on bucket "${result.bucket}". Try again in ${retryAfterSec} second${retryAfterSec === 1 ? "" : "s"}.`,
    {
      "Retry-After": String(retryAfterSec),
      "RateLimit-Limit": String(result.limit === Infinity ? 0 : result.limit),
      "RateLimit-Remaining": String(result.remaining === Infinity ? 0 : result.remaining),
      "RateLimit-Reset": String(result.reset),
      "X-RateLimit-Bucket": result.bucket,
    },
  );
}

// ============================================================
// POST HANDLER (the main MCP endpoint)
// ============================================================
//
// IMPORTANT: We do NOT consume Request.body ourselves.
// Per Fetch API: Request.body is a one-shot ReadableStream. If we call
// req.text() / req.json() / req.arrayBuffer() ourselves, the SDK's later
// call to req.text() gets an EMPTY body → SDK returns -32700
// "Parse error: Invalid JSON". So we only use headers for validation.
// The SDK's WebStandardStreamableHTTPServerTransport.handleRequest(req)
// reads + parses the body itself and returns a Response.

export async function POST(req: NextRequest): Promise<Response> {
  // Validate MCP-Protocol-Version header
  if (!req.headers.get("mcp-protocol-version")) {
    return mcpError(400, -32600, "Missing MCP-Protocol-Version header",
      "Streamable HTTP requires the MCP-Protocol-Version header. Use '2025-03-26' for the latest Streamable HTTP spec.");
  }

  // ── AUTH (now uses @/lib/mcp/auth — checks Redis for user keys + MCP_API_KEY env var) ──
  const clientIp = getClientIp(req);
  const authHeader = req.headers.get("authorization");
  const authResult = await validateApiKey(authHeader, clientIp);
  if (isAuthEnabled() && !authResult.valid) {
    const reason = !authHeader
      ? "Missing Authorization header. Expected: Authorization: Bearer <your-api-key>"
      : "Invalid API key. Get one at https://tokcalc.vercel.app/mcp";
    return unauthorized(reason);
  }

  // ── RATE LIMIT ──
  const rateLimitResult = await checkRateLimit(authResult.rateLimitId, authResult.valid);
  if (!rateLimitResult.success) {
    return tooManyRequests(rateLimitResult);
  }

  // ── BODY SIZE CHECK (via Content-Length header — does NOT consume the body stream) ──
  // The SDK will read + parse the body itself; we just pre-flight the size.
  const contentLength = Number(req.headers.get("content-length") ?? 0);
  if (contentLength > MAX_BODY_BYTES) {
    return mcpError(413, -32600, "Payload Too Large",
      `Body exceeds ${MAX_BODY_BYTES} byte limit (${contentLength} bytes received).`);
  }

  // ── STATELESS TRANSPORT ──
  // Pass the Request to the SDK directly — the SDK reads the body via
  // req.text() internally and dispatches the JSON-RPC message.
  // DO NOT call req.text() / req.json() / req.body ourselves before this.
  try {
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: undefined, // ← stateless mode (no sessions)
    });
    const mcpServer = createMcpServer();
    await mcpServer.connect(transport);

    // SDK handles: body read → JSON parse → JSON-RPC dispatch → response
    const response = await transport.handleRequest(req);
    return response;
  } catch (err) {
    console.error("[tokcalc-mcp/api] Transport error:", err);
    return mcpError(500, -32603, "Internal server error",
      err instanceof Error ? err.message : String(err));
  }
}

// ============================================================
// GET / DELETE — stateless mode doesn't support these
// ============================================================

export async function GET(): Promise<Response> {
  return mcpError(405, -32601, "Method GET not allowed on this server",
    "Stateless HTTP MCP server — only POST is supported. GET (SSE stream) and DELETE (session) are not implemented in stateless mode.",
    { Allow: "POST" },
  );
}

export async function DELETE(): Promise<Response> {
  return mcpError(405, -32601, "Method DELETE not allowed on this server",
    "Stateless HTTP MCP server — no sessions to delete.",
    { Allow: "POST" },
  );
}
