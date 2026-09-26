/**
 * tokcalc MCP Server — Public hosted endpoint (v0.2.0 final).
 *
 * URL: https://tokcalc.vercel.app/api/mcp
 *
 * BUGFIX (v0.2.1): Previous version called `await req.text()` to validate
 * the body, which consumed the one-shot Request stream. The SDK's
 * `handleRequest(req)` then got an empty body → returned -32700
 * "Parse error: Invalid JSON". Fix: don't consume the body ourselves;
 * let the SDK read it. We only use Content-Length header for size check.
 */

import { NextRequest } from "next/server";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { createMcpServer } from "@/lib/mcp/server";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import crypto from "node:crypto";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 30;

const ANON_IP_LIMIT = Number(process.env.MCP_RATE_LIMIT_ANON_IP_PER_MIN ?? 30);
const AUTHED_KEY_LIMIT = Number(process.env.MCP_RATE_LIMIT_AUTHED_KEY_PER_MIN ?? 120);
const MAX_BODY_BYTES = 256 * 1024;

let redisClient: Redis | null = null;
let anonIpLimiter: Ratelimit | null = null;
let authedKeyLimiter: Ratelimit | null = null;

function getLimiters(): { anonIp: Ratelimit; authedKey: Ratelimit } | null {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) return null;
  if (!redisClient) redisClient = new Redis({ url, token });
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

function isAuthEnabled(): boolean {
  return !!process.env.MCP_API_KEY;
}

function safeEqual(a: string, b: string): boolean {
  const aa = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  if (aa.length !== bb.length) return false;
  return crypto.timingSafeEqual(aa, bb);
}

function hashKey(key: string): string {
  return crypto.createHash("sha256").update(key).digest("hex").slice(0, 16);
}

function extractBearer(req: NextRequest): string | null {
  const authHeader = req.headers.get("authorization");
  if (!authHeader) return null;
  const match = /^Bearer\s+(.+)$/i.exec(authHeader.trim());
  return match ? match[1].trim() : null;
}

function getClientIp(req: NextRequest): string {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  const cfIp = req.headers.get("cf-connecting-ip");
  if (cfIp) return cfIp.trim();
  return "unknown";
}

interface AuthResult {
  authenticated: boolean;
  rateLimitId: string;
  reason?: string;
}

function authenticateRequest(req: NextRequest): AuthResult {
  const expectedKey = process.env.MCP_API_KEY;
  const clientIp = getClientIp(req);
  if (!expectedKey) {
    return { authenticated: false, rateLimitId: `anon-ip:${clientIp}` };
  }
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
      rateLimitId: `bad-key:${hashKey(providedKey)}`,
    };
  }
  return { authenticated: true, rateLimitId: `key:${hashKey(providedKey)}` };
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

function mcpError(statusCode: number, code: number, message: string, detail?: string, extraHeaders?: Record<string, string>): Response {
  return new Response(
    JSON.stringify({
      jsonrpc: "2.0",
      error: { code, message, data: detail ? { detail } : undefined },
      id: null,
    }),
    {
      status: statusCode,
      headers: { "Content-Type": "application/json", ...extraHeaders },
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
// POST HANDLER
// IMPORTANT: Do NOT call req.text() / req.json() ourselves.
// Per Fetch API, Request.body is a one-shot ReadableStream.
// The SDK's handleRequest(req) reads the body internally.
// If we consume it first, the SDK gets empty body → -32700 error.
// ============================================================

export async function POST(req: NextRequest): Promise<Response> {
  // Validate MCP-Protocol-Version header (does NOT consume body)
  if (!req.headers.get("mcp-protocol-version")) {
    return mcpError(400, -32600, "Missing MCP-Protocol-Version header",
      "Streamable HTTP requires the MCP-Protocol-Version header. Use '2025-03-26' for the latest Streamable HTTP spec.");
  }

  // AUTH (uses only headers — safe)
  const authResult = authenticateRequest(req);
  if (isAuthEnabled() && !authResult.authenticated && authResult.reason) {
    return unauthorized(authResult.reason);
  }

  // RATE LIMIT (uses Upstash — does NOT touch req body)
  const rateLimitResult = await checkRateLimit(authResult.rateLimitId, authResult.authenticated);
  if (!rateLimitResult.success) {
    return tooManyRequests(rateLimitResult);
  }

  // BODY SIZE CHECK (via Content-Length header — does NOT consume body stream)
  const contentLength = Number(req.headers.get("content-length") ?? 0);
  if (contentLength > MAX_BODY_BYTES) {
    return mcpError(413, -32600, "Payload Too Large",
      `Body exceeds ${MAX_BODY_BYTES} byte limit (${contentLength} bytes received).`);
  }

  // STATELESS TRANSPORT
  // Pass the Request directly to the SDK — it reads the body itself.
  // DO NOT call req.text() / req.json() / req.arrayBuffer() before this!
  try {
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
    });
    const mcpServer = createMcpServer();
    await mcpServer.connect(transport);

    const response = await transport.handleRequest(req);
    return response;
  } catch (err) {
    console.error("[tokcalc-mcp/api] Transport error:", err);
    return mcpError(500, -32603, "Internal server error",
      err instanceof Error ? err.message : String(err));
  }
}

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