#!/usr/bin/env node
/**
 * tokcalc MCP Server — HTTP transport entry point (v0.2.0-beta.1).
 *
 * Implements the MCP Streamable HTTP transport (protocol version 2025-03-26)
 * in STATELESS mode with BEARER API KEY auth + KV-BACKED RATE LIMITING.
 *
 * ─────────────────────────────────────────────────────────────────────
 * v0.2.0-beta.1 scope (upgrade from alpha.1):
 *   ✓ Stateless HTTP only (POST /mcp, no SSE stream, no sessions)
 *   ✓ MCP-Protocol-Version header validation
 *   ✓ JSON error responses for malformed input
 *   ✓ Bearer API key authentication (env var MCP_API_KEY)  ← NEW
 *   ✓ KV-backed rate limiting (Upstash Redis, sliding window)  ← NEW
 *   ✓ Request body size limit reduced to 256KB (was 1MB)  ← TIGHTENED
 *   ✗ NO SSE streaming (not needed for read-only tools)
 *   ✗ NO CORS (MCP clients are server-to-server, not browser)
 *   ✗ NO OAuth (deferred to v0.3.0 — bearer key is sufficient for beta.1)
 * ─────────────────────────────────────────────────────────────────────
 *
 * Required env vars (set on Vercel: Settings → Environment Variables):
 *   MCP_API_KEY                  — bearer API key clients must send (REQUIRED for prod)
 *   UPSTASH_REDIS_REST_URL       — Upstash Redis REST URL (REQUIRED for rate limiting)
 *   UPSTASH_REDIS_REST_TOKEN     — Upstash Redis REST token (REQUIRED for rate limiting)
 *
 * Optional overrides:
 *   PORT                                (default: 3000)
 *   MCP_RATE_LIMIT_ANON_IP_PER_MIN      (default: 30)  — limit for unauthenticated IP
 *   MCP_RATE_LIMIT_AUTHED_KEY_PER_MIN   (default: 120) — limit for authenticated API key
 *
 * Run locally (without env vars — auth disabled, rate limiting disabled):
 *   bun run http.ts
 *   PORT=8080 bun run http.ts
 *   node dist/http.js
 *
 * Run locally WITH auth + rate limit (set env vars first):
 *   export MCP_API_KEY="secret-key-here"
 *   export UPSTASH_REDIS_REST_URL="https://your-db.upstash.io"
 *   export UPSTASH_REDIS_REST_TOKEN="your-token"
 *   bun run http.ts
 *
 * Build (run from mini-services/mcp-server/):
 *   bun build http.ts --target=node --outfile dist/http.js
 *
 * Test with curl (no auth — anonymous, will be rate-limited):
 *   curl -X POST http://localhost:3000/mcp \
 *     -H "Content-Type: application/json" \
 *     -H "Accept: application/json" \
 *     -H "MCP-Protocol-Version: 2025-03-26" \
 *     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
 *
 * Test with curl (with auth — authenticated, looser rate limit):
 *   curl -X POST http://localhost:3000/mcp \
 *     -H "Content-Type: application/json" \
 *     -H "Accept: application/json" \
 *     -H "MCP-Protocol-Version: 2025-03-26" \
 *     -H "Authorization: Bearer $MCP_API_KEY" \
 *     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
 *
 * Per v0.2.0 research (Perplexity brief, 2025-09-25):
 *   - Single /mcp endpoint per Streamable HTTP spec
 *   - Stateless: sessionIdGenerator: undefined
 *   - Factory pattern: each request gets fresh server instance
 *   - Constant-time API key comparison (prevents timing attacks)
 *   - Sliding window rate limit via Upstash Redis
 *   - Returns 401 with WWW-Authenticate: Bearer on auth failure
 *   - Returns 429 with RateLimit-* + Retry-After headers on rate limit hit
 *   - NEVER logs raw API keys — only their SHA-256 hash
 */

import http from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createMcpServer } from "./server.js";
import { authenticateRequest, sendUnauthorized, isAuthEnabled, getClientIp } from "./auth.js";
import { checkRateLimit, sendTooManyRequests, isRateLimitEnabled } from "./rate-limit.js";

const PORT = Number(process.env.PORT ?? 3000);

/**
 * Validate that the MCP-Protocol-Version header is present.
 *
 * Per Streamable HTTP spec (2025-03-26): clients MUST send this header.
 * We don't enforce a specific version in alpha.1 (just presence); beta.1
 * will validate against a known-good list.
 */
function validateProtocolVersion(req: http.IncomingMessage): boolean {
  return !!req.headers["mcp-protocol-version"];
}

/**
 * Read and parse the JSON body from an HTTP request.
 *
 * Body size limit tightened from 1MB (alpha.1) to 256KB (beta.1) per
 * v0.2.0 research: "maximum request body: 64–256 KB".
 */
async function readJsonBody(req: http.IncomingMessage): Promise<unknown> {
  const MAX_BODY_BYTES = 256 * 1024; // 256KB (tightened from 1MB in alpha.1)
  const chunks: Buffer[] = [];
  let totalBytes = 0;

  for await (const chunk of req) {
    totalBytes += chunk.length;
    if (totalBytes > MAX_BODY_BYTES) {
      throw new Error(`Body too large: ${totalBytes} bytes exceeds ${MAX_BODY_BYTES} byte limit`);
    }
    chunks.push(chunk as Buffer);
  }

  const bodyText = Buffer.concat(chunks).toString("utf8");
  if (!bodyText.trim()) {
    throw new Error("Empty request body");
  }
  return JSON.parse(bodyText);
}

/**
 * Send a JSON error response with proper MCP-style structure.
 */
function sendError(
  res: http.ServerResponse,
  statusCode: number,
  error: string,
  message?: string,
): void {
  res.writeHead(statusCode, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    jsonrpc: "2.0",
    error: {
      code: statusCode === 400 ? -32600 : statusCode === 405 ? -32601 : -32603,
      message: error,
      data: message ? { detail: message } : undefined,
    },
    id: null,
  }));
}

// ============================================================
// HTTP SERVER
// ============================================================

const httpServer = http.createServer(async (req, res) => {
  // Only handle /mcp path (per Streamable HTTP spec convention)
  if (req.url !== "/mcp") {
    sendError(res, 404, "Not found", `Path ${req.url} is not a valid MCP endpoint. Use POST /mcp.`);
    return;
  }

  // Stateless mode only supports POST (no GET SSE stream, no DELETE session)
  if (req.method !== "POST") {
    res.writeHead(405, {
      "Content-Type": "application/json",
      "Allow": "POST",
    });
    res.end(JSON.stringify({
      jsonrpc: "2.0",
      error: {
        code: -32601,
        message: `Method ${req.method} not allowed on this server`,
        data: {
          detail: "Stateless HTTP MCP server — only POST is supported. GET (SSE stream) and DELETE (session) are not implemented in stateless mode. Use a stateful server if you need streaming.",
        },
      },
      id: null,
    }));
    return;
  }

  // Validate MCP-Protocol-Version header (per Streamable HTTP spec)
  if (!validateProtocolVersion(req)) {
    sendError(
      res,
      400,
      "Missing MCP-Protocol-Version header",
      "Streamable HTTP requires the MCP-Protocol-Version header. Use '2025-03-26' for the latest Streamable HTTP spec.",
    );
    return;
  }

  // ═══════════════════════════════════════════════════════════════
  // v0.2.0-beta.1 — AUTHENTICATION MIDDLEWARE
  // Order: auth FIRST, then rate limit, then body parse, then transport.
  // Rationale: don't read the body until we know the client is allowed.
  // ═══════════════════════════════════════════════════════════════
  const authResult = authenticateRequest(req);

  // If MCP_API_KEY is set and the request is unauthenticated, reject with 401.
  // (If MCP_API_KEY is NOT set, anonymous access is allowed — anonymous mode
  //  is enforced via the rate limiter's stricter per-IP limit.)
  if (isAuthEnabled() && !authResult.authenticated && authResult.reason) {
    sendUnauthorized(res, authResult.reason);
    return;
  }

  // ═══════════════════════════════════════════════════════════════
  // v0.2.0-beta.1 — RATE LIMIT MIDDLEWARE
  // Check before processing the request — fail fast.
  // Bucket: per-IP for anonymous, per-key for authenticated.
  // ═══════════════════════════════════════════════════════════════
  const rateLimitResult = await checkRateLimit(
    authResult.rateLimitId,
    authResult.authenticated,
  );
  if (!rateLimitResult.success) {
    sendTooManyRequests(res, rateLimitResult);
    return;
  }

  // Read and parse body
  let body: unknown;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    sendError(
      res,
      400,
      "Invalid JSON body",
      err instanceof Error ? err.message : String(err),
    );
    return;
  }

  // Stateless: create a fresh transport + server per request.
  // (All 7 tools are read-only — no need for session state.)
  // Per v0.2.0 research, mistake #5: "Creating a new transport for every
  // follow-up request instead of reusing the session transport" — this is
  // fine for STATELESS mode where there's no session to reuse.
  try {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined, // ← stateless mode (no sessions)
    });
    const mcpServer = createMcpServer();
    await mcpServer.connect(transport);

    // Let the SDK handle the JSON-RPC request/response cycle
    await transport.handleRequest(req, res, body as Record<string, unknown>);

    // After response is sent, transport + server can be GC'd.
    // No cleanup needed — no session state was created.
  } catch (err) {
    if (!res.headersSent) {
      sendError(
        res,
        500,
        "Internal server error",
        err instanceof Error ? err.message : String(err),
      );
    } else {
      // Headers already sent — can only log to stderr
      console.error("[tokcalc-mcp-http] Error after headers sent:", err);
    }
  }
});

// ============================================================
// START SERVER
// ============================================================

httpServer.listen(PORT, () => {
  console.log(`╔══════════════════════════════════════════════════════════════╗`);
  console.log(`║  tokcalc MCP HTTP server (v0.2.0-beta.1)                     ║`);
  console.log(`║  Listening on http://localhost:${PORT}/mcp` + " ".repeat(Math.max(0, 26 - String(PORT).length)) + "║");
  console.log(`╚══════════════════════════════════════════════════════════════╝`);
  console.log("");
  console.log(`  Tools:     7 (estimate_capacity, compare_gpus, recommend_topology,`);
  console.log(`                estimate_api_vs_self_host, list_models, list_gpus,`);
  console.log(`                get_mlperf_benchmarks)`);
  console.log(`  Mode:      stateless (no sessions, no SSE stream)`);
  console.log(`  Auth:      ${isAuthEnabled() ? "ENABLED — bearer API key (MCP_API_KEY set)" : "DISABLED — anonymous access (MCP_API_KEY not set)"}`);
  console.log(`  Rate limit: ${isRateLimitEnabled() ? "ENABLED — Upstash Redis (UPSTASH_REDIS_REST_URL set)" : "DISABLED — no KV configured (UPSTASH_REDIS_REST_URL not set)"}`);
  console.log(`  Protocol:  MCP Streamable HTTP (2025-03-26)`);
  console.log("");
  console.log(`  Test with curl:`);
  console.log(`    curl -X POST http://localhost:${PORT}/mcp \\`);
  console.log(`      -H "Content-Type: application/json" \\`);
  console.log(`      -H "Accept: application/json" \\`);
  console.log(`      -H "MCP-Protocol-Version: 2025-03-26" \\`);
  if (isAuthEnabled()) {
    console.log(`      -H "Authorization: Bearer \$MCP_API_KEY" \\`);
  }
  console.log(`      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'`);
  console.log("");
  console.log(`  Press Ctrl+C to stop.`);
});

// Graceful shutdown
process.on("SIGINT", () => {
  console.log("\n Shutting down...");
  httpServer.close(() => process.exit(0));
});

process.on("SIGTERM", () => {
  httpServer.close(() => process.exit(0));
});
