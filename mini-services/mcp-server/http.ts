#!/usr/bin/env node
/**
 * tokcalc MCP Server — HTTP transport entry point (v0.2.0-alpha.1).
 *
 * Implements the MCP Streamable HTTP transport (protocol version 2025-03-26)
 * in STATELESS mode. All 7 tools are read-only and don't depend on session
 * state, so we don't need session storage — a new transport + server instance
 * is created per request, which is clean for serverless/edge deployment.
 *
 * ─────────────────────────────────────────────────────────────────────
 * v0.2.0-alpha.1 scope:
 *   ✓ Stateless HTTP only (POST /mcp, no SSE stream, no sessions)
 *   ✓ MCP-Protocol-Version header validation
 *   ✓ JSON error responses for malformed input
 *   ✗ NO authentication (added in 0.2.0-beta.1 — bearer API key)
 *   ✗ NO rate limiting (added in 0.2.0-beta.1 — KV-backed)
 *   ✗ NO SSE streaming (not needed for read-only tools)
 *   ✗ NO CORS (MCP clients are server-to-server, not browser)
 * ─────────────────────────────────────────────────────────────────────
 *
 * Run locally:
 *   bun run http.ts                          # development
 *   PORT=8080 bun run http.ts                # custom port
 *   node dist/http.js                        # production build
 *
 * Build (run from mini-services/mcp-server/):
 *   bun build http.ts --target=node --outfile dist/http.js
 *
 * Test with curl:
 *   curl -X POST http://localhost:3000/mcp \
 *     -H "Content-Type: application/json" \
 *     -H "Accept: application/json" \
 *     -H "MCP-Protocol-Version: 2025-03-26" \
 *     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
 *
 * Per v0.2.0 research (Perplexity brief, 2025-09-25):
 *   - Single /mcp endpoint per Streamable HTTP spec
 *   - POST returns application/json for tool calls (no SSE needed)
 *   - Stateless: sessionIdGenerator: undefined
 *   - Factory pattern: each request gets fresh server instance via createMcpServer()
 *   - No shared transport state across requests (avoids mistake #5, #6)
 *   - 405 for GET / DELETE (stateless mode doesn't support SSE stream / sessions)
 *   - Logs to stdout are safe in HTTP mode (only stdio mode reserves stdout for JSON-RPC)
 */

import http from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createMcpServer } from "./server.js";

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
 * Enforces a 1MB body size limit (per v0.2.0 research recommendation:
 * "maximum request body: 64–256 KB" — we use 1MB to be lenient in alpha.1).
 */
async function readJsonBody(req: http.IncomingMessage): Promise<unknown> {
  const MAX_BODY_BYTES = 1024 * 1024; // 1MB
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
  console.log(`║  tokcalc MCP HTTP server (v0.2.0-alpha.1)                    ║`);
  console.log(`║  Listening on http://localhost:${PORT}/mcp` + " ".repeat(Math.max(0, 26 - String(PORT).length)) + "║");
  console.log(`╚══════════════════════════════════════════════════════════════╝`);
  console.log("");
  console.log(`  Tools:     7 (estimate_capacity, compare_gpus, recommend_topology,`);
  console.log(`                estimate_api_vs_self_host, list_models, list_gpus,`);
  console.log(`                get_mlperf_benchmarks)`);
  console.log(`  Mode:      stateless (no sessions, no SSE stream)`);
  console.log(`  Auth:      NONE (added in 0.2.0-beta.1 — bearer API key)`);
  console.log(`  Rate limit: NONE (added in 0.2.0-beta.1 — KV-backed)`);
  console.log(`  Protocol:  MCP Streamable HTTP (2025-03-26)`);
  console.log("");
  console.log(`  Test with curl:`);
  console.log(`    curl -X POST http://localhost:${PORT}/mcp \\`);
  console.log(`      -H "Content-Type: application/json" \\`);
  console.log(`      -H "Accept: application/json" \\`);
  console.log(`      -H "MCP-Protocol-Version: 2025-03-26" \\`);
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
