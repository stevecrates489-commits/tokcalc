#!/usr/bin/env node
/**
 * tokcalc MCP Server — stdio transport entry point.
 *
 * Default entry point for `npx tokcalc-mcp-server` and the three npm bin
 * aliases (`tokcalc-mcp`, `tokcalc-mcp-server`, `mcp-server`).
 *
 * Used by:
 *   - Local Cursor / Claude Desktop / Cline installs (stdio transport)
 *   - Self-hosted deployments that prefer stdio over HTTP
 *
 * For HTTP transport (v0.2.0-alpha.1+), use the http.ts entry point:
 *   tokcalc-mcp-http --port 3000
 *
 * Per v0.2.0 research: each entry point creates its own Server instance
 * via the createMcpServer() factory (no shared transport state).
 */

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createMcpServer } from "./server.js";

const transport = new StdioServerTransport();
await createMcpServer().connect(transport);

// IMPORTANT: log to stderr only — stdout is reserved for JSON-RPC framing
// (per v0.2.0 research, common mistake #14: "Writing logs to stdout in stdio mode")
console.error("tokcalc MCP server started (stdio transport) — 7 tools available");
