/**
 * tokcalc MCP Server — npm package surface (THIN SHIM).
 *
 * The real implementation lives in src/lib/mcp/core.ts — the single source of
 * truth shared with the website's /api/mcp endpoint. This file only re-exports
 * it so the bundler-relative imports in index.ts / http.ts keep working.
 *
 * Why: the tool handlers used to be duplicated here and in src/lib/mcp/server.ts.
 * Two copies is how a fix landed on one surface and not the other during the
 * 0.2.2–0.2.4 stale-dist episodes. Handlers now live in exactly one place; a
 * fix lands on both surfaces in the same build, by construction.
 */

export {
  SERVER_VERSION,
  createMcpServer,
} from "../../src/lib/mcp/core";
