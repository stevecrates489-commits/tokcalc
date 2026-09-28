/**
 * tokcalc MCP Server — website surface (THIN SHIM).
 *
 * The real implementation lives in ./core.ts — the single source of truth
 * shared with the npm package (mini-services/mcp-server/server.ts, which is
 * itself a shim over this core). This file only re-exports it so the
 * @/lib/mcp/server import in src/app/api/mcp/route.ts keeps working.
 *
 * Why: the tool handlers used to be duplicated here and in the npm package.
 * Two copies is how a fix landed on one surface and not the other during the
 * 0.2.2–0.2.4 stale-dist episodes. Handlers now live in exactly one place; a
 * fix lands on both surfaces in the same deploy, by construction.
 */

export {
  SERVER_VERSION,
  createMcpServer,
} from "./core";
