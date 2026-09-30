/**
 * Single source of truth for the MCP surface versions.
 *
 * Why this file exists: SERVER_VERSION was hardcoded to "0.2.4" in server.ts
 * while mini-services/mcp-server/package.json said 0.2.5 — the divergence was
 * only visible because external QA diffed the provenance fingerprint against
 * npm. Now both numbers live here and `scripts/prepublish-check.js` FAILS the
 * publish if package.json and MCP_SERVER_VERSION drift apart.
 *
 * When releasing: bump mini-services/mcp-server/package.json AND the constant
 * below in the same commit. The prepublish guard enforces it.
 */

/** npm package version this code corresponds to (@tokcalc/mcp-server). */
export const MCP_SERVER_VERSION = "0.3.0";

/** Model/GPU catalog version (bumped when src/lib/token-calc.ts catalog changes). */
export const CATALOG_VERSION = "0.3.0";
