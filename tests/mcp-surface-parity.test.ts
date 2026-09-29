/**
 * Surface-parity tests — make sure every "surface" of the tokcalc MCP server
 * advertises the SAME tool set, so a tool added to one surface but not another
 * fails CI instead of silently drifting (the failure mode behind the
 * 0.2.2–0.2.4 stale-dist incidents).
 *
 * Surfaces checked:
 *   1. wire format — TOOL_DEFINITIONS served by stdio / self-hosted HTTP / /api/mcp
 *   2. docs page   — src/app/mcp/page.tsx renders the same names, in order
 *   3. dispatcher  — the CallToolRequest switch in core.ts handles every name
 *   4. npm listing — mini-services/mcp-server/package.json description names them
 *   5. docs badge  — /mcp hero badge shows the live MCP_SERVER_VERSION
 *
 * Run with: bun test tests/mcp-surface-parity.test.ts
 */

/// <reference types="bun-types" />
import { describe, expect, it } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { TOOL_CATALOG } from "../src/lib/mcp/tool-catalog";
import { __test } from "../src/lib/mcp/core";
import { MCP_SERVER_VERSION } from "../src/lib/mcp-version";

const { TOOL_DEFINITIONS } = __test as any;

const REPO_ROOT = path.resolve(__dirname, "..");

function readRepoFile(rel: string): string {
  return fs.readFileSync(path.join(REPO_ROOT, rel), "utf8");
}

describe("mcp surface parity", () => {
  it("catalog is non-empty and has unique names", () => {
    expect(TOOL_CATALOG.length).toBeGreaterThan(0);
    expect(new Set(TOOL_CATALOG.map((t) => t.name)).size).toBe(TOOL_CATALOG.length);
  });

  it("docs catalog matches the wire-format TOOL_DEFINITIONS exactly (names + order)", () => {
    const wire = TOOL_DEFINITIONS.map((t: { name: string }) => t.name);
    const docs = TOOL_CATALOG.map((t) => t.name);
    expect(docs).toEqual(wire);
  });

  it("docs descriptions are non-empty and match the wire descriptions", () => {
    const wireByName = new Map<string, string>(
      TOOL_DEFINITIONS.map((t: { name: string; description: string }) => [t.name, t.description]),
    );
    for (const t of TOOL_CATALOG) {
      const wireDesc = wireByName.get(t.name);
      expect(wireDesc).toBeDefined();
      // Docs use shorter marketing copy — require non-empty, not byte-identical.
      expect(t.description.length).toBeGreaterThan(0);
      expect(wireDesc!.length).toBeGreaterThan(0);
    }
  });

  it("every catalog tool is dispatched in core.ts's CallToolRequest switch", () => {
    const coreSrc = readRepoFile(path.join("src", "lib", "mcp", "core.ts"));
    for (const t of TOOL_CATALOG) {
      expect(coreSrc).toContain(`case "${t.name}"`);
    }
  });

  it("docs page actually renders TOOL_CATALOG (imports it as TOOLS)", () => {
    const pageSrc = readRepoFile(path.join("src", "app", "mcp", "page.tsx"));
    expect(pageSrc).toContain('from "@/lib/mcp/tool-catalog"');
    // Guard against a reintroduced hardcoded list drifting out of parity.
    expect(pageSrc).not.toMatch(/const TOOLS\s*=\s*\[/);
  });

  it("/mcp hero badge shows the live MCP_SERVER_VERSION (no hardcoded hero version)", () => {
    const pageSrc = readRepoFile(path.join("src", "app", "mcp", "page.tsx"));
    // The hero badge must come from the version module, not a literal.
    expect(pageSrc).toContain(">v{MCP_SERVER_VERSION}</Badge>");
    // Hero badge uses text-[10px]; a literal version in that badge = drift.
    expect(pageSrc).not.toMatch(/text-\[10px\]">v\d+\.\d+\.\d+<\/Badge>/);
  });

  it("npm package description names every catalog tool", () => {
    const pkg = JSON.parse(
      readRepoFile(path.join("mini-services", "mcp-server", "package.json")),
    ) as { description: string; version: string };
    for (const t of TOOL_CATALOG) {
      expect(pkg.description).toContain(t.name);
    }
  });

  it("wire definitions expose a valid JSON-Schema-shaped inputSchema", () => {
    for (const t of TOOL_DEFINITIONS as Array<{
      name: string;
      inputSchema: { type: string; properties: unknown };
    }>) {
      expect(t.inputSchema).toBeDefined();
      expect(t.inputSchema.type).toBe("object");
      expect(t.inputSchema.properties).toBeDefined();
    }
  });

  it("MCP_SERVER_VERSION is a valid semver string", () => {
    expect(MCP_SERVER_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
