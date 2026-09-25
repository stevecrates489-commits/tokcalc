"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Terminal, Box, Sparkles, BookOpen, Zap, ExternalLink } from "lucide-react";
import Link from "next/link";

const TOOLS = [
  { name: "estimate_capacity", description: "VRAM/KV/throughput/latency for one config. Returns feasibility, performance ranges, cost, confidence." },
  { name: "compare_gpus", description: "Ranked GPU comparison for one workload. Sort by lowest cost, highest throughput, or best value." },
  { name: "recommend_topology", description: "TP/CP topology recommendation. Returns feasible GPU + topology designs for a context length." },
  { name: "estimate_api_vs_self_host", description: "Break-even analysis. Compares monthly API costs vs self-hosted GPU infrastructure." },
  { name: "list_models", description: "Discover supported model IDs. Filter by family, category, MoE." },
  { name: "list_gpus", description: "Discover supported GPU IDs. Filter by vendor, category, min VRAM." },
  { name: "get_mlperf_benchmarks", description: "Curated MLPerf Inference v4.1 audited configs. Cross-validate estimates against reference systems." },
];

const EXAMPLE_PROMPTS = [
  "How many H100s do I need to serve Llama 3.3 70B at 32K context for 100 concurrent users?",
  "Compare H100 vs H200 for serving Llama 3.3 70B FP8 at 8K context — which is better value?",
  "What's the break-even request volume for self-hosting Llama 70B vs using GPT-4o API at 50% utilization?",
  "Show me MLPerf v4.1 audited configurations for H200 with FP8 quantization",
  "Recommend a topology for serving Llama 3.3 70B at 128K context with batch size 8",
];

export default function McpDocsPage() {
  const [copied, setCopied] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 1500);
  };

  const cursorConfig = `{
  "mcpServers": {
    "tokcalc": {
      "command": "npx",
      "args": ["-y", "@tokcalc/mcp-server"]
    }
  }
}`;

  const httpStartCommand = `# Install globally (or use npx)
npm install -g @tokcalc/mcp-server

# Run the HTTP server (defaults to port 3000)
tokcalc-mcp-http

# Or with a custom port:
PORT=8080 tokcalc-mcp-http`;

  const curlTest = `curl -X POST http://localhost:3000/mcp \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json, text/event-stream" \\
  -H "MCP-Protocol-Version: 2025-03-26" \\
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'`;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <Link href="/" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mb-6">
          <ArrowLeft className="size-4" />
          Back to tokcalc
        </Link>

        {/* Hero */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">tokcalc MCP Server</h1>
            <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30 text-[10px]">v0.2.0-alpha.1</Badge>
          </div>
          <p className="text-muted-foreground text-sm sm:text-base leading-relaxed">
            7 read-only planning tools for AI agents (Cursor, Claude Desktop, Cline). Ask your AI:
            &quot;How many H100s do I need for Llama 70B at 32K context?&quot; — get a transparent,
            formula-based answer with confidence intervals.
          </p>
        </div>

        {/* What you get */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Box className="size-4 text-emerald-500" />
              What you get — 7 read-only tools
            </CardTitle>
            <CardDescription className="text-xs">
              All tools are read-only (no side effects, no mutations). Safe for production agent use.
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead className="bg-muted/40 text-[10px] uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Tool</th>
                  <th className="text-left px-4 py-2">What it does</th>
                </tr>
              </thead>
              <tbody>
                {TOOLS.map((t) => (
                  <tr key={t.name} className="border-t border-border/40">
                    <td className="px-4 py-2 font-mono text-xs font-medium align-top whitespace-nowrap">{t.name}</td>
                    <td className="px-4 py-2 text-xs text-muted-foreground">{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        {/* Install — Cursor / Claude Desktop */}
        <Card className="border-emerald-500/30 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Terminal className="size-4 text-emerald-500" />
              Install in Cursor / Claude Desktop (stdio transport)
            </CardTitle>
            <CardDescription className="text-xs">
              Recommended for local development. No HTTP server to run — Cursor/Claude spawns the process.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Add this to your MCP config file:
            </p>
            <ul className="text-xs text-muted-foreground space-y-1 ml-4 list-disc">
              <li><strong className="text-foreground">Cursor:</strong> <code className="font-mono">~/.cursor/mcp.json</code> (global) or <code className="font-mono">.cursor/mcp.json</code> (project)</li>
              <li><strong className="text-foreground">Claude Desktop:</strong> <code className="font-mono">~/Library/Application Support/Claude/claude_desktop_config.json</code> (macOS) or <code className="font-mono">%APPDATA%\Claude\claude_desktop_config.json</code> (Windows)</li>
              <li><strong className="text-foreground">Cline:</strong> uses VS Code MCP settings</li>
            </ul>
            <div className="relative">
              <pre className="bg-muted rounded-md p-3 text-xs font-mono overflow-x-auto border border-border/60"><code>{cursorConfig}</code></pre>
              <Button
                size="sm"
                variant="ghost"
                className="absolute top-2 right-2 text-[10px]"
                onClick={() => copyToClipboard(cursorConfig, "cursor")}
              >
                {copied === "cursor" ? "✓ Copied" : "Copy"}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Restart your editor. The 7 tools will appear in your MCP tool list within seconds.
            </p>
          </CardContent>
        </Card>

        {/* Install — HTTP (alpha) */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="size-4 text-emerald-500" />
              Self-hosted HTTP transport (v0.2.0-alpha.1)
            </CardTitle>
            <CardDescription className="text-xs">
              For agent platforms that need HTTP transport (no local <code>npx</code>). Stateless mode —
              no sessions, no SSE stream, no auth in alpha.1.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="relative">
              <pre className="bg-muted rounded-md p-3 text-xs font-mono overflow-x-auto border border-border/60"><code>{httpStartCommand}</code></pre>
              <Button
                size="sm"
                variant="ghost"
                className="absolute top-2 right-2 text-[10px]"
                onClick={() => copyToClipboard(httpStartCommand, "http")}
              >
                {copied === "http" ? "✓ Copied" : "Copy"}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-3">
              Test with curl:
            </p>
            <div className="relative">
              <pre className="bg-muted rounded-md p-3 text-xs font-mono overflow-x-auto border border-border/60"><code>{curlTest}</code></pre>
              <Button
                size="sm"
                variant="ghost"
                className="absolute top-2 right-2 text-[10px]"
                onClick={() => copyToClipboard(curlTest, "curl")}
              >
                {copied === "curl" ? "✓ Copied" : "Copy"}
              </Button>
            </div>
            <div className="text-[11px] text-amber-600 dark:text-amber-400 flex items-start gap-1.5 mt-2">
              <span className="font-medium">⚠ Alpha scope:</span>
              <span>v0.2.0-alpha.1 has NO authentication or rate limiting. Do NOT expose to the public internet.
              Bearer API key + KV-backed rate limits arrive in v0.2.0-beta.1.</span>
            </div>
          </CardContent>
        </Card>

        {/* Example prompts */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Sparkles className="size-4 text-emerald-500" />
              Example prompts to ask your AI
            </CardTitle>
            <CardDescription className="text-xs">
              Once installed, your AI agent can call tokcalc tools. Try these natural-language prompts.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {EXAMPLE_PROMPTS.map((prompt, i) => (
              <div key={i} className="text-xs p-3 rounded-md bg-muted/40 border border-border/40 italic">
                &quot;{prompt}&quot;
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Roadmap */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <BookOpen className="size-4 text-emerald-500" />
              Roadmap
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-xs">
            <div className="flex items-start gap-3">
              <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30 text-[9px] shrink-0">✓ shipped</Badge>
              <div>
                <div className="font-medium">v0.1.4 — stdio transport with 7 tools</div>
                <div className="text-muted-foreground">Cursor / Claude Desktop / Cline local installs</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <Badge className="bg-amber-500/10 text-amber-600 border-amber-500/30 text-[9px] shrink-0">alpha</Badge>
              <div>
                <div className="font-medium">v0.2.0-alpha.1 — stateless HTTP transport</div>
                <div className="text-muted-foreground">For agent platforms that need HTTP. No auth yet — local testing only.</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <Badge variant="outline" className="text-[9px] shrink-0">next</Badge>
              <div>
                <div className="font-medium">v0.2.0-beta.1 — bearer API key + rate limiting</div>
                <div className="text-muted-foreground">Public deploy-safe. KV-backed rate limits per IP / per key.</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <Badge variant="outline" className="text-[9px] shrink-0">v0.2.0</Badge>
              <div>
                <div className="font-medium">v0.2.0 — public hosted endpoint</div>
                <div className="text-muted-foreground">mcp.tokcalc.app — no install needed for hosted clients</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <Badge variant="outline" className="text-[9px] shrink-0">v0.3.0</Badge>
              <div>
                <div className="font-medium">v0.3.0 — OAuth 2.1 with PKCE</div>
                <div className="text-muted-foreground">Multi-user auth, scopes, dynamic registration</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Resources */}
        <Card className="border-border/60 shadow-sm mb-8">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <ExternalLink className="size-4 text-emerald-500" />
              Resources
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <a href="https://www.npmjs.com/package/@tokcalc/mcp-server" target="_blank" rel="noopener noreferrer" className="block p-2 rounded-md hover:bg-muted/40 transition-colors">
              <span className="font-mono text-foreground">npmjs.com/package/@tokcalc/mcp-server</span>
              <span className="text-muted-foreground ml-2">— npm package</span>
            </a>
            <a href="https://github.com/stevecrates489-commits/tokcalc/tree/main/mini-services/mcp-server" target="_blank" rel="noopener noreferrer" className="block p-2 rounded-md hover:bg-muted/40 transition-colors">
              <span className="font-mono text-foreground">github.com/stevecrates489-commits/tokcalc</span>
              <span className="text-muted-foreground ml-2">— source code in mini-services/mcp-server/</span>
            </a>
            <a href="https://github.com/stevecrates489-commits/tokcalc/issues" target="_blank" rel="noopener noreferrer" className="block p-2 rounded-md hover:bg-muted/40 transition-colors">
              <span className="font-mono text-foreground">github.com/stevecrates489-commits/tokcalc/issues</span>
              <span className="text-muted-foreground ml-2">— bug reports, feature requests</span>
            </a>
            <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer" className="block p-2 rounded-md hover:bg-muted/40 transition-colors">
              <span className="font-mono text-foreground">modelcontextprotocol.io</span>
              <span className="text-muted-foreground ml-2">— MCP spec reference</span>
            </a>
          </CardContent>
        </Card>

        {/* Internal links */}
        <div className="border-t border-border/60 pt-6 mt-8">
          <p className="text-xs text-muted-foreground mb-3">Related:</p>
          <div className="flex flex-wrap gap-2">
            <Link href="/compare/h100-vs-h200" className="text-xs text-emerald-500 hover:underline">H100 vs H200 →</Link>
            <Link href="/compare/gguf-q4-k-m-vs-q5-k-m" className="text-xs text-emerald-500 hover:underline">GGUF Q4_K_M vs Q5_K_M →</Link>
            <Link href="/self-host-vs-openai-api" className="text-xs text-emerald-500 hover:underline">Self-host vs API →</Link>
            <Link href="/" className="text-xs text-emerald-500 hover:underline">Full calculator →</Link>
          </div>
        </div>

        {/* JSON-LD structured data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "SoftwareApplication",
              name: "tokcalc MCP Server",
              applicationCategory: "DeveloperApplication",
              operatingSystem: "Cross-platform (Node.js 18+)",
              offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
              description: "Open-source MCP server for AI agents with 7 read-only LLM capacity planning tools. Supports stdio and HTTP transports.",
              version: "0.2.0-alpha.1",
              license: "https://www.apache.org/licenses/LICENSE-2.0",
              url: "https://tokcalc.vercel.app/mcp",
              downloadUrl: "https://www.npmjs.com/package/@tokcalc/mcp-server",
            }),
          }}
        />
      </div>
    </div>
  );
}
