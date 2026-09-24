# @tokcalc/mcp-server

**LLM serving capacity planner for AI agents.**

Open-source MCP (Model Context Protocol) server that lets AI agents (Cursor, Claude Desktop, Cline) estimate LLM serving capacity — model fit, KV cache, throughput, latency, multi-GPU topology, and cost.

## Tools

| Tool | What it does |
|---|---|
| `estimate_capacity` | VRAM/KV/throughput/latency/cost for one config |
| `compare_gpus` | Ranked GPU comparison for one workload |
| `recommend_topology` | TP/CP topology recommendation |
| `estimate_api_vs_self_host` | Break-even analysis |
| `list_models` | Discover supported model IDs (35 models) |
| `list_gpus` | Discover supported GPU IDs (30 GPUs) |

All tools are **read-only** — no side effects, no cloud credentials, no deployments.

## Install

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tokcalc": {
      "command": "npx",
      "args": ["-y", "@tokcalc/mcp-server"]
    }
  }
}
```

Restart Claude Desktop. The `tokcalc` server will be available as an MCP tool source.

### Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "tokcalc": {
      "command": "npx",
      "args": ["-y", "@tokcalc/mcp-server"]
    }
  }
}
```

### Cline (VS Code)

Add the same config to Cline's MCP settings.

## Example prompts

Ask your AI agent:

> "I need to serve Llama 3.3 70B at 32K context for 50 concurrent users. What GPU topology do you recommend, and how much will it cost per month?"

> "Compare H100 vs H200 for serving Qwen 2.5 72B in FP8 with continuous batching."

> "At what daily request volume does self-hosting Llama 70B on H200 beat the GPT-4o API?"

The agent calls `list_models` → `list_gpus` → `recommend_topology` → `estimate_capacity` and returns a structured plan with throughput ranges, latency, VRAM, cost, and confidence levels.

## Supported models (35)

Llama 3/3.1/3.3, Llama 4 Scout/Maverick, Mistral 7B, Mixtral 8x7B/8x22B, Mistral Large 3, Pixtral 12B, Codestral, Qwen 2/2.5/3 (incl. MoE + VL), DeepSeek V3/R1/Coder V2, Gemma 2, Phi-3/4, SmolLM2, Falcon 3, OLMo 2, BGE-M3, E5, GTE.

## Supported GPUs (30)

NVIDIA H100/H200/B200/B300, A100, L40S, L4, T4, V100, RTX 4090/3090/5090, RTX PRO 6000 Blackwell, AMD MI300X/MI325X, Intel Gaudi 3, Google TPU v5p/Trillium, Groq LPU, Cerebras CS-3, Apple M2/M3/M4 Ultra/Max.

## License

Apache 2.0 — same as the main tokcalc project.

## Links

- [Live calculator](https://tokcalc.vercel.app)
- [GitHub](https://github.com/stevecrates489-commits/tokcalc)
- [CONTRIBUTING](https://github.com/stevecrates489-commits/tokcalc/blob/main/CONTRIBUTING.md)
