import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "H100 vs H200 for LLM Inference: Throughput, Latency, VRAM & Cost",
  description:
    "Side-by-side comparison of NVIDIA H100 SXM5 80GB vs H200 SXM5 141GB for Llama 3.3 70B FP8 serving. Interactive throughput, latency, VRAM, and cost calculator with transparent formulas.",
  keywords: [
    "H100 vs H200",
    "H100 H200 comparison",
    "LLM inference GPU comparison",
    "H100 LLM throughput",
    "H200 LLM throughput",
    "H100 vs H200 VRAM",
    "Llama 70B GPU comparison",
    "NVIDIA H100 H200",
  ],
  openGraph: {
    title: "H100 vs H200 for LLM Inference — Throughput, VRAM & Cost",
    description:
      "Interactive comparison of H100 80GB vs H200 141GB for Llama 3.3 70B serving. See how context length affects capacity.",
    type: "article",
  },
  alternates: {
    canonical: "https://tokcalc.vercel.app/compare/h100-vs-h200",
  },
};

export default function Page({ children }: { children: React.ReactNode }) {
  return children;
}
