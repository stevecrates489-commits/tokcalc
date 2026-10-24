import type { MetadataRoute } from "next";
import { MODEL_PAGE_IDS } from "@/lib/content/model-content";
import { GPU_PAGE_IDS } from "@/lib/content/gpu-content";
import { modelGpuPairs } from "@/lib/content/model-content";

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://tokcalc.vercel.app";

/**
 * Dynamic sitemap.
 *
 * This replaces the hand-maintained public/sitemap.xml, which listed four URLs
 * while the site actually served dozens of pages. Because it is generated from
 * the catalog, a new model or GPU in token-calc.ts is automatically
 * discoverable the next time the site builds — no manual XML editing.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();

  const staticRoutes: MetadataRoute.Sitemap = [
    {
      url: SITE_URL,
      lastModified: now,
      changeFrequency: "weekly",
      priority: 1.0,
    },
    {
      url: `${SITE_URL}/models`,
      lastModified: now,
      changeFrequency: "weekly",
      priority: 0.9,
    },
    {
      url: `${SITE_URL}/gpus`,
      lastModified: now,
      changeFrequency: "weekly",
      priority: 0.9,
    },
    {
      url: `${SITE_URL}/mcp`,
      lastModified: now,
      changeFrequency: "monthly",
      priority: 0.8,
    },
    {
      url: `${SITE_URL}/compare/h100-vs-h200`,
      lastModified: now,
      changeFrequency: "monthly",
      priority: 0.9,
    },
    {
      url: `${SITE_URL}/compare/gguf-q4-k-m-vs-q5-k-m`,
      lastModified: now,
      changeFrequency: "monthly",
      priority: 0.8,
    },
    {
      url: `${SITE_URL}/self-host-vs-openai-api`,
      lastModified: now,
      changeFrequency: "monthly",
      priority: 0.8,
    },
  ];

  const modelRoutes: MetadataRoute.Sitemap = MODEL_PAGE_IDS.map((modelId) => ({
    url: `${SITE_URL}/models/${modelId}`,
    lastModified: now,
    changeFrequency: "weekly",
    priority: 0.8,
  }));

  const gpuRoutes: MetadataRoute.Sitemap = GPU_PAGE_IDS.map((gpuId) => ({
    url: `${SITE_URL}/gpus/${gpuId}`,
    lastModified: now,
    changeFrequency: "weekly",
    priority: 0.8,
  }));

  // Model-on-GPU pages answer the highest-intent query ("can I run X on Y"),
  // so they carry the highest priority of the generated set.
  const pairRoutes: MetadataRoute.Sitemap = modelGpuPairs().map(({ modelId, gpuId }) => ({
    url: `${SITE_URL}/models/${modelId}/on/${gpuId}`,
    lastModified: now,
    changeFrequency: "weekly",
    priority: 0.8,
  }));

  return [...staticRoutes, ...modelRoutes, ...gpuRoutes, ...pairRoutes];
}
