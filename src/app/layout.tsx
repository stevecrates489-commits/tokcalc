import type { Metadata, Viewport } from "next";
import Script from "next/script";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/components/theme-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

/**
 * Resolve the canonical site URL for OG / Twitter metadata.
 *
 * Priority:
 *   1. NEXT_PUBLIC_SITE_URL — explicit override (set on Vercel for custom domain)
 *   2. NEXT_PUBLIC_VERCEL_URL — auto-injected by Vercel for preview deployments
 *   3. Hardcoded fallback — works out-of-the-box on tokcalc.vercel.app
 */
function getSiteUrl(): string {
  if (process.env.NEXT_PUBLIC_SITE_URL) {
    return process.env.NEXT_PUBLIC_SITE_URL;
  }
  if (process.env.NEXT_PUBLIC_VERCEL_URL) {
    return `https://${process.env.NEXT_PUBLIC_VERCEL_URL}`;
  }
  return "https://tokcalc.vercel.app";
}

const SITE_URL = getSiteUrl();

/**
 * Plausible Analytics — privacy-friendly, no cookies, GDPR-compliant.
 * Script ID is tied to this specific Plausible account.
 * Init script enables custom event tracking via window.plausible().
 */
const PLAUSIBLE_SCRIPT_SRC = "https://plausible.io/js/pa-HD58q2wnqtv43X23yV9-U.js";
const PLAUSIBLE_INIT = `
window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};
plausible.init()
`;

/**
 * Sentry — error monitoring + performance tracing.
 * DSN is public (client-side SDK), safe to commit.
 *
 * Uses the "loader" pattern: a single inline script that dynamically creates
 * a <script> element for the Sentry CDN, then calls Sentry.init() in the
 * onload callback. This guarantees correct execution order (CDN loads
 * before init runs) without blocking page render.
 */
const SENTRY_DSN = "https://2991cb014b0e9ad6d24d0192cf4bfb36@o4512137181003776.ingest.us.sentry.io/4512137191882752";
const SENTRY_CDN = "https://browser.sentry-cdn.com/8.49.0/bundle.min.js";
const SENTRY_LOADER = `
(function(){
  var s = document.createElement('script');
  s.src = '${SENTRY_CDN}';
  s.crossOrigin = 'anonymous';
  s.onload = function() {
    if (typeof Sentry !== 'undefined') {
      Sentry.init({
        dsn: '${SENTRY_DSN}',
        tracesSampleRate: 0.1,
        environment: '${process.env.NODE_ENV || "development"}',
        release: 'tokcalc@0.3.0',
        beforeSend: function(event) {
          if (event.environment === 'development') return null;
          return event;
        }
      });
    }
  };
  document.head.appendChild(s);
})();
`;

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "tokcalc — LLM serving capacity planner",
    template: "%s · tokcalc",
  },
  description:
    "Plan your LLM deployment before you rent the GPUs. Estimate model fit, KV-cache, prefill/decode throughput, continuous batching, latency, multi-GPU scaling, cloud cost, API cost, and self-hosting break-even — with transparent formulas and cited benchmarks.",
  keywords: [
    "LLM",
    "tokens per second",
    "inference",
    "throughput",
    "GPU",
    "A100",
    "H100",
    "H200",
    "MI300X",
    "capacity planner",
    "vLLM",
    "quantization",
    "GGUF",
    "FP8",
    "build vs buy",
    "self-host vs API",
    "KV cache",
    "TTFT",
  ],
  authors: [{ name: "tokcalc", url: SITE_URL }],
  creator: "tokcalc",
  applicationName: "tokcalc",
  publisher: "tokcalc",
  icons: {
    icon: [
      { url: "/og-icon-256.png", sizes: "256x256", type: "image/png" },
      { url: "/og-icon-256.png", sizes: "192x192", type: "image/png" },
      { url: "/og-icon-256.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/og-icon-256.png", sizes: "180x180", type: "image/png" }],
    shortcut: ["/og-icon-256.png"],
  },
  manifest: "/manifest.json",
  openGraph: {
    title: "tokcalc — LLM serving capacity planner",
    description:
      "Plan your LLM deployment before you rent the GPUs. Estimate throughput, latency, KV-cache, batching, multi-GPU scaling, and cost — with transparent formulas.",
    url: SITE_URL,
    siteName: "tokcalc",
    type: "website",
    locale: "en_US",
    images: [
      {
        url: "/og.png",
        secure_url: "/og.png",
        width: 2400,
        height: 1260,
        alt: "tokcalc — Plan your LLM deployment before you rent the GPUs. Sample: Llama 3 70B · 2×H200 · FP8 · 128K context · 47 tok/s · 1.4s TTFT · $0.34/M tokens.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "tokcalc — LLM serving capacity planner",
    description:
      "Plan your LLM deployment before you rent the GPUs. Transparent formulas, cited benchmarks, no black-box throughput assumptions.",
    images: [
      {
        url: "/og.png",
        width: 2400,
        height: 1260,
        alt: "tokcalc — Plan your LLM deployment before you rent the GPUs. Sample: Llama 3 70B · 2×H200 · FP8 · 128K context · 47 tok/s · 1.4s TTFT · $0.34/M tokens.",
      },
    ],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  alternates: {
    canonical: SITE_URL,
  },
  category: "technology",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0a0e0a" },
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
  ],
  colorScheme: "dark light",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

/**
 * JSON-LD structured data for Google rich results.
 * Type: SoftwareApplication — appropriate for a devtool web app.
 */
const jsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "tokcalc",
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Web",
  description:
    "Open-source LLM serving capacity planner. Estimate model fit, KV-cache, throughput, latency, multi-GPU scaling, and cost — with transparent formulas.",
  url: SITE_URL,
  offers: {
    "@type": "Offer",
    price: "0",
    priceCurrency: "USD",
  },
  license: "https://www.apache.org/licenses/LICENSE-2.0",
  isAccessibleForFree: true,
  author: {
    "@type": "Organization",
    name: "tokcalc",
    url: SITE_URL,
  },
  featureList: [
    "Model fit / VRAM calculation",
    "KV-cache by context and concurrency",
    "Prefill vs decode separation (TTFT / ITL)",
    "Continuous batching / paged attention",
    "Multi-GPU topology recommendation",
    "Prompt caching economics (Anthropic / OpenAI)",
    "Reasoning tokens (o1 / R1 / Claude thinking)",
    "Build-vs-buy break-even calculator",
    "Long-context capacity planner (4K → 1M)",
    "Transparent formulas with cited benchmarks",
  ],
  softwareVersion: "0.3.0",
  datePublished: "2026-09-20",
  discussionUrl: "https://github.com/stevecrates489-commits/tokcalc",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* JSON-LD structured data for Google rich results */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {/* Plausible Analytics — privacy-friendly, no cookies */}
        {/* Custom events: window.plausible('event_name', { props: {...} }) */}
        <Script
          src={PLAUSIBLE_SCRIPT_SRC}
          strategy="afterInteractive"
          async
        />
        <Script
          id="plausible-init"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{ __html: PLAUSIBLE_INIT }}
        />

        {/* Sentry — error monitoring via loader pattern (CDN + init in one script, onload guarantees order) */}
        <Script
          id="sentry-loader"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{ __html: SENTRY_LOADER }}
        />

        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          {children}
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
