import Link from "next/link";
import { ArrowLeft } from "lucide-react";

/**
 * Shared chrome for the programmatic content pages (per-model, per-GPU,
 * model-on-GPU). Keeping it in one component means every generated page has
 * the same navigation, breadcrumbs, and footer — and a change lands
 * everywhere at once instead of in 40 hand-written files.
 */

export function ContentShell({
  children,
  breadcrumb,
  eyebrow,
}: {
  children: React.ReactNode;
  /** Path segments above the current page, e.g. [{label:"Models", href:"/models"}] */
  breadcrumb?: { label: string; href: string }[];
  /** Small label above the H1, e.g. "Model" or "Comparison". */
  eyebrow?: string;
}) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <nav className="flex items-center gap-1.5 text-sm text-muted-foreground mb-6 flex-wrap">
          <Link href="/" className="inline-flex items-center gap-1.5 hover:text-foreground">
            <ArrowLeft className="size-4" />
            tokcalc
          </Link>
          {breadcrumb?.map((crumb) => (
            <span key={crumb.href} className="inline-flex items-center gap-1.5">
              <span className="text-border">/</span>
              <Link href={crumb.href} className="hover:text-foreground">
                {crumb.label}
              </Link>
            </span>
          ))}
        </nav>
        {children}
      </div>
    </div>
  );
}

/** Footer block shared by all content pages. */
export function ContentFooter() {
  return (
    <footer className="mt-12 pt-6 border-t border-border/40 text-xs text-muted-foreground space-y-2">
      <p>
        Every number on this page is computed by tokcalc&apos;s open-source
        capacity model — the same formulas that run the{" "}
        <Link href="/" className="underline underline-offset-2 hover:text-foreground">
          calculator
        </Link>
        , the MCP server, and the CLI. No marketing multipliers, no hidden
        assumptions.
      </p>
      <p>
        <Link href="/models" className="underline underline-offset-2 hover:text-foreground">
          All models
        </Link>
        {" · "}
        <Link href="/gpus" className="underline underline-offset-2 hover:text-foreground">
          All GPUs
        </Link>
        {" · "}
        <Link href="/mcp" className="underline underline-offset-2 hover:text-foreground">
          MCP server
        </Link>
        {" · "}
        <a
          href="https://github.com/stevecrates489-commits/tokcalc"
          target="_blank"
          rel="noopener noreferrer"
          className="underline underline-offset-2 hover:text-foreground"
        >
          Source
        </a>
      </p>
    </footer>
  );
}

/**
 * JSON-LD FAQPage block.
 *
 * Rendered server-side so the structured data is in the initial HTML payload —
 * Google reads it without executing anything.
 */
export function FaqJsonLd({ faqs }: { faqs: { q: string; a: string }[] }) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqs.map((f) => ({
      "@type": "Question",
      name: f.q,
      acceptedAnswer: { "@type": "Answer", text: f.a },
    })),
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}
