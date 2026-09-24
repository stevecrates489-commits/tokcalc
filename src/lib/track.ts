/**
 * Plausible Analytics custom event tracking.
 *
 * Usage:
 *   track("shared_scenario", { tab: "calculator" })
 *   track("switched_tab", { tab: "build-vs-buy" })
 *   track("toggled_feature", { feature: "continuous_batching", enabled: true })
 *
 * The window.plausible function is set up by the init script in layout.tsx.
 * If Plausible isn't loaded yet (e.g., during SSR or ad-blocker), calls are
 * silently no-ops — no errors thrown.
 *
 * Custom events show up in your Plausible dashboard under "Custom Events"
 * (may need to be enabled in Plausible site settings → "Custom events").
 */

type EventProps = Record<string, string | number | boolean | undefined>;

export function track(eventName: string, props?: EventProps): void {
  if (typeof window === "undefined") return;

  // Cast window to access the plausible function we set up via init script
  const w = window as Window & {
    plausible?: (eventName: string, options?: { props?: EventProps }) => void;
  };

  if (typeof w.plausible === "function") {
    w.plausible(eventName, props ? { props } : undefined);
  }
  // Silent no-op if Plausible isn't loaded (SSR, ad-blocker, dev mode)
}
