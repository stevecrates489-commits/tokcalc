"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

/**
 * Theme toggle button.
 *
 * Renders a STABLE button (same aria-label, same icon) on server + first client
 * render. Only AFTER `useEffect` runs (post-hydration) do we swap in the actual
 * sun/moon icon. This avoids hydration mismatches caused by `resolvedTheme`
 * being undefined on the server but defined on the client.
 *
 * Refs: https://github.com/pacocoursey/next-themes#avoid-hydration-mismatches
 */
export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  // eslint-disable-next-line react-hooks/set-state-in-effect
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  const isDark = resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      // STABLE label — never depends on theme. This is the hydration-safe choice.
      aria-label="Toggle theme"
      className="size-8 rounded-md"
      // suppressHydrationWarning is the recommended escape hatch for theme toggles
      // that swap attributes post-mount.
      suppressHydrationWarning
    >
      {mounted ? (
        isDark ? (
          <Sun className="size-4" />
        ) : (
          <Moon className="size-4" />
        )
      ) : (
        // Stable placeholder used during SSR + first client render.
        // Must match exactly between server and client to avoid hydration warnings.
        <Sun className="size-4" />
      )}
    </Button>
  );
}
