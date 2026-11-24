import { useCallback, useEffect, useState } from "react";

// Light or dark. With no saved choice the app follows the system setting (and
// keeps following it if that changes); the header toggle saves an explicit
// choice. index.html applies the same rule before first paint.
export const THEME_STORAGE_KEY = "philosearch:theme";

export type Theme = "light" | "dark";

const systemQuery = () => window.matchMedia("(prefers-color-scheme: dark)");

function savedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null; // storage unavailable (private mode)
  }
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.classList.contains("dark") ? "dark" : "light",
  );

  // Follow system changes until the user picks a theme themselves
  useEffect(() => {
    const query = systemQuery();
    const onChange = () => {
      if (savedTheme()) return;
      const next: Theme = query.matches ? "dark" : "light";
      applyTheme(next);
      setTheme(next);
    };
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((current) => {
      const next: Theme = current === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        // Choosing what the system already uses clears the override, so the
        // app goes back to following the system
        const system: Theme = systemQuery().matches ? "dark" : "light";
        if (next === system) localStorage.removeItem(THEME_STORAGE_KEY);
        else localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        // Not saved; the choice still applies this visit
      }
      return next;
    });
  }, []);

  return { theme, toggleTheme };
}
