import { cn } from "@/lib/utils";

// The PhiloSearch mark: a lowercase serif phi on the brand blue.
// Same drawing as public/favicon.svg.
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cn("w-7 h-7 shrink-0", className)}>
      <rect width="32" height="32" rx="6" fill="#1d4ed8" />
      <text
        x="16"
        y="22.5"
        textAnchor="middle"
        fontFamily="Georgia, 'Times New Roman', serif"
        fontSize="20"
        fill="#ffffff"
      >
        φ
      </text>
    </svg>
  );
}
