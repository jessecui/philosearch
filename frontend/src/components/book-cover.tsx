import type { BookInfo } from "@/lib/books";
import { cn } from "@/lib/utils";

const CONNECTING_WORDS = new Set(["a", "an", "and", "the", "of", "to", "in", "on", "for", "by", "with"]);

// A small generated cover: the book's colour, its initials and a rule.
// Uses `coverImage` instead when one is configured in lib/books.ts.
export function BookCover({
  book,
  size = "md",
  className,
}: {
  book: BookInfo;
  size?: "sm" | "md";
  className?: string;
}) {
  const dimensions = size === "sm" ? "w-6 h-8" : "w-10 h-14";

  if (book.coverImage) {
    return (
      <img
        src={book.coverImage}
        alt=""
        className={cn(dimensions, "rounded-sm object-cover shrink-0 shadow-sm", className)}
      />
    );
  }

  // First letters of the first two words (any script), skipping a leading
  // "The"; falls back to the title's first letter or digit, then "?"
  const allWords = book.title.replace(/^The\s+/i, "").split(/[\s_\-.]+/);
  // Skip connecting words ("The Myth of Sisyphus" -> "MS", not "MO")
  const mainWords = allWords.filter((w) => !CONNECTING_WORDS.has(w.toLowerCase()));
  const words = mainWords.length ? mainWords : allWords;
  const letters = words
    .map((w) => w.match(/\p{L}/u)?.[0])
    .filter((c): c is string => Boolean(c))
    .slice(0, 2);
  const initials = (
    letters.join("") ||
    book.title.match(/[\p{L}\p{N}]/u)?.[0] ||
    "?"
  ).toLocaleUpperCase();

  return (
    <div
      aria-hidden="true"
      className={cn(
        dimensions,
        "rounded-sm shrink-0 flex flex-col items-center justify-center gap-0.5 shadow-sm",
        "text-white/90 font-serif select-none",
        className,
      )}
      style={{ backgroundColor: book.coverColor }}
    >
      <span className={size === "sm" ? "text-[9px] leading-none" : "text-[13px] leading-none tracking-wide"}>
        {initials}
      </span>
      {size === "md" && <span className="block w-4 h-px bg-white/50" />}
    </div>
  );
}
