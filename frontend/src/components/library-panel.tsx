import { Check, ChevronDown, Library } from "lucide-react";
import { useEffect, useRef } from "react";

import { BookCover } from "@/components/book-cover";
import { bookInfo } from "@/lib/books";
import { cn } from "@/lib/utils";

export interface LibraryDoc {
  doc_id: string;
  filename: string;
  author: string | null;
  total_paragraphs: number;
}

interface LibraryPanelProps {
  docs: LibraryDoc[];
  selected: Set<string>;
  onToggle: (docId: string) => void;
  onSelectAll: () => void;
  onSelectNone: () => void;
  loaded: boolean;
  loadError: string;
  onRetry: () => void;
  retrying: boolean;
  disabled: boolean;
  open: boolean; // small screens only; always open from lg up
  onOpenChange: (open: boolean) => void;
}

export function LibraryPanel({
  docs,
  selected,
  onToggle,
  onSelectAll,
  onSelectNone,
  loaded,
  loadError,
  onRetry,
  retrying,
  disabled,
  open,
  onOpenChange,
}: LibraryPanelProps) {
  const summary = docs.length ? `${selected.size} of ${docs.length} selected` : "";

  // When a retry succeeds, the focused Retry button disappears; move focus to
  // the Library heading so keyboard users don't land back at the page top
  const headingRef = useRef<HTMLHeadingElement>(null);
  const previousError = useRef(loadError);
  useEffect(() => {
    if (previousError.current && !loadError) headingRef.current?.focus();
    previousError.current = loadError;
  }, [loadError]);

  return (
    <section
      aria-labelledby="library-heading"
      className="bg-white border border-gray-200 rounded-lg lg:max-h-[calc(100vh-3rem)] lg:flex lg:flex-col"
    >
      <div className="flex items-center justify-between gap-2 px-4 py-3 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <h2
            id="library-heading"
            ref={headingRef}
            tabIndex={-1}
            className="flex items-center gap-2 text-sm font-semibold text-gray-900 focus:outline-none"
          >
            <Library className="w-4 h-4 text-gray-500" aria-hidden="true" />
            Library
            <span className="text-xs font-normal text-gray-500">{summary}</span>
          </h2>
          {/* Collapse control for small screens; the list is always shown from lg up */}
          <button
            type="button"
            onClick={() => onOpenChange(!open)}
            aria-expanded={open}
            aria-controls="library-list"
            aria-labelledby="library-heading"
            className="lg:hidden p-1 -m-1 rounded text-gray-400 hover:text-gray-600 cursor-pointer"
          >
            <ChevronDown
              className={cn("w-4 h-4 transition-transform", open && "rotate-180")}
              aria-hidden="true"
            />
          </button>
        </div>
        {docs.length > 0 && (
          <div className={cn("items-center gap-1 text-xs", open ? "flex" : "hidden lg:flex")}>
            <button
              type="button"
              onClick={onSelectAll}
              disabled={disabled}
              className="px-2 py-1 rounded text-blue-700 hover:bg-blue-50 disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
            >
              All
            </button>
            <button
              type="button"
              onClick={onSelectNone}
              disabled={disabled}
              className="px-2 py-1 rounded text-blue-700 hover:bg-blue-50 disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
            >
              None
            </button>
          </div>
        )}
      </div>

      {/* Outside the collapsible list so it's visible on mobile too */}
      {loadError && (
        <div role="alert" className="flex items-start justify-between gap-2 px-4 py-3 border-b border-gray-200">
          <p className="text-sm text-red-700">{loadError}</p>
          <button
            type="button"
            onClick={onRetry}
            disabled={retrying}
            className="shrink-0 px-2 py-1 -my-1 rounded text-xs font-medium text-blue-700 hover:bg-blue-50 disabled:text-gray-400 disabled:hover:bg-transparent cursor-pointer disabled:cursor-wait"
          >
            {retrying ? "Retrying…" : "Retry"}
          </button>
        </div>
      )}

      <div
        id="library-list"
        className={cn("p-2 lg:overflow-y-auto lg:min-h-0", open ? "block" : "hidden lg:block")}
      >
        {!loadError && !loaded && <p className="px-2 py-3 text-sm text-gray-500">Loading books…</p>}
        {!loadError && loaded && docs.length === 0 && (
          <p className="px-2 py-3 text-sm text-gray-500">
            No books are indexed yet. Run <code className="text-xs">python -m app.scripts.index_texts</code> in
            backend/.
          </p>
        )}
        <ul className="space-y-1">
          {docs.map((doc) => {
            const book = bookInfo(doc.filename, doc.author);
            const isSelected = selected.has(doc.doc_id);
            return (
              <li key={doc.doc_id}>
                <button
                  type="button"
                  role="checkbox"
                  aria-checked={isSelected}
                  onClick={() => onToggle(doc.doc_id)}
                  disabled={disabled}
                  className={cn(
                    "w-full flex items-center gap-3 p-2 rounded-md border text-left transition-colors cursor-pointer",
                    "disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-blue-600",
                    isSelected
                      ? "border-blue-200 bg-blue-50/60 hover:bg-blue-50"
                      : "border-transparent hover:bg-gray-50",
                  )}
                >
                  <BookCover book={book} className={cn(!isSelected && "opacity-40 grayscale")} />
                  <span className="flex-1 min-w-0">
                    <span
                      className={cn(
                        "block text-sm font-medium truncate",
                        isSelected ? "text-gray-900" : "text-gray-500",
                      )}
                    >
                      {book.title}
                    </span>
                    <span className="block text-xs text-gray-500 truncate">
                      {book.author ?? "Unknown author"}
                      {book.date && ` · ${book.date}`}
                    </span>
                    <span className="block text-[11px] text-gray-400 truncate">
                      {book.tradition ? `${book.tradition} · ` : ""}
                      {doc.total_paragraphs.toLocaleString()} passages
                    </span>
                  </span>
                  <span
                    aria-hidden="true"
                    className={cn(
                      "w-4 h-4 rounded border flex items-center justify-center shrink-0",
                      isSelected ? "bg-blue-700 border-blue-700" : "bg-white border-gray-300",
                    )}
                  >
                    {isSelected && <Check className="w-3 h-3 text-white" strokeWidth={3} />}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
