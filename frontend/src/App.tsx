import Cookies from "js-cookie";
import { AlertCircle, ChevronDown, Loader2, Search } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { AuthModal } from "@/components/auth-modal";
import { BookCover } from "@/components/book-cover";
import { LogoMark } from "@/components/logo-mark";
import { LibraryPanel, type LibraryDoc } from "@/components/library-panel";
import { parseAnswer } from "@/lib/answer-format";
import { API_BASE, errorMessage } from "@/lib/api";
import { bookInfo, compareBooksChronologically, formatSectionLabel } from "@/lib/books";
import { cn } from "@/lib/utils";

const EXAMPLE_QUESTIONS = [
  "What does it mean to live authentically?",
  "When should I act versus wait?",
  "Is it better to know the truth or to be happy?",
  "What can limits teach us?",
  "When should I trust my own judgment?",
  "How should I respond to things outside my control?",
];

// The books a user has unticked, remembered by filename (document ids change
// when the index is rebuilt). Storing exclusions means books indexed later
// start out selected.
const EXCLUDED_STORAGE_KEY = "philosearch:excluded-books";

function loadExcludedBooks(): string[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(EXCLUDED_STORAGE_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((f) => typeof f === "string") : [];
  } catch {
    return [];
  }
}

// Returned by the backend when a requested book id is no longer indexed
const STALE_SELECTION_DETAIL = "Some selected books are no longer indexed.";

const PHASE_LABELS = {
  searching: "Finding relevant passages…",
  thinking: "Reading sources…",
  generating: "Writing answer…",
} as const;

interface Source {
  filename: string;
  author: string | null;
  paragraph_idx: number;
  section_label?: string | null; // e.g. "Book 6, §49"
  excerpt?: string; // short citation: matched sentence + same-passage neighbours
  text: string; // full passage
  score: number | null;
}

// Longest excerpt shown before trimming. Only a single very long sentence
// (common in Nietzsche) exceeds it; it is cut at a clause break, never mid-word.
const EXCERPT_DISPLAY_LIMIT = 500;

function trimExcerpt(excerpt: string): string {
  if (excerpt.length <= EXCERPT_DISPLAY_LIMIT) return excerpt;
  const head = excerpt.slice(0, EXCERPT_DISPLAY_LIMIT);
  const clauseBreak = Math.max(
    head.lastIndexOf("; "),
    head.lastIndexOf(": "),
    head.lastIndexOf("—"),
    head.lastIndexOf(", "),
  );
  const cut = clauseBreak > 150 ? clauseBreak : head.lastIndexOf(" ");
  return `${head.slice(0, cut).trimEnd()} …`;
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => Cookies.get("creator_auth") === "true",
  );
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<
    "searching" | "thinking" | "generating"
  >("searching");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState("");
  const [sourcesExpanded, setSourcesExpanded] = useState(true);
  const [library, setLibrary] = useState<LibraryDoc[]>([]);
  const [libraryError, setLibraryError] = useState("");
  const [libraryLoaded, setLibraryLoaded] = useState(false);
  const [libraryRetrying, setLibraryRetrying] = useState(false);
  const [selectedBooks, setSelectedBooks] = useState<Set<string>>(new Set());
  const [libraryOpen, setLibraryOpen] = useState(false); // small screens only
  const [highlightedSource, setHighlightedSource] = useState<number | null>(
    null,
  );
  const [expandedPassages, setExpandedPassages] = useState<Set<number>>(
    new Set(),
  );

  const togglePassage = (idx: number) =>
    setExpandedPassages((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });

  const searchInputRef = useRef<HTMLInputElement>(null);
  const wasLoading = useRef(false);

  // The search box is disabled while answering; return focus to it afterwards
  // (without scrolling away from the answer) so a follow-up can be typed
  useEffect(() => {
    // Only when focus was actually lost (disabled input -> body), so a text
    // selection or a focused citation button is left alone
    const focusWasDropped =
      !document.activeElement || document.activeElement === document.body;
    if (wasLoading.current && !isLoading && focusWasDropped) {
      searchInputRef.current?.focus({ preventScroll: true });
    }
    wasLoading.current = isLoading;
  }, [isLoading]);

  // Load the list of indexed books for the library panel (public endpoint).
  // Also re-run when a search reports that the selected ids are stale.
  const loadLibrary = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE}/documents`);
      if (!response.ok) throw new Error(await errorMessage(response));
      const data: { documents: LibraryDoc[] } = await response.json();
      const docs = [...data.documents].sort((a, b) =>
        compareBooksChronologically(a.filename, b.filename),
      );
      const excluded = new Set(loadExcludedBooks());
      const kept = docs.filter((d) => !excluded.has(d.filename));
      setLibrary(docs);
      // Never start with nothing selected: that only blocks searching
      setSelectedBooks(new Set((kept.length ? kept : docs).map((d) => d.doc_id)));
      setLibraryError("");
      setLibraryLoaded(true);
      return true;
    } catch {
      // Drop any previously loaded list: searches then really cover all books
      // (no stale ids are sent), matching the message
      setLibrary([]);
      setSelectedBooks(new Set());
      setLibraryError("Couldn't load the book list. Searching all books.");
      return false;
    }
  }, []);

  useEffect(() => {
    void loadLibrary();
  }, [loadLibrary]);

  const updateSelection = (next: Set<string>) => {
    setSelectedBooks(next);
    try {
      localStorage.setItem(
        EXCLUDED_STORAGE_KEY,
        JSON.stringify(library.filter((d) => !next.has(d.doc_id)).map((d) => d.filename)),
      );
    } catch {
      // Storage unavailable (private mode); the selection still works this session
    }
  };

  const toggleBook = (docId: string) => {
    const next = new Set(selectedBooks);
    if (next.has(docId)) next.delete(docId);
    else next.add(docId);
    updateSelection(next);
  };

  const noBooksSelected = library.length > 0 && selectedBooks.size === 0;

  // The in-flight answer stream, so a new search or a reset can cancel it
  const streamController = useRef<AbortController | null>(null);

  useEffect(() => () => streamController.current?.abort(), []);

  // Clear highlight on click anywhere
  useEffect(() => {
    if (highlightedSource === null) return;

    const clearHighlight = () => setHighlightedSource(null);

    // Use capture phase to catch click before it bubbles
    window.addEventListener("click", clearHighlight, { capture: true });

    return () =>
      window.removeEventListener("click", clearHighlight, { capture: true });
  }, [highlightedSource]);

  const scrollToSource = useCallback(
    (sourceNum: number) => {
      // Validate source exists
      if (sourceNum < 1 || sourceNum > sources.length) return;

      // Expand sources if collapsed
      setSourcesExpanded(true);

      // Small delay to allow expansion
      setTimeout(() => {
        const element = document.getElementById(`source-${sourceNum}`);
        if (element) {
          setHighlightedSource(sourceNum);
          element.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 100);
    },
    [sources.length],
  );

  const handleAuthenticated = () => {
    setIsAuthenticated(true);
    setError(""); // e.g. a "session expired" message from the last search
  };

  // Render the model's Markdown-ish answer: paragraphs, "#" headings, "> "
  // quotes, and bullet / numbered lists (which may have blank lines between
  // items, start at any number, and contain indented sub-bullets)
  const formatAnswer = (text: string) => {
    return parseAnswer(text).map((block, i) => {
      const spacing = i > 0 ? "mt-4" : "";
      if (block.kind === "ordered" || block.kind === "unordered") {
        const items = block.items.map((item, j) => (
          <li key={j} className="leading-7">
            {formatInlineText(item.text)}
            {item.children.length > 0 &&
              (() => {
                const SubList = item.childrenOrdered ? "ol" : "ul";
                return (
                  <SubList
                    className={cn(
                      "list-outside pl-5 mt-1 space-y-1",
                      item.childrenOrdered ? "list-decimal" : "list-disc",
                    )}
                  >
                    {item.children.map((child, k) => (
                      <li key={k}>{formatInlineText(child)}</li>
                    ))}
                  </SubList>
                );
              })()}
          </li>
        ));
        const listClass = "list-outside pl-5 space-y-1.5 my-3 marker:text-gray-400";
        return block.kind === "ordered" ? (
          <ol key={i} start={block.start} className={cn(listClass, "list-decimal")}>
            {items}
          </ol>
        ) : (
          <ul key={i} className={cn(listClass, "list-disc")}>
            {items}
          </ul>
        );
      }
      if (block.kind === "rule") {
        return <hr key={i} className="my-5 border-gray-200" />;
      }
      if (block.kind === "heading") {
        return (
          <p key={i} className={cn(spacing, "font-semibold text-gray-900")}>
            {formatInlineText(block.text)}
          </p>
        );
      }
      if (block.kind === "quote") {
        return (
          <blockquote
            key={i}
            className={cn(spacing, "border-l-2 border-gray-300 pl-4 font-serif text-gray-600")}
          >
            {formatInlineText(block.text)}
          </blockquote>
        );
      }
      return (
        <p key={i} className={spacing}>
          {formatInlineText(block.text)}
        </p>
      );
    });
  };

  const formatInlineText = useCallback(
    (text: string) => {
      // Match **bold**, *italic*, and [n] citations; bold/italic content is
      // rendered recursively so citations inside them stay clickable
      const render = (segment: string, keyPrefix: string): React.ReactNode[] => {
        const parts: React.ReactNode[] = [];
        let lastIndex = 0;
        // Bold may contain *italic*, even at its end ("**In *Meditations***",
        // closed by the optional third star). Each star inside bold can match
        // only one way, so an unclosed "**" can't trigger exponential
        // backtracking that would freeze the page.
        const regex =
          /(\*\*(?=\S)(?:[^*\n]|\*(?!\*))+?\*\*\*?|\*[^*\n]+\*|\[\d+\])/g;
        let match;

        while ((match = regex.exec(segment)) !== null) {
          if (match.index > lastIndex) {
            parts.push(segment.substring(lastIndex, match.index));
          }

          const matchedText = match[0];
          const key = `${keyPrefix}${match.index}`;
          if (matchedText.startsWith("**")) {
            parts.push(
              <strong key={key} className="font-semibold text-gray-900">
                {render(matchedText.slice(2, -2), `${key}-`)}
              </strong>,
            );
          } else if (matchedText.startsWith("*")) {
            parts.push(<em key={key}>{render(matchedText.slice(1, -1), `${key}-`)}</em>);
          } else {
            // Citation marker - only clickable if that source exists
            const num = parseInt(matchedText.slice(1, -1));
            if (num >= 1 && num <= sources.length) {
              parts.push(
                <button
                  key={key}
                  type="button"
                  onClick={() => scrollToSource(num)}
                  aria-label={`Go to source ${num}`}
                  className="ml-0.5 px-1 rounded bg-blue-50 text-blue-700 text-[12px] font-medium align-[1px] hover:bg-blue-100 cursor-pointer"
                >
                  {num}
                </button>,
              );
            } else {
              parts.push(matchedText);
            }
          }

          lastIndex = regex.lastIndex;
        }

        if (lastIndex < segment.length) {
          parts.push(segment.substring(lastIndex));
        }
        return parts;
      };

      return render(text, "");
    },
    [sources.length, scrollToSource],
  );

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || noBooksSelected) return;

    streamController.current?.abort();
    const controller = new AbortController();
    streamController.current = controller;

    setIsLoading(true);
    setLoadingPhase("searching");
    setAnswer("");
    setSources([]);
    setExpandedPassages(new Set());
    setHighlightedSource(null);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/search-and-generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          top_k_context: 10,
          use_reranking: true,
          top_k_faiss: 30,
          context_window: 0, // neighbouring paragraphs are not used in answers
          temperature: 0.7,
          max_output_tokens: 8192,
          // Only the books ticked in the library (all books if the list didn't load)
          // (omitted when every book is selected, so no filter is applied)
          doc_ids:
            library.length && selectedBooks.size < library.length ? [...selectedBooks] : undefined,
        }),
        signal: controller.signal,
      });

      if (response.status === 401) {
        // Access cookie expired: show the access prompt again
        Cookies.remove("creator_auth");
        setIsAuthenticated(false);
        throw new Error("Your session expired. Please verify access again.");
      }
      if (!response.ok) {
        throw new Error(await errorMessage(response));
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let buffer = "";
      let finished = false; // saw a "done" or "error" event
      let failed = false; // saw an "error" event
      let receivedText = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));

            if (data.type === "sources") {
              setSources(data.data);
              setLoadingPhase("thinking");
            } else if (data.type === "token") {
              receivedText = true;
              setLoadingPhase("generating");
              setAnswer((prev) => prev + data.data);
            } else if (data.type === "done") {
              finished = true;
            } else if (data.type === "error") {
              finished = failed = true;
              setError(data.data.message);
            }
          }
        }
      }

      if (!finished) {
        setError("The answer was interrupted before it finished. Please try again.");
      } else if (!failed && !receivedText) {
        setError("No answer was generated for this question. Please try again.");
      }
    } catch (err) {
      if (controller.signal.aborted) return; // cancelled by reset or a new search
      const message = err instanceof Error ? err.message : "An error occurred";
      if (message === STALE_SELECTION_DETAIL) {
        // The index changed since the page loaded: reload the book list and
        // re-map the selection by filename
        const reloaded = await loadLibrary();
        if (controller.signal.aborted) return; // reset or a new search meanwhile
        setError(
          reloaded
            ? "The book list changed since this page loaded and has been refreshed. Please search again."
            : "The book list changed since this page loaded. Please reload the page.",
        );
      } else {
        setError(message);
      }
    } finally {
      if (streamController.current === controller) {
        streamController.current = null;
        setIsLoading(false);
      }
    }
  };

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  const handleReset = () => {
    streamController.current?.abort();
    streamController.current = null;
    setIsLoading(false);
    setQuery("");
    setAnswer("");
    setSources([]);
    setExpandedPassages(new Set());
    setError("");
    setSourcesExpanded(true);
    setHighlightedSource(null);
  };

  const citedBooks = new Set(sources.map((s) => bookInfo(s.filename).title)).size;

  return (
    <>
      {!isAuthenticated && <AuthModal onAuthenticated={handleAuthenticated} />}
      <div className="min-h-screen bg-[#f6f7f9] text-gray-800">
        {/* Top bar */}
        <header className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-3">
            <button
              type="button"
              onClick={handleReset}
              className="flex items-center gap-2 cursor-pointer"
              aria-label="PhiloSearch, start a new search"
            >
              <LogoMark />
              <span className="text-[17px] font-semibold text-gray-900 tracking-tight">
                PhiloSearch
              </span>
            </button>
            <span className="hidden sm:block h-5 w-px bg-gray-200" aria-hidden="true" />
            <p className="hidden sm:block text-sm text-gray-500 truncate">
              Semantic search and cited answers across classic philosophical texts
            </p>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)] items-start">
          {/* Library */}
          <aside className="lg:sticky lg:top-6">
            <LibraryPanel
              docs={library}
              selected={selectedBooks}
              onToggle={toggleBook}
              onSelectAll={() => updateSelection(new Set(library.map((d) => d.doc_id)))}
              onSelectNone={() => updateSelection(new Set())}
              loaded={libraryLoaded}
              loadError={libraryError}
              onRetry={async () => {
                setLibraryRetrying(true);
                await loadLibrary();
                setLibraryRetrying(false);
              }}
              retrying={libraryRetrying}
              disabled={isLoading}
              open={libraryOpen}
              onOpenChange={setLibraryOpen}
            />
          </aside>

          <main className="space-y-4 min-w-0">
            {/* Search */}
            <section className="bg-white border border-gray-200 rounded-lg p-4 sm:p-5">
              <form onSubmit={handleSearch} className="space-y-3">
                <label htmlFor="question" className="block text-sm font-medium text-gray-900">
                  Ask a question
                </label>
                <div className="flex flex-col sm:flex-row gap-2">
                  <div className="relative flex-1">
                    <Search
                      className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                      aria-hidden="true"
                    />
                    <input
                      ref={searchInputRef}
                      id="question"
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      maxLength={2000}
                      placeholder="e.g. Is suffering necessary for a meaningful life?"
                      className={cn(
                        "w-full h-10 pl-9 pr-3 rounded-md border border-gray-300 bg-white",
                        "text-sm text-gray-900 placeholder:text-gray-400",
                        "focus:outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-600/20",
                        "disabled:bg-gray-50 disabled:text-gray-500",
                      )}
                      disabled={isLoading}
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={isLoading || !query.trim() || noBooksSelected}
                    className={cn(
                      "h-10 px-5 rounded-md text-sm font-medium flex items-center justify-center gap-2 cursor-pointer",
                      "bg-blue-700 text-white hover:bg-blue-800",
                      "disabled:bg-gray-200 disabled:text-gray-500 disabled:cursor-not-allowed",
                    )}
                  >
                    Search
                  </button>
                </div>
                {noBooksSelected ? (
                  <p className="text-xs text-amber-700">Select at least one book in the library to search.</p>
                ) : (
                  library.length > 0 &&
                  selectedBooks.size < library.length && (
                    <p className="text-xs text-gray-500">
                      Searching {selectedBooks.size} of {library.length} books.
                    </p>
                  )
                )}
              </form>

              {!answer && !isLoading && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <p className="text-xs font-medium text-gray-500 mb-2">Example questions</p>
                  <div className="flex flex-wrap gap-2">
                    {EXAMPLE_QUESTIONS.map((question) => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => handleExampleClick(question)}
                        className="px-3 py-1.5 rounded-md border border-gray-200 bg-white text-xs text-gray-700 hover:border-gray-300 hover:bg-gray-50 cursor-pointer"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* Status */}
            {isLoading && !answer && (
              <div
                role="status"
                className="bg-white border border-gray-200 rounded-lg px-5 py-4 flex items-center gap-3 text-sm text-gray-600"
              >
                <Loader2 className="w-4 h-4 animate-spin text-blue-700" aria-hidden="true" />
                {PHASE_LABELS[loadingPhase]}
              </div>
            )}

            {/* Error */}
            {error && (
              <div
                role="alert"
                className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 flex items-start gap-2 text-sm text-red-800"
              >
                <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
                <p>{error}</p>
              </div>
            )}

            {/* Answer */}
            {answer && (
              <section aria-labelledby="answer-heading" className="bg-white border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between gap-2 px-5 py-3 border-b border-gray-200">
                  <h2 id="answer-heading" className="text-sm font-semibold text-gray-900">
                    Answer
                  </h2>
                  <span className="text-xs text-gray-500 flex items-center gap-2">
                    {isLoading && (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
                    )}
                    {sources.length > 0 &&
                      `${sources.length} sources from ${citedBooks} ${citedBooks === 1 ? "book" : "books"}`}
                  </span>
                </div>
                <div className="px-5 py-4 text-[15px] leading-7 text-gray-800">
                  {formatAnswer(answer)}
                </div>
              </section>
            )}

            {/* Sources */}
            {answer && sources.length > 0 && (
              <section aria-labelledby="sources-heading" className="bg-white border border-gray-200 rounded-lg">
                <h2 id="sources-heading" className="text-sm font-semibold text-gray-900">
                  <button
                    type="button"
                    onClick={() => setSourcesExpanded(!sourcesExpanded)}
                    aria-expanded={sourcesExpanded}
                    aria-controls="sources-list"
                    className="w-full flex items-center justify-between gap-2 px-5 py-3 cursor-pointer"
                  >
                    <span>
                      Sources <span className="font-normal text-gray-500">({sources.length})</span>
                    </span>
                    <ChevronDown
                      className={cn("w-4 h-4 text-gray-500 transition-transform", sourcesExpanded && "rotate-180")}
                      aria-hidden="true"
                    />
                  </button>
                </h2>
                {sourcesExpanded && (
                  <ol id="sources-list" className="border-t border-gray-200 divide-y divide-gray-100">
                    {sources.map((source, idx) => {
                      const book = bookInfo(source.filename, source.author);
                      const label = formatSectionLabel(source.section_label, source.filename);
                      const excerpt = trimExcerpt(source.excerpt ?? source.text);
                      const canExpand = excerpt !== source.text;
                      return (
                        <li
                          key={idx}
                          id={`source-${idx + 1}`}
                          className={cn(
                            "px-5 py-4 transition-colors duration-300 scroll-mt-6",
                            highlightedSource === idx + 1 ? "bg-blue-50" : "bg-white",
                          )}
                        >
                          <div className="flex items-start gap-3">
                            <span className="mt-0.5 w-6 h-6 shrink-0 rounded bg-gray-100 text-gray-700 text-xs font-medium flex items-center justify-center">
                              {idx + 1}
                            </span>
                            <div className="flex-1 min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                                <BookCover book={book} size="sm" />
                                <span className="text-sm font-medium text-gray-900 wrap-anywhere">
                                  {book.title}
                                </span>
                                {book.author && (
                                  <span className="text-sm text-gray-500">{book.author}</span>
                                )}
                                {label && (
                                  <span className="text-xs text-gray-600 bg-gray-100 rounded px-1.5 py-0.5 wrap-anywhere">
                                    {label}
                                  </span>
                                )}
                                {source.score !== null && (
                                  <span className="ml-auto text-xs text-gray-400 tabular-nums">
                                    Relevance {source.score.toFixed(4)}
                                  </span>
                                )}
                              </div>
                              <p className="font-serif text-[15px] leading-7 text-gray-700 whitespace-pre-line">
                                {expandedPassages.has(idx) ? source.text : excerpt}
                              </p>
                              {canExpand && (
                                <button
                                  type="button"
                                  onClick={() => togglePassage(idx)}
                                  className="text-xs font-medium text-blue-700 hover:underline cursor-pointer"
                                >
                                  {expandedPassages.has(idx) ? "Show excerpt only" : "Show full passage"}
                                </button>
                              )}
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                )}
              </section>
            )}

            {!answer && !isLoading && !error && (
              <p className="text-xs text-gray-500 px-1">
                Answers cite numbered passages from the selected books. Click a number in the answer to jump to its source.
              </p>
            )}
          </main>
        </div>
      </div>
    </>
  );
}
