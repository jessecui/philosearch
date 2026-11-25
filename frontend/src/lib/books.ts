// Display metadata for the books in backend/texts, keyed by filename stem
// before the "__" (e.g. "Meditations" for Meditations__Marcus_Aurelius.txt).
// Documents not listed here (e.g. uploads) fall back to their filename.

export interface BookInfo {
  title: string;
  author: string | null;
  date?: string; // when it was written, as displayed
  year?: number; // for sorting the library oldest first (negative = BC)
  tradition?: string;
  coverColor: string; // background of the generated cover
  coverImage?: string; // optional real cover, e.g. "/covers/meditations.jpg" in frontend/public
  initials?: string; // cover initials when the title's first letters read badly
}

const BOOKS: Record<string, Omit<BookInfo, "author"> & { author: string }> = {
  Beyond_Good_and_Evil: {
    title: "Beyond Good and Evil",
    author: "Friedrich Nietzsche",
    date: "1886",
    year: 1886,
    tradition: "Existentialism",
    coverColor: "#6b2d2d",
  },
  Letters_from_a_Stoic: {
    title: "Letters from a Stoic",
    author: "Seneca",
    date: "c. 65",
    year: 65,
    tradition: "Stoicism",
    coverColor: "#33597a", // Stoics share a blue family
  },
  Nicomachean_Ethics: {
    title: "Nicomachean Ethics",
    author: "Aristotle",
    date: "c. 340 BC",
    year: -340,
    tradition: "Virtue ethics",
    coverColor: "#2c5a57",
  },
  Meditations: {
    title: "Meditations",
    author: "Marcus Aurelius",
    date: "c. 170–180",
    year: 175,
    tradition: "Stoicism",
    coverColor: "#243b5c", // Stoics share a blue family
  },
  Self_Reliance: {
    title: "Self-Reliance",
    author: "Ralph Waldo Emerson",
    date: "1841",
    year: 1841,
    tradition: "Transcendentalism",
    coverColor: "#3f5a3a",
  },
  Tao_Te_Ching: {
    title: "Tao Te Ching",
    author: "Lao Tzu",
    date: "c. 400 BC",
    year: -400,
    tradition: "Taoism",
    coverColor: "#7a5a24",
    initials: "TC", // "Tao Ching" rather than a stuttering "TT"
  },
  The_Enchiridion: {
    title: "The Enchiridion",
    author: "Epictetus",
    date: "c. 125",
    year: 125,
    tradition: "Stoicism",
    coverColor: "#3d437a", // Stoics share a blue family
  },
  The_Myth_of_Sisyphus: {
    title: "The Myth of Sisyphus",
    author: "Albert Camus",
    date: "1942",
    year: 1942,
    tradition: "Absurdism",
    coverColor: "#4f3b63",
  },
};

const FALLBACK_COLORS = ["#3d4a5c", "#27415f", "#4f3b63", "#3f5a3a", "#6b2d2d", "#7a5a24"];

function bookStem(filename: string): string {
  return filename.replace(/\.[^.]+$/, "").split("__")[0];
}

export function bookInfo(filename: string, author?: string | null): BookInfo {
  const stem = bookStem(filename);
  // Own keys only: a stem like "toString" must not match Object.prototype
  const known = Object.hasOwn(BOOKS, stem) ? BOOKS[stem] : undefined;
  if (known) return known;
  const title = stem.replace(/_/g, " ").trim() || filename;
  const hash = [...title].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7);
  return {
    title,
    author: author ?? null,
    coverColor: FALLBACK_COLORS[hash % FALLBACK_COLORS.length],
  };
}

// Library order: oldest first; books without a known year (e.g. uploads)
// go last, alphabetically
export function compareBooksChronologically(aFilename: string, bFilename: string): number {
  const a = bookInfo(aFilename);
  const b = bookInfo(bFilename);
  if (a.year !== undefined && b.year !== undefined && a.year !== b.year) return a.year - b.year;
  if (a.year !== undefined && b.year === undefined) return -1;
  if (a.year === undefined && b.year !== undefined) return 1;
  return a.title.localeCompare(b.title);
}

// Section labels come from each book's own structure: "Book 6, §49",
// "Chapter 8", "Chapter I, §1", "§52". Some books title their divisions
// instead of numbering them; name what each title is, per book.
const TITLED_DIVISIONS: Record<string, Record<string, string>> = {
  The_Myth_of_Sisyphus: {
    "Absurdity and Suicide": "Chapter",
    "Absurd Walls": "Chapter",
    "Philosophical Suicide": "Chapter",
    "Absurd Freedom": "Chapter",
    "The Absurd Man": "Part",
    "Don Juanism": "Chapter",
    Drama: "Chapter",
    Conquest: "Chapter",
    "Philosophy and Fiction": "Chapter",
    Kirilov: "Chapter",
    "Ephemeral Creation": "Chapter",
    "The Myth of Sisyphus": "Part",
  },
  Beyond_Good_and_Evil: {
    "From the Heights": "Aftersong", // Nietzsche's closing poem
  },
};

// Books whose numbered sections are conventionally called something other
// than "§" ("Book 1, §3" -> "Book 1, Chapter 3")
const SECTION_NAMES: Record<string, string> = {
  Nicomachean_Ethics: "Chapter",
};

export function formatSectionLabel(label: string | null | undefined, filename: string): string | null {
  if (!label) return null;
  const stem = bookStem(filename);
  const divisions = Object.hasOwn(TITLED_DIVISIONS, stem) ? TITLED_DIVISIONS[stem] : undefined;
  const kind = divisions && Object.hasOwn(divisions, label) ? divisions[label] : undefined;
  if (kind) return `${kind}: ${label}`;
  const sectionName = Object.hasOwn(SECTION_NAMES, stem) ? SECTION_NAMES[stem] : undefined;
  return sectionName ? label.replace(/§(\S+)/, `${sectionName} $1`) : label;
}
