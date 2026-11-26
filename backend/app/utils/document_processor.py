from PyPDF2 import PdfReader
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import os
import re
import nltk
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class DocumentProcessor:
    """Process different document types and extract text."""

    def __init__(self):
        """Initialize the document processor and download NLTK data."""
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

    def extract_text(self, file_path: str, file_ext: str) -> str:
        """
        Extract text from a document based on its file extension.

        Args:
            file_path: Path to the document file
            file_ext: File extension (e.g., '.pdf', '.txt', '.md', '.docx')

        Returns:
            Extracted text as a string
        """
        if file_ext == ".pdf":
            return self._extract_from_pdf(file_path)
        elif file_ext == ".txt" or file_ext == ".md":
            return self._extract_from_text(file_path)
        elif file_ext == ".docx":
            return self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            # Empty pages are skipped: their blank line would switch the whole
            # document out of line mode
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(p.strip("\n") for p in pages if p.strip()).strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from a TXT or MD file."""
        try:
            with open(file_path, "rb") as f:
                return _decode_text(f.read()).strip()
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = Document(file_path)
            # Each DOCX paragraph is a real paragraph, so they are separated by
            # a blank line, as in the books' text files. Paragraph styles say
            # what Word knows: headings become markdown "#" headings and list
            # items get a bullet, so neither is mistaken for the other. Tables
            # are kept in document order, one row per line.
            blocks: List[str] = []

            def walk(container) -> None:
                for child in container.iterchildren():
                    tag = child.tag.rsplit("}", 1)[-1]
                    if tag == "p":
                        paragraph = Paragraph(child, doc)
                        text = _docx_text(child)
                        if not text:
                            for box in _docx_text_boxes(child):
                                walk(box)
                            continue
                        style = (paragraph.style.name if paragraph.style is not None else "") or ""
                        if style == "Title" or style.startswith("Heading"):
                            text = f"# {text}"
                        elif style.startswith("List"):
                            text = f"• {text}"
                        blocks.append(text)
                        # Text boxes anchored in the paragraph, once each
                        for box in _docx_text_boxes(child):
                            walk(box)
                    elif tag == "tbl":
                        rows = []
                        for row in Table(child, doc).rows:
                            cells, seen = [], set()
                            for cell in row.cells:
                                # A merged cell appears once per grid column; keep it once
                                if id(cell._tc) in seen:
                                    continue
                                seen.add(id(cell._tc))
                                text = " ".join(
                                    t for t in (_docx_text(p) for p in cell._tc.iter(_W + "p")) if t
                                )
                                if text:
                                    cells.append(text)
                            if cells:
                                rows.append(" | ".join(cells))
                        if rows:
                            blocks.append("\n".join(rows))
                    elif tag in ("sdt", "sdtContent", "ins", "customXml", "smartTag"):
                        # Content controls and tracked insertions wrap paragraphs
                        walk(child)

            walk(doc.element.body)
            return "\n\n".join(blocks)
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")

    def split_into_sentences_and_paragraphs(
        self, text: str
    ) -> Tuple[List[str], List[int]]:
        """
        Split text into sentences and track which passage each sentence belongs to.

        Thin wrapper around parse_document() for callers that don't need labels.

        Returns:
            Tuple of (sentences, paragraph_indices) where paragraph_indices[i] is
            the passage index (0-based) for sentences[i]
        """
        parsed = self.parse_document(text)
        return (parsed.sentences, parsed.paragraph_indices)

    def parse_document(self, text: str) -> "ParsedDocument":
        """
        Split text into citable passages and sentences, using the book's own structure.

        Passages follow the author's divisions rather than arbitrary lengths:
        - Headings ("THE SIXTH BOOK", "CHAPTER IV. ...", "Absurd Walls") are not
          indexed as text; they become the label of the passages beneath them.
        - Numbered sections ("XLIX.", "63.", "52.") and bare chapter numbers
          (Tao Te Ching) always start a new passage and set its section label.
          The number is removed from the passage text, since the label shows it.
          Bare chapter numbers must count up from 1; other bare numbers are
          page numbers and are dropped.
        - Markdown "#" headings are headings; an attribution line
          ("— Seneca") never is.
        - Runs of numbered footnotes ("5 Apropos of ...") that end a chapter,
          asterisk note markers and inline "[Footnote: ...]" notes are dropped.
          A numbered list in the middle of prose is kept.
        - A paragraph starting in lowercase finishes the previous sentence
          (prose resuming after a quoted verse), so it joins that passage, up
          to twice MAX_PASSAGE_WORDS when the previous one stops mid-sentence.
        - A paragraph opening on a linking word ("Similarly, ...", "But ...")
          joins the previous passage when both fit within MAX_PASSAGE_WORDS.
        - Short fragments (epigraphs, verse lines, "And this third:") are joined
          to the previous passage in the same section instead of standing alone.
        - Stanzas of a numbered chapter are kept together as one passage.
        - Each passage's display text keeps its line breaks (verse) and puts a
          blank line between the paragraphs it joins.

        Returns:
            ParsedDocument with sentences, a passage index per sentence, a
            label per passage (e.g. "Book 6, §49"; absent when the text has no
            detectable structure) and each passage's display text
        """
        if not text or not text.strip():
            return ParsedDocument([], [], {})
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
        # Roman section "I." only when the text also has a section "II.";
        # otherwise "I. Kant argued..." is a name initial
        has_roman_two = bool(re.search(r"(?m)^\s*II\.\s", text))

        # A blank line may contain spaces, tabs or non-breaking spaces
        raw_paragraphs = re.split(r"\n[^\S\n]*\n", text)
        # Text without blank lines (typical of PDF extraction) is split per
        # line; each line is then usually part of a wrapped sentence, so only
        # unambiguous headings are recognised and footnote stripping is off
        line_mode = len(raw_paragraphs) == 1
        if line_mode:
            # Rejoin words hyphenated across lines ("philoso-\nphy")
            raw_paragraphs = _LINE_END_HYPHEN.sub("", text).split("\n")

        heading: Optional[str] = None
        section: Optional[str] = None
        in_chapter_mode = False  # True while inside a bare-numbered chapter
        pending_section_start = False  # a bare chapter number stood alone
        last_footnote = 0  # number of the footnote just before, 0 if none
        last_chapter = 0  # number of the last bare-numbered chapter, 0 if none
        seen_roman_section = False
        text_since_heading = False  # body text seen since the last heading
        passages: List[dict] = []  # {"label": str | None, "parts": [str], "words": int}

        def label() -> Optional[str]:
            if section and section.startswith("Chapter"):
                return section
            return ", ".join(p for p in (heading, section) if p) or None

        def is_chapter_number(number: int) -> bool:
            # Chapters count up from 1, and chapter 1 comes before any body
            # text (after a heading at most). Page numbers fail this: page
            # "1" follows the first page's text, and "2" doesn't follow a
            # chapter 1. Line-mode text has no bare chapter numbers.
            if line_mode:
                return False
            if last_chapter and number == last_chapter + 1:
                return True
            return number == 1 and not text_since_heading

        def previous_text_line(idx: int) -> str:
            """The nearest earlier non-empty line that isn't a bare number."""
            for earlier in reversed(raw_paragraphs[:idx]):
                earlier = earlier.strip()
                if earlier and not _CHAPTER_NUMBER_LINE.match(earlier):
                    return earlier
            return ""

        for raw_idx, raw in enumerate(raw_paragraphs):
            lines = [ln.strip() for ln in raw.strip().split("\n")]
            lines = [ln for ln in lines if ln]
            if not lines:
                continue

            # A chapter number on its own ("1" then a blank line, then text).
            # Chapters count up from 1, so any other bare number is a page
            # number, as is every bare number in line mode; all are dropped.
            if len(lines) == 1 and _CHAPTER_NUMBER_LINE.match(lines[0]):
                number = int(lines[0])
                if is_chapter_number(number):
                    section = f"Chapter {number}"
                    last_chapter = number
                    in_chapter_mode = pending_section_start = True
                continue

            # A whole single-line paragraph that reads like a title
            # In line mode a "standalone" line is really one wrapped line, so the
            # following line must start a new sentence for it to be a heading
            following = (
                raw_paragraphs[raw_idx + 1].strip()
                if line_mode and raw_idx + 1 < len(raw_paragraphs)
                else None
            )
            # In wrapped text a heading follows a finished sentence; a
            # heading-like line in mid-sentence is a running page header
            if line_mode and len(lines) == 1 and _is_heading_line(lines[0], standalone=True, line_mode=True):
                before = previous_text_line(raw_idx)
                if before and not _SENTENCE_END.search(before) and not _is_heading_line(
                    before, standalone=True, line_mode=True
                ):
                    continue
            # A numbered line in title case ("1. Courage", "2. Related Work") is
            # a list item or numbered section with its text, not a bare heading;
            # an all-caps one ("237. SEVEN APOPHTHEGMS FOR WOMEN") is a heading
            numbered = _ARABIC_SECTION.match(lines[0])
            numbered_text = bool(numbered) and not lines[0][numbered.end():].isupper()
            if len(lines) == 1 and not numbered_text and _is_heading_line(
                lines[0], standalone=True, line_mode=line_mode, next_line=following
            ):
                heading, section, in_chapter_mode = _heading_label(lines[0]), None, False
                text_since_heading = False
                # "237. SEVEN APOPHTHEGMS FOR WOMEN" is both a heading and a section
                m = _ARABIC_SECTION.match(lines[0])
                if m:
                    heading, section = None, f"§{m.group(1)}{m.group(2)}"
                continue

            # Leading heading lines inside a paragraph ("THE FIRST BOOK\nI. Of my...")
            popped_heading = False
            while len(lines) > 1 and _is_heading_line(
                lines[0], standalone=False, line_mode=line_mode, next_line=lines[1]
            ):
                heading, section, in_chapter_mode = _heading_label(lines.pop(0)), None, False
                text_since_heading = False
                popped_heading = True
            # The rest of a heading-only paragraph is its title
            # ("CHAPTER I.\nThe Period" -> "Chapter I: The Period")
            if (
                popped_heading
                and len(lines) == 1
                and not _ARABIC_SECTION.match(lines[0])
                and not _ROMAN_SECTION.match(lines[0])
                and _is_heading_line(lines[0], standalone=True, line_mode=line_mode)
            ):
                heading = f"{heading}: {_heading_label(lines[0])}"
                continue

            starts_section = pending_section_start
            pending_section_start = False
            if _CHAPTER_NUMBER_LINE.match(lines[0]) and len(lines) > 1:
                number = int(lines.pop(0))
                if is_chapter_number(number):
                    section = f"Chapter {number}"
                    last_chapter = number
                    in_chapter_mode = starts_section = True
            body = "\n".join(lines)

            # Footnotes come in runs numbered 1, 2, 3 ... at the end of a
            # chapter; a lone "12 Angry Men is a film..." or a numbered list
            # in the middle of prose ("1 Cup of flour") is kept as text
            footnote = None if line_mode else _FOOTNOTE.match(body)
            if footnote and int(footnote.group(1)) == 1:
                if not _is_footnote_run(raw_paragraphs, raw_idx):
                    footnote = None
            if footnote and int(footnote.group(1)) in (1, last_footnote + 1):
                last_footnote = int(footnote.group(1))
                continue
            last_footnote = 0

            m = _ROMAN_SECTION.match(body)
            if m and len(m.group(1)) == 1 and m.group(1) != "I" and not seen_roman_section:
                # A lone "C." / "L." / "V." is far more often a name initial
                # ("C. Wright Mills") than section 100 / 50 / 5 of a text that
                # has not used roman section numbers yet
                m = None
            if m and m.group(1) == "I" and not has_roman_two:
                m = None
            m = m or _ARABIC_SECTION.match(body)
            if m and len(lines) > 1 and _ARABIC_SECTION.match(lines[1]):
                # "1. Gather...\n2. Weigh..." in one paragraph is a list
                m = None
            if m and line_mode and raw_idx > 0:
                # In wrapped text a line can start with a number mid-sentence
                # ("...until the year\n476. After that..."); a section starts
                # only after a sentence or heading has ended
                # (or a lead-in such as "three steps:")
                before = previous_text_line(raw_idx)
                if (
                    before
                    and not _SENTENCE_END.search(before)
                    and not before.endswith(":")
                    and not _is_heading_line(before, standalone=True, line_mode=True)
                ):
                    m = None
            if m and not starts_section:
                if m.re is _ROMAN_SECTION:
                    seen_roman_section = True
                    section = f"§{_roman_to_int(m.group(1))}"
                else:
                    section = f"§{m.group(1)}{m.group(2)}"
                in_chapter_mode = False
                starts_section = True
                # The number is now the passage label, so it isn't repeated
                # in the text ("9. Courage is..." under "Book 3, Chapter 9")
                body = body[m.end():].lstrip()
                # A few sections open in lowercase after their number
                # ("XXIV. if any should put this question...")
                body = body[:1].upper() + body[1:]

            # Notes whose text isn't in the file ("those* who", "chriae *") or
            # that interrupt the author mid-sentence ("[Footnote: ...]")
            if not line_mode:
                # A paragraph with markdown emphasis ("*a* point") keeps its asterisks
                if not _EMPHASIS_OPENER.search(body):
                    body = _NOTE_MARKER.sub("", body)
                body = _INLINE_FOOTNOTE.sub("", body).strip()
                if not body:
                    continue

            if _is_wrapped_prose(lines):
                body = _LINE_END_HYPHEN.sub("", body)
            words = len(body.split())
            # The passage as shown to readers: verse keeps its line breaks, while
            # hard-wrapped prose is unwrapped into one paragraph
            display = re.sub(r"\s*\n\s*", " ", body) if _is_wrapped_prose(lines) else body
            prev = passages[-1] if passages else None
            same_section = prev is not None and prev["label"] == label()
            fits = prev is not None and prev["words"] + words <= MAX_PASSAGE_WORDS
            # A paragraph opening in lowercase finishes the sentence before it,
            # typically prose resuming after a quoted verse ("...long hence,"
            # / "according to our Virgil, who..."). When the previous paragraph
            # really stops mid-sentence it may grow the passage to twice the
            # usual limit; otherwise it joins only if both fit, so a document
            # written all in lowercase isn't merged into one passage.
            continues_sentence = body[:1].islower() and (
                fits
                or (
                    prev is not None
                    and not _SENTENCE_END.search(prev["parts"][-1])
                    and prev["words"] + words <= 2 * MAX_PASSAGE_WORDS
                )
            )
            # A paragraph opening on a linking word ("Similarly, people who...",
            # "But it is not...") answers the paragraph before it, so it joins
            # that passage when the two fit together
            continues_thought = fits and bool(LEAD_IN_WORD.match(body))
            joins_previous = (
                not starts_section
                and same_section
                and (
                    # Line mode: each "paragraph" is one wrapped line, so keep
                    # joining; long runs are split at sentence boundaries below
                    line_mode
                    or continues_sentence
                    or continues_thought
                    or (fits and (in_chapter_mode or words < SHORT_FRAGMENT_WORDS))
                )
            )

            text_since_heading = True
            if joins_previous:
                prev["parts"].append(body)
                prev["display"].append(display)
                prev["words"] += words
                prev["wrapped"] = prev["wrapped"] or line_mode or _is_wrapped_prose(lines)
            else:
                passages.append(
                    {
                        "label": label(),
                        "parts": [body],
                        "display": [display],
                        "words": words,
                        "wrapped": line_mode or _is_wrapped_prose(lines),
                    }
                )

        sentences: List[str] = []
        paragraph_indices: List[int] = []
        paragraph_labels: Dict[int, str] = {}
        paragraph_texts: Dict[int, str] = {}
        current_para_idx = 0

        for passage in passages:
            # A long block of hard-wrapped prose (a PDF page, a DOCX section) has
            # no author paragraph breaks to follow, so split it at sentence
            # boundaries into passages of about MAX_PASSAGE_WORDS words
            chunk_words = 0
            # Any passage beyond MAX_UNSPLIT_PASSAGE_WORDS is split the same way
            passage_chars = sum(len(part) for part in passage["parts"])
            split_long = (
                (passage["wrapped"] and passage["words"] > WRAPPED_SPLIT_WORDS)
                or passage["words"] > MAX_UNSPLIT_PASSAGE_WORDS
                or passage_chars > MAX_UNSPLIT_PASSAGE_CHARS
            )
            chunk_chars = 0
            added = False
            # In line mode the parts are wrapped lines of the same text, so
            # sentences may continue across them
            parts = [" ".join(passage["parts"])] if line_mode else passage["parts"]
            for part in parts:
                carry = ""  # a bare section number nltk split off ("1.")
                for sentence in _sentences(part):
                    if _BARE_NUMBER_SENTENCE.match(sentence):
                        carry = f"{carry}{sentence} "
                        continue
                    sentence_words = len(sentence.split())
                    if (
                        split_long
                        and added
                        and (
                            chunk_words + sentence_words > MAX_PASSAGE_WORDS
                            or chunk_chars + len(sentence) > MAX_SENTENCE_CHARS
                        )
                    ):
                        if passage["label"]:
                            paragraph_labels[current_para_idx] = passage["label"]
                        current_para_idx += 1
                        chunk_words = chunk_chars = 0
                    sentences.append(carry + sentence)
                    paragraph_indices.append(current_para_idx)
                    chunk_words += sentence_words
                    chunk_chars += len(sentence) + 1
                    carry = ""
                    added = True
                if carry.strip():
                    sentences.append(carry.strip())
                    paragraph_indices.append(current_para_idx)
                    added = True
            if added:
                if passage["label"]:
                    paragraph_labels[current_para_idx] = passage["label"]
                # Paragraphs of a passage are separated by a blank line. Passages
                # split at sentence boundaries, and line-mode text, have no
                # breaks of the author's to keep; their text is rebuilt from
                # the sentences.
                if not (split_long or line_mode):
                    paragraph_texts[current_para_idx] = "\n\n".join(passage["display"])
                current_para_idx += 1

        return ParsedDocument(sentences, paragraph_indices, paragraph_labels, paragraph_texts)


# --- Structure detection -----------------------------------------------------

# Passages are grown past this by merging only when a lowercase paragraph
# finishes an unfinished sentence (up to twice this); a long single section
# stays whole up to MAX_UNSPLIT_PASSAGE_WORDS
MAX_PASSAGE_WORDS = 250
# Longer passages (only in uploads; the longest in the library is ~1,300 words)
# are split at sentence boundaries
MAX_UNSPLIT_PASSAGE_WORDS = 3000
# A "sentence" longer than this (text with no full stops) is cut into pieces
# so it can still be embedded; the library's longest sentence is ~440 words
MAX_SENTENCE_WORDS = 1000
# ... and so is one longer than this many characters (text without spaces,
# such as Chinese, or very long words); the library's longest is ~2,700. The
# API accepts excerpts of up to 5,000 characters, and an excerpt is never
# shorter than its sentence.
MAX_SENTENCE_CHARS = 5000
# A passage longer than this many characters is split like a long one
MAX_UNSPLIT_PASSAGE_CHARS = 20000
# Unnumbered paragraphs shorter than this are joined to the previous passage
SHORT_FRAGMENT_WORDS = 20
# Hard-wrapped prose blocks longer than this are split into ~MAX_PASSAGE_WORDS
WRAPPED_SPLIT_WORDS = 400

# "XLIX. That which..." but not initials such as "C. S. Lewis" or "V. I. Lenin"
_ROMAN_SECTION = re.compile(r"^([IVXLC]{1,9})\.\s(?![A-Z]\.)")
# "63. He who...", "65A. We..."; up to three digits, so a paragraph opening
# on a year ("1984. Orwell's novel...") isn't read as section 1984
_ARABIC_SECTION = re.compile(r"^(\d{1,3})([A-Z]?)\.\s")
_BARE_NUMBER_SENTENCE = re.compile(r"^(\d{1,4}[A-Z]?|[IVXLC]{1,9})\.$")
_CHAPTER_NUMBER_LINE = re.compile(r"^\d{1,3}$")  # "13" alone on a line
_FOOTNOTE = re.compile(r"^(\d{1,2}) (?=[A-Z\"“'‘])")  # "5 Apropos of the notion..."
# A paragraph that opens by answering the one before it ("Similarly, ...")
LEAD_IN_WORD = re.compile(
    r"^[\"“‘'(]*(but|and|yet|so|nor|or|thus|hence|therefore|accordingly|consequently"
    r"|similarly|likewise|nevertheless|nonetheless|however|still|instead)\b",
    re.IGNORECASE,
)
# Text that ends a sentence: "...done." / "...it?”"
_SENTENCE_END = re.compile(r"[.!?…][\"”’')\]]*\s*$")
# A bullet or numbered list item: "- Courage", "• Justice", "2. Weigh", "3) Decide"
_LIST_ITEM = re.compile(r"^(?:[-*•+–]|\d{1,3}[.)])\s")
# A word hyphenated across a line break: "philoso-\nphy"
_LINE_END_HYPHEN = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
# An asterisk opening markdown emphasis: "*a* point", "**bold**"
_EMPHASIS_OPENER = re.compile(r"(?:^|[\s(\[\"“‘'])\*+[^\s*]")
# An asterisk note marker: "those* who", "chriae * to learn", "honey’.*".
# A single asterisk after a word or punctuation, so "2 * 3" and markdown
# "**bold**" are left alone.
_NOTE_MARKER = re.compile(r"(?<=[^\s\d*]) ?\*(?=\s|$)")
_INLINE_FOOTNOTE = re.compile(r"\s*\[Footnote:[^\]]*\]")  # "[Footnote: Like the frog: staccato.]"
# "THE SIXTH BOOK" / "HIS FIRST BOOK" as the whole line, or "CHAPTER IX",
# "Part 2", "Chapter 3: The Return", optionally followed by a title in caps or
# title case. "The second book of Plato's Republic opens..." and "Part 2 of the
# argument shows..." are prose, not headings.
_LEAD_IN_WORDS = {"but", "and", "or", "yet", "so", "nor", "for", "then", "thus", "hence", "again"}
_LETTER_HEADING = re.compile(r"^(?:letter|epistle)\s+(?:\d{1,3}|[IVXLC]{1,7}):\s+\S", re.IGNORECASE)
_STRUCTURAL_HEADING = re.compile(
    r"^(?:(?:the|his)\s+\w+\s+book|(?:chapter|book|part|section)\s+(?:[ivxlc]+|\d+))"
    r"[.:]?(?:\s+(?P<title>.+))?$",
    re.IGNORECASE,
)
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?=\S)")
_HEADING_KEYWORD = re.compile(
    r"^(section|chapter|book|part|appendix|preface|introduction)\b", re.IGNORECASE
)
_SMALL_WORDS = {"a", "an", "and", "the", "of", "in", "on", "to", "for", "by", "or", "at", "with"}
_ORDINALS = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
    "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10, "ELEVENTH": 11, "TWELFTH": 12,
}


@dataclass
class ParsedDocument:
    """Sentences grouped into citable passages, with a label per passage."""

    sentences: List[str]
    paragraph_indices: List[int]
    paragraph_labels: Dict[int, str] = field(default_factory=dict)
    # Passage text with its line and paragraph breaks, for display; passages
    # missing here are shown as their sentences joined by spaces
    paragraph_texts: Dict[int, str] = field(default_factory=dict)


def _is_heading_line(
    line: str,
    standalone: bool,
    line_mode: bool = False,
    next_line: Optional[str] = None,
) -> bool:
    """
    Decide whether a line is a heading rather than prose.

    Args:
        line: The line to test
        standalone: True when the line is a whole paragraph on its own
        line_mode: True when the text had no blank lines, so every line is a
            fragment of wrapped prose; only explicit or all-caps headings count
        next_line: For a leading line inside a paragraph, the line after it.
            A heading is followed by the start of a new sentence, so a
            lowercase continuation ("I SAY UNTO YOU\nthat one must...") means
            the line is prose.
    """
    words = line.split()
    # Markdown: "# Notes", "## What is virtue?"
    if _MARKDOWN_HEADING.match(line) and len(line) <= 160:
        return True
    # An attribution ("— Friedrich Nietzsche") belongs to the quotation above,
    # and a bullet ("* Courage", "• Justice") is a list item
    if line[:1] in "—–-~*•+":
        return False
    # "Letter 14: On the Reasons for Withdrawing from the World" (Seneca):
    # the numbered prefix is unambiguous, so the title may be any case or length
    if _LETTER_HEADING.match(line) and len(words) <= 24 and len(line) <= 160:
        return True
    if not words or len(words) > 12 or len(line) > 90:
        return False

    # "THE SIXTH BOOK", "CHAPTER IX. WHAT IS NOBLE?", "Part 2"
    structural = _STRUCTURAL_HEADING.match(line)
    if structural and len(words) <= 10:
        title = structural.group("title")
        if not title:
            return True
        title_words = [w for w in title.split() if w[0].isalpha()]
        if title.isupper() or all(
            w[0].isupper() or w.lower() in _SMALL_WORDS for w in title_words
        ):
            return True

    if _ROMAN_SECTION.match(line) and not standalone:
        return False
    # A table row ("VIRTUE | DEFICIENCY | EXCESS") is not a heading
    if " | " in line or line.startswith("|"):
        return False
    # A numbered list item without a period ("2 Eggs") is not a title
    if re.match(r"^\d+\s", line):
        return False
    # Headings don't end like sentences or clauses
    if re.search(r"[.!?;:,\"”’')]$", line):
        return False
    if next_line is not None and not (next_line[:1].isupper() or next_line[:1].isdigit()):
        return False

    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    # A lone conjunction ("But", introducing a quotation) is never a heading
    if len(words) == 1 and words[0].lower() in _LEAD_IN_WORDS:
        return False
    if sum(c.isupper() for c in letters) / len(letters) >= 0.8:
        return True
    if line_mode:
        return False

    # Title case ("Absurd Walls", "APPENDIX: Hope and the Absurd in ..."):
    # allowed for a standalone line, or a leading line that names a division
    title_case = all(
        w[0].isupper() or w.lower() in _SMALL_WORDS for w in words if w[0].isalpha()
    )
    return title_case and (standalone or bool(_HEADING_KEYWORD.match(line)))


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


_MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
# Subtrees whose text isn't part of the paragraph: deletions and the old
# position of moved text (tracked changes), the fallback copy Word stores of
# every drawing, and text boxes (extracted as paragraphs of their own)
_DOCX_SKIP = {_W + "del", _W + "moveFrom", _MC + "Fallback", _W + "txbxContent"}


def _docx_text(p_element) -> str:
    """A DOCX paragraph's text, including tracked insertions and fields
    (which python-docx's paragraph.text can skip), excluding deletions."""
    parts: List[str] = []

    def visit(node) -> None:
        for child in node:
            tag = child.tag
            if tag in _DOCX_SKIP:
                continue
            if tag == _W + "t":
                parts.append(child.text or "")
            elif tag == _W + "tab":
                parts.append("\t")
            elif tag in (_W + "br", _W + "cr"):
                parts.append("\n")
            elif tag == _W + "noBreakHyphen":
                parts.append("-")
            elif tag == _W + "sym":
                code = int(child.get(_W + "char", "0"), 16)
                # Symbol-font characters are stored in the private-use area
                if code >= 0xF000:
                    code -= 0xF000
                if code >= 0x20:
                    parts.append(chr(code))
            else:
                visit(child)

    visit(p_element)
    return "".join(parts).strip()


def _docx_text_boxes(p_element) -> List:
    """Text boxes in a paragraph, skipping the fallback copy Word also stores."""
    boxes = []
    for box in p_element.iter(_W + "txbxContent"):
        parent, fallback = box.getparent(), False
        while parent is not None and parent is not p_element:
            if parent.tag == _MC + "Fallback":
                fallback = True
                break
            parent = parent.getparent()
        if not fallback:
            boxes.append(box)
    return boxes


def _decode_text(data: bytes) -> str:
    """
    Decode an uploaded text file: UTF-16 when it has a UTF-16 byte-order mark
    (Windows "Unicode"), else UTF-8 (a BOM is removed). A file that is mostly
    valid UTF-8 with a few stray bytes keeps its UTF-8 text (the stray bytes
    are dropped); otherwise it is taken as Windows "ANSI" (cp1252), with
    latin-1 as the last resort.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    # UTF-8 with a few stray bytes still has valid UTF-8 characters, and more
    # of them than bad bytes; a cp1252 file ("café" as a single byte) has none
    lenient = data.decode("utf-8-sig", errors="replace")
    bad = lenient.count("\ufffd")
    valid_non_ascii = sum(1 for c in lenient if ord(c) > 0x7F) - bad
    if valid_non_ascii > bad:
        return lenient.replace("\ufffd", "")
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _is_footnote_run(raw_paragraphs: List[str], start: int) -> bool:
    """
    True when the paragraph at `start` ("1 From the point of view...") opens a
    run of footnotes: notes numbered 1, 2, 3 ... that end a chapter, so the run
    is followed by a heading, a chapter number, another run, or the end of the
    text. A numbered list in the middle of prose is followed by more prose.
    """
    paragraphs = [p.strip() for p in raw_paragraphs[start:] if p.strip()]
    expected, i = 1, 0
    while i < len(paragraphs):
        m = _FOOTNOTE.match(paragraphs[i])
        if not (m and int(m.group(1)) == expected):
            break
        # Notes are sentences; list items ("1 Gather the evidence") often aren't
        if not _SENTENCE_END.search(paragraphs[i]):
            return False
        expected, i = expected + 1, i + 1
    if expected <= 2:  # a single "1 ..." is prose
        return False
    if i == len(paragraphs):
        return True
    after = paragraphs[i]
    first_line = after.split("\n")[0].strip()
    if _CHAPTER_NUMBER_LINE.match(first_line):
        return True
    m = _FOOTNOTE.match(after)
    if m and int(m.group(1)) == 1:
        return True
    return "\n" not in after and _is_heading_line(after, standalone=True)


def _sentences(text: str) -> List[str]:
    """
    nltk's sentences, with any longer than MAX_SENTENCE_WORDS words or
    MAX_SENTENCE_CHARS characters cut into pieces (at spaces where possible).
    """
    out: List[str] = []
    for sentence in nltk.sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence.split()) <= MAX_SENTENCE_WORDS and len(sentence) <= MAX_SENTENCE_CHARS:
            out.append(sentence)
            continue
        piece: List[str] = []
        piece_chars = 0
        for word in sentence.split():
            # A single "word" longer than the limit (no spaces) is sliced
            while len(word) > MAX_SENTENCE_CHARS:
                if piece:
                    out.append(" ".join(piece))
                    piece, piece_chars = [], 0
                out.append(word[:MAX_SENTENCE_CHARS])
                word = word[MAX_SENTENCE_CHARS:]
            if piece and (
                len(piece) >= MAX_SENTENCE_WORDS
                or piece_chars + 1 + len(word) > MAX_SENTENCE_CHARS
            ):
                out.append(" ".join(piece))
                piece, piece_chars = [], 0
            piece.append(word)
            piece_chars += len(word) + (1 if piece_chars else 0)
        if piece:
            out.append(" ".join(piece))
    return out


def _is_wrapped_prose(lines: List[str]) -> bool:
    """
    True for a block of hard-wrapped prose: many lines of similar, moderate
    length (PDF extraction). Books in this repo use one line per
    paragraph, and verse has short lines, so neither qualifies.
    """
    if len(lines) < 5:
        return False
    # Table rows ("Virtue | Vice") and list items keep their line breaks
    if sum(" | " in line for line in lines) * 2 >= len(lines):
        return False
    if sum(bool(_LIST_ITEM.match(line)) for line in lines) * 2 >= len(lines):
        return False
    lengths = sorted(len(line) for line in lines)
    median = lengths[len(lengths) // 2]
    return 40 <= median <= 120 and lengths[-1] <= 160


def _heading_label(line: str) -> str:
    """Turn a raw heading into a short citation label."""
    line = _MARKDOWN_HEADING.sub("", line)
    # Markdown emphasis around a heading ("## **Bold Title**")
    line = re.sub(r"^[*_]+(?=\S)|(?<=\S)[*_]+$", "", line)
    upper = line.upper()
    m = re.match(r"^(?:THE|HIS) (\w+) BOOK", upper)
    if m and m.group(1) in _ORDINALS:
        return f"Book {_ORDINALS[m.group(1)]}"
    m = re.match(r"^(CHAPTER|BOOK|PART) ([IVXLC]+|\d+)\b", upper)
    if m:
        return f"{m.group(1).title()} {m.group(2)}"
    line = re.sub(r"^section:\s*", "", line, flags=re.IGNORECASE)

    # "AN ABSURD REASONING Absurdity and Suicide" -> part title in caps, then the
    # chapter title; the chapter title is the more specific label.
    # "FROM THE HEIGHTS By F. W. Nietzsche ..." -> caps title, then a byline.
    # Words without lowercase letters ("SECTION 2", "APPENDIX A", "Q&A") all
    # belong to the capitals.
    words = line.split()
    caps = 0
    while caps < len(words) and not any(c.islower() for c in words[caps]):
        caps += 1
    if 0 < caps < len(words) and not words[0].endswith(":") and any(
        c.isalpha() for w in words[:caps] for c in w
    ):
        rest = " ".join(words[caps:])
        line = " ".join(words[:caps]) if rest.lower().startswith("by ") else rest

    if not any(c.islower() for c in line):
        line = _title_case(line)
    elif not any(c.islower() for c in line.split(":")[0]):
        # "APPENDIX: Hope and the Absurd in ..." -> caps head, title-case tail
        head, sep, tail = line.partition(":")
        line = _title_case(head) + sep + tail
    return line if len(line) <= 80 else line[:77].rstrip() + "..."


_ROMAN_NUMERAL = re.compile(r"^(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")


def _title_case(text: str) -> str:
    """Title case for an all-caps heading. Roman numerals ("II.") and
    abbreviations ("U.S.", "Q&A") stay in capitals, and a word after a
    period or colon starts a new phrase ("III. A New Hope")."""
    out: List[str] = []
    phrase_start = True
    words = text.split()
    for i, word in enumerate(words):
        core = word.strip(".,:;!?()[]\"'“”‘’")
        if core and (_ROMAN_NUMERAL.match(core) or re.search(r"[A-Za-z][.&'][A-Za-z]", core)):
            out.append(word)
        # Small words stay lowercase, except first and last ("Appendix A")
        elif not phrase_start and i < len(words) - 1 and word.lower() in _SMALL_WORDS:
            out.append(word.lower())
        else:
            out.append(word[:1].upper() + word[1:].lower())
        phrase_start = word.endswith((".", ":"))
    return " ".join(out)


def _roman_to_int(numeral: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for i, ch in enumerate(numeral):
        v = values[ch]
        total += -v if i + 1 < len(numeral) and values[numeral[i + 1]] > v else v
    return total
