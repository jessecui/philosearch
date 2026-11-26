// Parse the model's Markdown-ish answer into blocks: paragraphs, "#" headings,
// "> " quotes, and bullet / numbered lists. Lists may have blank lines between
// items, start at any number, and contain indented sub-bullets.
// What follows an item's first line, in order: paragraphs under the item
// and sub-lists, which may alternate
export type ItemPart =
  | { kind: "text"; text: string }
  | { kind: "sub"; ordered: boolean; items: string[] };
export type ListItem = {
  text: string;
  parts: ItemPart[];
  indent: number; // leading whitespace of the item's line
};
export type AnswerBlock =
  | { kind: "paragraph"; text: string }
  | { kind: "heading"; text: string }
  | { kind: "quote"; text: string }
  | { kind: "rule" }
  | { kind: "ordered"; start: number; items: ListItem[] }
  | { kind: "unordered"; start: number; items: ListItem[] };

export function parseAnswer(text: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let prose: string[] = [];
  let quoteOpen = false; // consecutive "> " lines form one quote
  let previousBlank = false;
  const flushProse = () => {
    if (prose.length) blocks.push({ kind: "paragraph", text: prose.join(" ") });
    prose = [];
  };

  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) {
      flushProse(); // a blank line ends a paragraph or quote, but not a list
      quoteOpen = false;
      previousBlank = true;
      continue;
    }
    const afterBlank = previousBlank;
    previousBlank = false;
    if (/^([-*_])(\s*\1){2,}$/.test(line)) {
      flushProse(); // "---" horizontal rule
      quoteOpen = false;
      blocks.push({ kind: "rule" });
      continue;
    }
    const ordered = line.match(/^(\d+)[.)]\s+(.*)$/);
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const quote = line.match(/^>\s?(.*)$/);
    const last = blocks[blocks.length - 1];
    const lastIsList =
      !prose.length && (last?.kind === "ordered" || last?.kind === "unordered");

    // Bullet or number indented further than the current list item: a
    // sub-point of that item (a list indented as a whole stays one level)
    const indent = indentWidth(raw);
    const parentIndent = lastIsList && last.items.length ? last.items[last.items.length - 1].indent : 0;
    if ((bullet || ordered) && lastIsList && last.items.length && indent >= parentIndent + 2) {
      const parent = last.items[last.items.length - 1];
      const lastPart = parent.parts[parent.parts.length - 1];
      const content = ordered ? ordered[2] : bullet![1];
      // A switch between bullets and numbers starts a new sub-list
      if (lastPart?.kind === "sub" && lastPart.ordered === Boolean(ordered)) lastPart.items.push(content);
      else parent.parts.push({ kind: "sub", ordered: Boolean(ordered), items: [content] });
      continue;
    }
    if (!quote) quoteOpen = false;

    // A plain line right after a list item continues it ("1. **Stoicism**\n
    // Accept what..."); indented under it after a blank line, it starts a
    // paragraph inside the item
    if (
      !ordered && !bullet && !heading && !quote && lastIsList && last.items.length &&
      (!afterBlank || indent > parentIndent)
    ) {
      const item = last.items[last.items.length - 1];
      const lastPart = item.parts[item.parts.length - 1];
      if (afterBlank) item.parts.push({ kind: "text", text: line });
      else if (lastPart?.kind === "sub") lastPart.items[lastPart.items.length - 1] += ` ${line}`;
      else if (lastPart?.kind === "text") lastPart.text += ` ${line}`;
      else item.text += ` ${line}`;
      continue;
    }
    if (ordered || bullet) {
      flushProse();
      const kind = ordered ? "ordered" : "unordered";
      const content = ordered ? ordered[2] : bullet![1];
      const prev = blocks[blocks.length - 1];
      if (prev && prev.kind === kind) {
        prev.items.push({ text: content, parts: [], indent });
      } else {
        blocks.push({
          kind,
          start: ordered ? parseInt(ordered[1]) : 1,
          items: [{ text: content, parts: [], indent }],
        });
      }
      continue;
    }
    if (heading) {
      flushProse();
      blocks.push({ kind: "heading", text: heading[1] });
      continue;
    }
    if (quote) {
      flushProse();
      if (quoteOpen && last && last.kind === "quote") last.text += ` ${quote[1]}`;
      else blocks.push({ kind: "quote", text: quote[1] });
      quoteOpen = true;
      continue;
    }
    prose.push(line);
  }
  flushProse();
  return blocks;
}

// Width of a line's leading whitespace, counting a tab as 4 columns
function indentWidth(line: string): number {
  let width = 0;
  for (const ch of line) {
    if (ch === " ") width += 1;
    else if (ch === "\t") width += 4;
    else break;
  }
  return width;
}
