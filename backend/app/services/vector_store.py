import faiss
import functools
import numpy as np
import json
import os
import threading
import time
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict, field

if TYPE_CHECKING:
    from app.services.reranker_service import CrossEncoderReranker


@dataclass
class DocumentMetadata:
    """Metadata for an indexed document."""

    doc_id: str
    filename: str
    file_type: str
    total_sentences: int
    total_paragraphs: int
    author: Optional[str] = None


@dataclass
class SentenceMetadata:
    """Metadata for an indexed sentence."""

    doc_id: str
    sentence_idx: int
    paragraph_idx: int
    sentence_text: str


@dataclass
class SearchResult:
    """Result from a similarity search - one per paragraph."""

    doc_id: str
    filename: str
    paragraph_idx: int
    paragraph_text: str
    author: Optional[str] = None
    reranking_score: Optional[float] = None
    context_paragraphs_before: Optional[List[str]] = None
    context_paragraphs_after: Optional[List[str]] = None
    # Short citation shown to the user: the matched sentence plus up to two
    # neighbours from the same passage (paragraph_text is the full passage)
    excerpt_text: Optional[str] = None
    # Where the passage sits in the work, e.g. "Book 6, §49" (None if unknown)
    section_label: Optional[str] = None


# Citation excerpts: grow around the matched sentence, inside its own passage
# only, while the total stays within this many characters
EXCERPT_CHAR_BUDGET = 300
EXCERPT_MAX_SENTENCES = 3


def _locked(method):
    """Serialize access to the index: API handlers run in a thread pool."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _transactional(method):
    """Undo in-memory changes if the method fails (e.g. the save to disk), so
    memory never holds a document that was not persisted."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        snapshot = self._snapshot()
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self._restore(snapshot)
            raise

    return wrapper


class VectorStore:
    """FAISS-based vector store for sentence embeddings with paragraph hierarchy."""

    def __init__(self, data_dir: str = "data", embedding_dim: int = 384):
        """
        Initialize the vector store.

        Args:
            data_dir: Directory to store FAISS index and metadata
            embedding_dim: Dimension of embedding vectors (default 384 for all-MiniLM-L6-v2)
        """
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self._lock = threading.RLock()
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.metadata_path = os.path.join(data_dir, "metadata.json")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            if not os.path.exists(self.metadata_path):
                raise RuntimeError(
                    f"{self.index_path} exists but {self.metadata_path} is missing "
                    "(an interrupted save or re-index). Rebuild with "
                    "`python -m app.scripts.index_texts`."
                )
            self.index = faiss.read_index(self.index_path)
            self._load_metadata()
            if self.index.ntotal > len(self.sentences):
                # Vectors are only ever appended, so extra trailing vectors are
                # from a write whose metadata never landed: drop them
                print(
                    f"⚠️  faiss.index has {self.index.ntotal - len(self.sentences)} "
                    "vectors without metadata; discarding them"
                )
                self.index.remove_ids(
                    faiss.IDSelectorRange(len(self.sentences), self.index.ntotal)
                )
            if self.index.ntotal != len(self.sentences):
                raise RuntimeError(
                    f"Index files in {data_dir} are out of sync: faiss.index has "
                    f"{self.index.ntotal} vectors but metadata.json lists "
                    f"{len(self.sentences)} sentences. Rebuild with "
                    "`python -m app.scripts.index_texts`."
                )
        else:
            # Create a new FAISS index (L2 distance)
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents: Dict[str, DocumentMetadata] = {}
            self.sentences: List[SentenceMetadata] = []
            self.paragraph_cache: Dict[Tuple[str, int], str] = (
                {}
            )  # (doc_id, para_idx) -> full paragraph text
            self.paragraph_labels: Dict[Tuple[str, int], str] = {}
        self._rebuild_paragraph_lookup()

    def _load_metadata(self):
        """Load metadata from disk."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = {
                k: DocumentMetadata(**v) for k, v in data["documents"].items()
            }
            self.sentences = [SentenceMetadata(**s) for s in data["sentences"]]
            self.paragraph_cache = {
                (parts[0], int(parts[1])): v
                for k, v in data["paragraph_cache"].items()
                if (parts := k.split("::"))
            }
            # Older indexes have no labels; citations then show no section
            self.paragraph_labels = {
                (parts[0], int(parts[1])): v
                for k, v in data.get("paragraph_labels", {}).items()
                if (parts := k.split("::"))
            }

    def _rebuild_paragraph_lookup(self):
        """Map (doc_id, para_idx) and doc_id -> global sentence indices."""
        self._paragraph_sentences: Dict[Tuple[str, int], List[int]] = {}
        doc_sentences: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self.sentences):
            self._paragraph_sentences.setdefault(
                (meta.doc_id, meta.paragraph_idx), []
            ).append(idx)
            doc_sentences.setdefault(meta.doc_id, []).append(idx)
        self._doc_sentence_ids: Dict[str, np.ndarray] = {
            doc_id: np.array(ids, dtype=np.int64) for doc_id, ids in doc_sentences.items()
        }

    def _faiss_search(
        self, query_embedding: List[float], k: int, doc_ids: Optional[List[str]]
    ):
        """
        FAISS search over all sentences, or only those of `doc_ids` when given
        (the books selected in the UI). Returns (distances, indices), or None
        when nothing is searchable.
        """
        query_array = np.array([query_embedding], dtype=np.float32)
        if doc_ids is None:
            if self.index.ntotal == 0:
                return None
            return self.index.search(query_array, min(k, self.index.ntotal))

        allowed = [self._doc_sentence_ids[d] for d in dict.fromkeys(doc_ids) if d in self._doc_sentence_ids]
        if not allowed:
            return None
        allowed_ids = np.concatenate(allowed)
        selector = faiss.IDSelectorBatch(allowed_ids)  # must outlive the search
        params = faiss.SearchParameters(sel=selector)
        return self.index.search(query_array, min(k, len(allowed_ids)), params=params)

    def _excerpt_span(self, sentence_idx: int) -> Tuple[int, int]:
        """
        Choose the sentences to show for a match: the matched sentence plus its
        nearest neighbours in the same passage, alternating before and after,
        while the excerpt stays within EXCERPT_CHAR_BUDGET characters and
        EXCERPT_MAX_SENTENCES sentences. Never crosses into another passage.

        Returns:
            (first, last) global sentence indices, inclusive
        """
        meta = self.sentences[sentence_idx]
        members = self._paragraph_sentences[(meta.doc_id, meta.paragraph_idx)]
        first = last = sentence_idx
        length = len(meta.sentence_text)

        while last - first + 1 < EXCERPT_MAX_SENTENCES:
            grew = False
            for candidate in (first - 1, last + 1):
                if last - first + 1 >= EXCERPT_MAX_SENTENCES:
                    break
                if candidate < members[0] or candidate > members[-1]:
                    continue
                extra = len(self.sentences[candidate].sentence_text) + 1
                if length + extra <= EXCERPT_CHAR_BUDGET:
                    length += extra
                    first, last = min(first, candidate), max(last, candidate)
                    grew = True
            if not grew:
                break
        return (first, last)

    def _candidate_spans(self, sentence_idx: int) -> List[Tuple[int, int]]:
        """
        All excerpt windows worth scoring for a match: every run of up to
        EXCERPT_MAX_SENTENCES consecutive sentences in the same passage that
        contains the matched sentence and fits EXCERPT_CHAR_BUDGET, keeping only
        the largest ones (a window inside another candidate is dropped).

        The matched sentence is often a conclusion ("Thus it is like the Tao.")
        whose key point sits two sentences earlier, so offering the reranker the
        before / centred / after windows lets it choose the most informative.
        """
        meta = self.sentences[sentence_idx]
        members = self._paragraph_sentences[(meta.doc_id, meta.paragraph_idx)]
        lo, hi = members[0], members[-1]

        def fits(first: int, last: int) -> bool:
            text_len = sum(
                len(self.sentences[i].sentence_text) + 1 for i in range(first, last + 1)
            )
            return first == last or text_len - 1 <= EXCERPT_CHAR_BUDGET

        for size in range(EXCERPT_MAX_SENTENCES, 0, -1):
            spans = [
                (first, first + size - 1)
                for first in range(sentence_idx - size + 1, sentence_idx + 1)
                if first >= lo and first + size - 1 <= hi and fits(first, first + size - 1)
            ]
            if spans:
                return spans
        return [(sentence_idx, sentence_idx)]

    def _excerpt_text(self, span: Tuple[int, int]) -> str:
        return " ".join(
            self.sentences[i].sentence_text for i in range(span[0], span[1] + 1)
        )

    def _snapshot(self):
        return (
            dict(self.documents),
            list(self.sentences),
            dict(self.paragraph_cache),
            dict(self.paragraph_labels),
            self.index,
            self.index.ntotal,
        )

    def _restore(self, snapshot):
        documents, sentences, cache, labels, index, ntotal = snapshot
        if index is self.index and index.ntotal > ntotal:
            # index_document appends in place: remove what it added
            index.remove_ids(faiss.IDSelectorRange(ntotal, index.ntotal))
        self.index = index
        self.documents, self.sentences = documents, sentences
        self.paragraph_cache, self.paragraph_labels = cache, labels
        self._rebuild_paragraph_lookup()

    def _write_metadata(self, path: str):
        data = {
            "documents": {k: asdict(v) for k, v in self.documents.items()},
            "sentences": [asdict(s) for s in self.sentences],
            "paragraph_cache": {
                f"{k[0]}::{k[1]}": v for k, v in self.paragraph_cache.items()
            },
            "paragraph_labels": {
                f"{k[0]}::{k[1]}": v for k, v in self.paragraph_labels.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _save(self):
        """
        Persist the index and metadata together. Both are written to temp
        files first; only when both writes succeed are they renamed into
        place, back to back, so a failed write (e.g. disk full) leaves the
        previous pair untouched.
        """
        index_tmp = f"{self.index_path}.tmp"
        metadata_tmp = f"{self.metadata_path}.tmp"
        try:
            faiss.write_index(self.index, index_tmp)
            self._write_metadata(metadata_tmp)
        except Exception:
            for tmp in (index_tmp, metadata_tmp):
                if os.path.exists(tmp):
                    os.remove(tmp)
            raise
        os.replace(index_tmp, self.index_path)
        os.replace(metadata_tmp, self.metadata_path)

    @_locked
    @_transactional
    def index_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        sentences: List[str],
        paragraph_indices: List[int],
        embeddings: List[List[float]],
        author: Optional[str] = None,
        paragraph_labels: Optional[Dict[int, str]] = None,
    ) -> bool:
        """
        Index a document with its sentences and embeddings.

        Args:
            doc_id: Unique document identifier
            filename: Original filename
            file_type: File extension
            sentences: List of sentence strings
            paragraph_indices: Paragraph index for each sentence
            embeddings: Embedding vectors for each sentence
            author: Author name (optional)
            paragraph_labels: Section label per paragraph index, e.g.
                {12: "Book 2, §3"} (optional)

        Returns:
            True if successful
        """
        # Check if document already exists
        if doc_id in self.documents:
            return False
        if not sentences:
            raise ValueError("Document has no sentences to index")
        if not (len(sentences) == len(paragraph_indices) == len(embeddings)):
            raise ValueError(
                "sentences, paragraph_indices and embeddings must have the same length"
            )
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if embeddings_array.ndim != 2 or embeddings_array.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.embedding_dim}), "
                f"got {embeddings_array.shape}"
            )

        # Add to FAISS first: if this fails, no metadata has been changed and the
        # FAISS ids stay aligned with self.sentences
        self.index.add(embeddings_array)

        # Store document metadata
        total_paragraphs = max(paragraph_indices) + 1 if paragraph_indices else 0
        self.documents[doc_id] = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            total_sentences=len(sentences),
            total_paragraphs=total_paragraphs,
            author=author,
        )

        # Build paragraph cache (group sentences by paragraph)
        paragraph_texts: Dict[int, List[str]] = {}
        for sent, para_idx in zip(sentences, paragraph_indices):
            if para_idx not in paragraph_texts:
                paragraph_texts[para_idx] = []
            paragraph_texts[para_idx].append(sent)

        for para_idx, para_sentences in paragraph_texts.items():
            self.paragraph_cache[(doc_id, para_idx)] = " ".join(para_sentences)
        for para_idx, label in (paragraph_labels or {}).items():
            self.paragraph_labels[(doc_id, para_idx)] = label

        # Add sentence metadata
        for idx, (sentence, para_idx) in enumerate(zip(sentences, paragraph_indices)):
            self.sentences.append(
                SentenceMetadata(
                    doc_id=doc_id,
                    sentence_idx=idx,
                    paragraph_idx=para_idx,
                    sentence_text=sentence,
                )
            )

        self._rebuild_paragraph_lookup()

        # Persist to disk
        self._save()

        return True

    @_locked
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar sentences, returning unique paragraphs.

        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top paragraphs to find
            doc_ids: Only search these documents (None = all)

        Returns:
            List of search results, one per unique paragraph
        """
        # Search FAISS index - get more results to ensure enough unique paragraphs
        found = self._faiss_search(query_embedding, top_k * 5, doc_ids)
        if found is None:
            return []
        distances, indices = found

        # Group by paragraph, keep best score (and its sentence) per paragraph
        seen_paragraphs: Dict[Tuple[str, int], float] = {}
        best_sentence: Dict[Tuple[str, int], int] = {}

        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            sentence_meta = self.sentences[idx]
            key = (sentence_meta.doc_id, sentence_meta.paragraph_idx)
            similarity = 1.0 / (1.0 + distance)

            if key not in seen_paragraphs or similarity > seen_paragraphs[key]:
                seen_paragraphs[key] = similarity
                best_sentence[key] = int(idx)

        # Build results sorted by similarity
        results = []
        sorted_paragraphs = sorted(
            seen_paragraphs.items(), key=lambda x: x[1], reverse=True
        )

        for (doc_id, para_idx), similarity in sorted_paragraphs[:top_k]:
            doc_meta = self.documents.get(doc_id)
            if not doc_meta:
                continue

            paragraph_text = self.paragraph_cache.get((doc_id, para_idx), "")

            span = self._excerpt_span(best_sentence[(doc_id, para_idx)])
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    filename=doc_meta.filename,
                    paragraph_idx=para_idx,
                    paragraph_text=paragraph_text,
                    author=doc_meta.author,
                    excerpt_text=self._excerpt_text(span),
                    section_label=self.paragraph_labels.get((doc_id, para_idx)),
                )
            )

        return results

    @_locked
    @_transactional
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index.
        Note: FAISS doesn't support deletion, so we rebuild the index.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        if doc_id not in self.documents:
            return False

        # Remove document metadata
        del self.documents[doc_id]

        # Find indices to keep
        indices_to_keep = [
            i for i, sent in enumerate(self.sentences) if sent.doc_id != doc_id
        ]

        # Rebuild index with remaining embeddings
        if indices_to_keep:
            # Extract embeddings to keep
            old_index = self.index
            embeddings_to_keep = []

            for idx in indices_to_keep:
                # Reconstruct vector from index
                vector = old_index.reconstruct(int(idx))
                embeddings_to_keep.append(vector)

            # Create new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if embeddings_to_keep:
                embeddings_array = np.array(embeddings_to_keep, dtype=np.float32)
                self.index.add(embeddings_array)

            # Update sentences list
            self.sentences = [self.sentences[i] for i in indices_to_keep]
        else:
            # No documents left, create empty index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.sentences = []

        # Clean up paragraph cache and labels
        self.paragraph_cache = {
            k: v for k, v in self.paragraph_cache.items() if k[0] != doc_id
        }
        self.paragraph_labels = {
            k: v for k, v in self.paragraph_labels.items() if k[0] != doc_id
        }
        self._rebuild_paragraph_lookup()

        # Persist changes
        self._save()

        return True

    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all indexed documents.

        Returns:
            List of document metadata
        """
        return list(self.documents.values())

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Get metadata for a specific document.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        return self.documents.get(doc_id)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_documents": len(self.documents),
            "total_sentences": self.index.ntotal,
            "total_paragraphs": len(self.paragraph_cache),
            "embedding_dimension": self.embedding_dim,
        }

    def get_paragraph_with_context(
        self, doc_id: str, para_idx: int, context_window: int = 2
    ) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Get a paragraph with surrounding context paragraphs.

        Args:
            doc_id: Document ID
            para_idx: Paragraph index
            context_window: Number of paragraphs before and after to include (default: 2)

        Returns:
            Tuple of (paragraph_text, paragraphs_before, paragraphs_after)
            Returns (None, [], []) if paragraph not found
        """
        # Get the main paragraph
        main_paragraph = self.paragraph_cache.get((doc_id, para_idx))
        if not main_paragraph:
            return (None, [], [])

        # Get document metadata to know total paragraphs
        doc_meta = self.documents.get(doc_id)
        if not doc_meta:
            return (main_paragraph, [], [])

        # Get context before
        paragraphs_before = []
        for i in range(max(0, para_idx - context_window), para_idx):
            para = self.paragraph_cache.get((doc_id, i))
            if para:
                paragraphs_before.append(para)

        # Get context after
        paragraphs_after = []
        for i in range(
            para_idx + 1, min(doc_meta.total_paragraphs, para_idx + context_window + 1)
        ):
            para = self.paragraph_cache.get((doc_id, i))
            if para:
                paragraphs_after.append(para)

        return (main_paragraph, paragraphs_before, paragraphs_after)

    def search_with_reranking(
        self,
        query_text: str,
        query_embedding: List[float],
        reranker: "CrossEncoderReranker",
        top_k_faiss: int = 50,
        top_k_paragraphs: int = 5,
        context_window: int = 2,
        doc_ids: Optional[List[str]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Two-stage retrieval: FAISS candidate retrieval + cross-encoder reranking.

        Args:
            query_text: Search query text (needed for cross-encoder)
            query_embedding: Query embedding vector (for FAISS stage)
            reranker: CrossEncoderReranker instance
            top_k_faiss: Number of sentence candidates from FAISS (default: 50)
            top_k_paragraphs: Number of passages to return, one citation each (default: 5)
            context_window: Number of paragraphs before/after to include (default: 2)
            doc_ids: Only search these documents (None = all)

        Each matched sentence yields a few short excerpt windows around it (see
        _candidate_spans); the cross-encoder scores those excerpts, and each
        passage is cited once, by its best-scoring excerpt.

        Returns:
            Tuple of (search_results, timing_info) where:
            - search_results: List of reranked SearchResult objects with context
            - timing_info: Dict with 'faiss_time', 'reranking_time', 'total_time'
        """
        total_start = time.time()

        # Stage 1 (under the lock): FAISS retrieval, candidate excerpts, and a
        # snapshot of everything needed to build results. The cross-encoder
        # then runs without the lock, so concurrent searches don't queue
        # behind each other and an index change can't alter the snapshot.
        faiss_start = time.time()
        with self._lock:
            found = self._faiss_search(query_embedding, top_k_faiss, doc_ids)
            if found is None:
                return ([], {"faiss_time": 0.0, "reranking_time": 0.0, "total_time": 0.0})
            distances, indices = found

            # Candidate citation excerpts around each matched sentence. Matches
            # in the same passage can produce the same window; score it once.
            candidates: List[Tuple[Tuple[str, int], str]] = []  # (passage, excerpt)
            seen_spans = set()
            passages: Dict[Tuple[str, int], dict] = {}
            for idx in indices[0]:
                if idx == -1:
                    continue
                for span in self._candidate_spans(int(idx)):
                    if span in seen_spans:
                        continue
                    seen_spans.add(span)
                    meta = self.sentences[span[0]]
                    key = (meta.doc_id, meta.paragraph_idx)
                    candidates.append((key, self._excerpt_text(span)))
                    if key not in passages:
                        doc_meta = self.documents.get(meta.doc_id)
                        if not doc_meta:
                            continue
                        _, before, after = self.get_paragraph_with_context(
                            meta.doc_id, meta.paragraph_idx, context_window
                        )
                        passages[key] = {
                            "filename": doc_meta.filename,
                            "author": doc_meta.author,
                            "text": self.paragraph_cache.get(key, ""),
                            "label": self.paragraph_labels.get(key),
                            "before": before,
                            "after": after,
                        }

        faiss_time = time.time() - faiss_start

        if not candidates:
            return (
                [],
                {
                    "faiss_time": faiss_time,
                    "reranking_time": 0.0,
                    "total_time": time.time() - total_start,
                },
            )

        # Stage 2 (no lock): cross-encoder reranking of the excerpts users see
        scores, reranking_time = reranker.score_texts(
            query_text, [excerpt for _, excerpt in candidates]
        )
        ranked = sorted(
            zip(candidates, scores), key=lambda item: item[1], reverse=True
        )

        # One citation per passage: its best-scoring excerpt
        results = []
        cited_paragraphs = set()
        for (key, excerpt), score in ranked:
            info = passages.get(key)
            if info is None or key in cited_paragraphs:
                continue
            cited_paragraphs.add(key)
            results.append(
                SearchResult(
                    doc_id=key[0],
                    filename=info["filename"],
                    paragraph_idx=key[1],
                    paragraph_text=info["text"] or excerpt,
                    author=info["author"],
                    reranking_score=float(score),
                    context_paragraphs_before=info["before"],
                    context_paragraphs_after=info["after"],
                    excerpt_text=excerpt,
                    section_label=info["label"],
                )
            )
            if len(results) >= top_k_paragraphs:
                break

        total_time = time.time() - total_start

        timing_info = {
            "faiss_time": faiss_time,
            "reranking_time": reranking_time,
            "total_time": total_time,
        }

        return (results, timing_info)
