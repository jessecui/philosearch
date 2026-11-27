from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Header,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import hmac
import logging
import os
import uuid
import json
from pathlib import Path
from typing import Optional, AsyncGenerator
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send
from app.services.embedding_service import EmbeddingService
from app.utils.document_processor import DocumentExtractionError, DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.index_sync import download_index_if_missing
from app.services.distributed_ingestion import DistributedIngestionPipeline
from app.services.reranker_service import CrossEncoderReranker
from app.services.generation_service import GeminiGenerator
from app.auth import Gate
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(
    title="Document Embedding API",
    docs_url=None,  # Disable /docs
    redoc_url=None,  # Disable /redoc
    openapi_url=None,  # Disable /openapi.json
)

logger = logging.getLogger("philosearch")


def _generation_error_message(e: Exception) -> str:
    """User-facing message for a failed answer stream (details go to the log)."""
    logger.error("Generation failed: %s", e)
    text = str(e)
    if any(code in text for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
        return "The AI model is busy right now. Please try again in a moment."
    return "Something went wrong while generating the answer. Please try again."


def _internal_error(e: Exception) -> HTTPException:
    """Log the real error server-side; give the client a generic message."""
    logger.exception("Request failed: %s", e)
    return HTTPException(status_code=500, detail="Internal server error. Please try again.")

# Initialize services
# A deployment built from the repository has no index; fetch it if INDEX_REPO
# is set. First, so a bad HF_TOKEN is reported here rather than as a failed
# model download.
download_index_if_missing()
embedding_service = EmbeddingService()
document_processor = DocumentProcessor()
vector_store = VectorStore()
reranker = CrossEncoderReranker()  # Initialize reranker for 2-stage retrieval


# --- Auth ---

# The sign-in gate (auth.py): POST /login and GET /me, the same in Attune and
# MentorML. The signing key is derived from the creator's name and the Gemini
# API key unless AUTH_SECRET is set.
gate = Gate("philosearch", key_source=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""))
app.include_router(gate.router)


def _admin_token_is_valid(supplied: Optional[str]) -> bool:
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token or not supplied:
        return False
    # HTTP headers arrive decoded as latin-1; re-encoding that way recovers the
    # exact bytes the client sent, so a UTF-8 token compares correctly
    try:
        supplied_bytes = supplied.encode("latin-1")
    except UnicodeEncodeError:
        supplied_bytes = supplied.encode("utf-8")
    return hmac.compare_digest(supplied_bytes, admin_token.encode("utf-8"))


async def require_admin(x_admin_token: str | None = Header(default=None)):
    """
    Require the X-Admin-Token header for endpoints that change the corpus or
    accept uploads. Disabled entirely unless ADMIN_TOKEN is set.
    """
    if not os.getenv("ADMIN_TOKEN", ""):
        raise HTTPException(
            status_code=403,
            detail="Document management is disabled. Set ADMIN_TOKEN to enable it.",
        )
    if not _admin_token_is_valid(x_admin_token):
        raise HTTPException(status_code=403, detail="Invalid admin token")


# Admin routes accept uploads. FastAPI reads the whole request body before it
# runs dependencies, so the token and size are checked here, before the body
# is received, to stop anonymous multi-GB uploads.
def _max_upload_mb() -> int:
    raw = os.getenv("MAX_UPLOAD_MB", "").strip()
    if raw.isascii() and raw.isdigit() and int(raw) > 0:
        return int(raw)
    if raw:
        print(f"⚠️  Ignoring invalid MAX_UPLOAD_MB={raw!r}; using 50")
    return 50


MAX_UPLOAD_BYTES = _max_upload_mb() * 1024 * 1024
# The request body also carries multipart boundaries and headers
_MULTIPART_OVERHEAD_BYTES = 64 * 1024
# Every other route takes small JSON bodies; the largest legitimate one is a
# /generate call echoing back search results. Without a cap, FastAPI would
# read an arbitrarily large body into memory before validating it.
MAX_JSON_BODY_BYTES = 4 * 1024 * 1024
_ADMIN_ROUTES = {("POST", "/index"), ("POST", "/index-distributed"), ("POST", "/embed")}


class AdminGateMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            # Match the path the router sees: without any proxy root_path prefix
            path = scope["path"]
            root_path = scope.get("root_path", "")
            if root_path and path.startswith(root_path):
                path = path[len(root_path):]
            method, path = scope["method"], path.rstrip("/") or "/"
            is_admin = (method, path) in _ADMIN_ROUTES or (
                method == "DELETE" and path.startswith("/documents/")
            )
            if is_admin:
                headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
                status, detail = None, None
                if not os.getenv("ADMIN_TOKEN", ""):
                    status, detail = 403, "Document management is disabled. Set ADMIN_TOKEN to enable it."
                elif not _admin_token_is_valid(headers.get("x-admin-token")):
                    status, detail = 403, "Invalid admin token"
                else:
                    length = headers.get("content-length", "")
                    if length.isdigit() and int(length) > MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD_BYTES:
                        status, detail = 413, _too_large_message()
                if status:
                    response = JSONResponse({"detail": detail}, status_code=status)
                    await response(scope, receive, send)
                    return
                # No (or a small) Content-Length, e.g. chunked: count the bytes
                # as they arrive and stop reading once over the limit
                receive = _limited_receive(receive, MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD_BYTES)
            else:
                headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
                length = headers.get("content-length", "")
                if length.isdigit() and int(length) > MAX_JSON_BODY_BYTES:
                    response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                    await response(scope, receive, send)
                    return
                receive = _limited_receive(receive, MAX_JSON_BODY_BYTES)
        await self.app(scope, receive, send)


def _too_large_message() -> str:
    return f"Upload too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"


def _limited_receive(receive: Receive, limit: int) -> Receive:
    received = 0

    async def wrapped():
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(status_code=413, detail="Request body too large")
        return message

    return wrapped


app.add_middleware(AdminGateMiddleware)


# --- Uploads ---

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _upload_name_and_ext(file: UploadFile) -> tuple[str, str]:
    """
    Display name (basename only, never used as a path) and validated extension.
    """
    display_name = os.path.basename((file.filename or "upload").replace("\\", "/"))
    file_ext = os.path.splitext(display_name)[1].lower()
    # Search results echo the name back to /generate, which caps it at 300;
    # a "__Author" suffix is kept, so the author survives the cut
    stem = os.path.splitext(display_name)[0]
    title, sep, author = stem.partition("__")
    room = 300 - len(file_ext) - len(sep) - len(author)
    if room < 1:  # an author name alone over the limit: cut everything
        title, sep, author, room = stem, "", "", 300 - len(file_ext)
    display_name = title[:room] + sep + author + file_ext
    if file_ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )
    return display_name, file_ext


def _author_from_name(display_name: str) -> Optional[str]:
    """"Republic__Plato.txt" -> "Plato", following the Title__Author naming
    that index_texts uses for the library's books; None without "__"."""
    stem = os.path.splitext(display_name)[0]
    if "__" not in stem:
        return None
    return stem.split("__", 1)[1].replace("_", " ").strip() or None


def _new_upload_path(file_ext: str) -> str:
    """A random path inside UPLOAD_DIR (never derived from the client's name)."""
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{file_ext}")


async def _save_upload(file: UploadFile, file_path: str) -> None:
    """Copy an upload to file_path in chunks, enforcing MAX_UPLOAD_BYTES."""
    written = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=_too_large_message())
            buffer.write(chunk)


# Initialize distributed ingestion pipeline only if ENABLE_RAY is set
if os.getenv("ENABLE_RAY", "false").lower() == "true":
    distributed_pipeline = DistributedIngestionPipeline(num_workers=8, batch_size=32)
else:
    print("ℹ️  Ray distributed processing disabled. Set ENABLE_RAY=true to enable.")
    distributed_pipeline = None

# Initialize Gemini generator (only if API key is configured)
try:
    generator = GeminiGenerator()
except Exception as e:
    print(f"⚠️  Gemini generator not initialized: {e}")
    print("   RAG generation endpoints will not be available.")
    print("   Set GEMINI_API_KEY in .env to enable generation features.")
    generator = None

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Request models
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranking: bool = False  # Enable 2-stage retrieval with cross-encoder reranking
    top_k_faiss: int = Field(default=50, ge=1, le=200)  # FAISS candidates (stage 1)
    context_window: int = Field(default=2, ge=0, le=5)  # Paragraphs before/after
    doc_ids: Optional[list[str]] = Field(default=None, max_length=200)  # Books to search (None = all)


class GenerateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k_context: int = Field(default=5, ge=1, le=20)  # Excerpts to retrieve
    use_reranking: bool = True  # Use cross-encoder reranking for better quality
    top_k_faiss: int = Field(default=50, ge=1, le=200)  # FAISS candidates
    context_window: int = Field(default=0, ge=0, le=5)  # Paragraphs before/after
    doc_ids: Optional[list[str]] = Field(default=None, max_length=200)  # Books to search (None = all)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192, ge=1, le=8192)


class ResultItem(BaseModel):
    """One result as returned by /search."""

    doc_id: Optional[str] = None
    filename: str = Field(min_length=1, max_length=300)
    author: Optional[str] = Field(default=None, max_length=300)
    paragraph_idx: int = 0
    paragraph_text: str = Field(max_length=50_000)
    excerpt_text: Optional[str] = Field(default=None, max_length=5_000)
    section_label: Optional[str] = Field(default=None, max_length=300)
    # NaN or infinity would make the SSE sources event invalid JSON
    reranking_score: Optional[float] = Field(default=None, allow_inf_nan=False)
    context_paragraphs_before: Optional[list[str]] = None
    context_paragraphs_after: Optional[list[str]] = None


class GenerateFromResultsRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    results: list[ResultItem] = Field(max_length=50)  # Pre-fetched results from /search
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192, ge=1, le=8192)


STALE_DOC_IDS_DETAIL = "Some selected books are no longer indexed."
NO_BOOKS_SELECTED_DETAIL = "Select at least one book to search."


def _require_known_doc_ids(doc_ids: Optional[list[str]]) -> None:
    """
    Reject a selection that names books no longer in the index (e.g. the page
    was loaded before a re-index), instead of silently searching fewer books.
    The frontend reloads its book list when it sees this detail.
    """
    if doc_ids is not None and not doc_ids:
        raise HTTPException(status_code=404, detail=NO_BOOKS_SELECTED_DETAIL)
    if doc_ids and any(doc_id not in vector_store.documents for doc_id in doc_ids):
        raise HTTPException(status_code=404, detail=STALE_DOC_IDS_DETAIL)


@app.get("/health")
async def health():
    return {"status": "healthy"}


# Streamed answers must reach the browser token by token: no caching, and no
# buffering by reverse proxies such as nginx
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.post("/embed", dependencies=[Depends(require_admin)])
async def embed_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD, or DOCX) and get sentence-level embeddings.

    Returns:
        - filename: Original filename
        - file_type: Type of file uploaded
        - text_length: Number of characters in extracted text
        - sentence_count: Number of sentences extracted
        - paragraph_count: Number of paragraphs detected
        - sentences: List of sentence strings
        - paragraph_indices: List of paragraph indices for each sentence
        - embeddings: List of embedding vectors for each sentence
        - embedding_dimension: Dimension of each embedding vector
    """
    display_name, file_ext = _upload_name_and_ext(file)

    file_path = None
    try:
        # Save uploaded file temporarily
        file_path = _new_upload_path(file_ext)
        await _save_upload(file, file_path)

        # Extract text from document
        text = await run_in_threadpool(
            document_processor.extract_text, file_path, file_ext
        )

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="No text could be extracted from the document"
            )

        # Generate sentence-level embeddings
        sentences, paragraph_indices, embeddings = await run_in_threadpool(
            embedding_service.embed_by_sentence, text, document_processor
        )

        # Calculate paragraph count (max index + 1)
        paragraph_count = max(paragraph_indices) + 1 if paragraph_indices else 0

        return JSONResponse(
            content={
                "filename": display_name,
                "file_type": file_ext,
                "text_length": len(text),
                "sentence_count": len(sentences),
                "paragraph_count": paragraph_count,
                "sentences": sentences,
                "paragraph_indices": paragraph_indices,
                "embeddings": embeddings,
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            }
        )

    except HTTPException:
        raise
    except DocumentExtractionError as e:
        # An unreadable file won't succeed on retry: say why instead of a 500
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise _internal_error(e)
    finally:
        # Always remove the temporary upload
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.post("/index", dependencies=[Depends(require_admin)])
async def index_document(file: UploadFile = File(...)):
    """
    Upload and index a document for semantic search.

    Returns:
        - doc_id: Unique document identifier
        - filename: Original filename
        - file_type: Type of file uploaded
        - sentence_count: Number of sentences indexed
        - paragraph_count: Number of paragraphs detected
    """
    display_name, file_ext = _upload_name_and_ext(file)

    file_path = None
    try:
        # Save uploaded file temporarily
        file_path = _new_upload_path(file_ext)
        await _save_upload(file, file_path)

        # Extract text from document
        text = await run_in_threadpool(
            document_processor.extract_text, file_path, file_ext
        )

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="No text could be extracted from the document"
            )

        # Split into passages (with section labels) and embed each sentence
        parsed = await run_in_threadpool(document_processor.parse_document, text)
        sentences, paragraph_indices = parsed.sentences, parsed.paragraph_indices
        if not sentences:
            raise HTTPException(
                status_code=400, detail="No sentences could be extracted from the document"
            )
        embeddings = await run_in_threadpool(embedding_service.embed_texts, sentences)

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Index in vector store
        success = await run_in_threadpool(
            vector_store.index_document,
            doc_id=doc_id,
            filename=display_name,
            file_type=file_ext,
            sentences=sentences,
            paragraph_indices=paragraph_indices,
            embeddings=embeddings,
            author=_author_from_name(display_name),
            paragraph_labels=parsed.paragraph_labels,
            paragraph_texts=parsed.paragraph_texts,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to index document")

        paragraph_count = max(paragraph_indices) + 1 if paragraph_indices else 0

        return JSONResponse(
            content={
                "doc_id": doc_id,
                "filename": display_name,
                "file_type": file_ext,
                "sentence_count": len(sentences),
                "paragraph_count": paragraph_count,
                "message": "Document indexed successfully",
            }
        )

    except HTTPException:
        raise
    except DocumentExtractionError as e:
        # An unreadable file won't succeed on retry: say why instead of a 500
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise _internal_error(e)
    finally:
        # Always remove the temporary upload
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.post("/index-distributed", dependencies=[Depends(require_admin)])
async def index_document_distributed(file: UploadFile = File(...)):
    """
    Upload and index a document using distributed Ray workers for faster processing.
    Ideal for large documents (1000+ sentences).

    Returns:
        - doc_id: Unique document identifier
        - filename: Original filename
        - file_type: Type of file uploaded
        - sentence_count: Number of sentences indexed
        - paragraph_count: Number of paragraphs detected
        - message: Status message
        - performance: total_time, parsing_time, splitting_time,
          embedding_time and indexing_time (seconds), throughput
          (sentences per second) and workers (Ray workers used)
    """
    # Check if Ray is enabled
    if distributed_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Distributed processing not available. Set ENABLE_RAY=true to enable.",
        )

    display_name, file_ext = _upload_name_and_ext(file)

    file_path = None
    try:
        start_time = time.time()

        # Save uploaded file temporarily
        file_path = _new_upload_path(file_ext)
        await _save_upload(file, file_path)

        # Extract text from document
        parse_start = time.time()
        text = await run_in_threadpool(
            document_processor.extract_text, file_path, file_ext
        )
        parse_time = time.time() - parse_start

        if not text.strip():
            raise HTTPException(
                status_code=400, detail="No text could be extracted from the document"
            )

        # Split into sentences and paragraphs
        split_start = time.time()
        parsed = await run_in_threadpool(document_processor.parse_document, text)
        sentences, paragraph_indices = parsed.sentences, parsed.paragraph_indices
        split_time = time.time() - split_start
        if not sentences:
            raise HTTPException(
                status_code=400, detail="No sentences could be extracted from the document"
            )

        # Generate embeddings (distributed)
        embeddings, metrics = await run_in_threadpool(
            distributed_pipeline.process_document, sentences, paragraph_indices
        )

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Index in vector store
        index_start = time.time()
        success = await run_in_threadpool(
            vector_store.index_document,
            doc_id=doc_id,
            filename=display_name,
            file_type=file_ext,
            sentences=sentences,
            paragraph_indices=paragraph_indices,
            embeddings=embeddings,
            author=_author_from_name(display_name),
            paragraph_labels=parsed.paragraph_labels,
            paragraph_texts=parsed.paragraph_texts,
        )
        index_time = time.time() - index_start

        if not success:
            raise HTTPException(status_code=500, detail="Failed to index document")

        total_time = time.time() - start_time
        paragraph_count = max(paragraph_indices) + 1 if paragraph_indices else 0

        return JSONResponse(
            content={
                "doc_id": doc_id,
                "filename": display_name,
                "file_type": file_ext,
                "sentence_count": len(sentences),
                "paragraph_count": paragraph_count,
                "message": "Document indexed successfully with distributed processing",
                "performance": {
                    "total_time": round(total_time, 2),
                    "parsing_time": round(parse_time, 2),
                    "splitting_time": round(split_time, 2),
                    "embedding_time": round(metrics.embedding_time, 2),
                    "indexing_time": round(index_time, 2),
                    "throughput": round(metrics.sentences_per_second, 1),
                    "workers": distributed_pipeline.num_workers,
                },
            }
        )

    except HTTPException:
        raise
    except DocumentExtractionError as e:
        # An unreadable file won't succeed on retry: say why instead of a 500
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise _internal_error(e)
    finally:
        # Always remove the temporary upload
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.post("/search", dependencies=[Depends(gate.require)])
async def search_documents(request: SearchRequest):
    """
    Search indexed documents for semantically similar passages.

    Args:
        query: Search query text
        top_k: Number of final results to return (default: 5)
        use_reranking: Enable 2-stage retrieval with cross-encoder reranking (default: False)
        top_k_faiss: Number of candidates from FAISS stage (default: 50, only used if use_reranking=True)
        context_window: Number of paragraphs before/after for context expansion
            (default: 2, only used if use_reranking=True)

    Returns:
        List of search results, one per passage, with matched sentences and
        paragraph context
        If use_reranking=True, also includes reranking_score and timing breakdown
    """
    _require_known_doc_ids(request.doc_ids)
    try:
        # Generate query embedding
        query_embedding = (
            await run_in_threadpool(embedding_service.embed_texts, [request.query])
        )[0]

        # Choose retrieval method
        if request.use_reranking:
            # 2-stage retrieval: FAISS + cross-encoder reranking
            results, timing = await run_in_threadpool(
                vector_store.search_with_reranking,
                query_text=request.query,
                query_embedding=query_embedding,
                reranker=reranker,
                top_k_faiss=request.top_k_faiss,
                top_k_paragraphs=request.top_k,
                context_window=request.context_window,
                doc_ids=request.doc_ids,
            )

            # Convert results to JSON-serializable format (with reranking data)
            response_results = []
            for result in results:
                result_dict = {
                    "doc_id": result.doc_id,
                    "filename": result.filename,
                    "author": result.author,
                    "paragraph_idx": result.paragraph_idx,
                    "paragraph_text": result.paragraph_text,
                    "excerpt_text": result.excerpt_text,
                    "section_label": result.section_label,
                    "reranking_score": (
                        float(result.reranking_score)
                        if result.reranking_score is not None
                        else None
                    ),
                }

                # Add context paragraphs if present
                if result.context_paragraphs_before:
                    result_dict["context_paragraphs_before"] = (
                        result.context_paragraphs_before
                    )

                if result.context_paragraphs_after:
                    result_dict["context_paragraphs_after"] = (
                        result.context_paragraphs_after
                    )

                response_results.append(result_dict)

            return JSONResponse(
                content={
                    "query": request.query,
                    "results": response_results,
                    "total_results": len(response_results),
                    "timing": {
                        "faiss_time": round(timing["faiss_time"], 3),
                        "reranking_time": round(timing["reranking_time"], 3),
                        "total_time": round(timing["total_time"], 3),
                    },
                    "retrieval_method": "faiss_with_reranking",
                }
            )

        else:
            # FAISS-only retrieval
            results = await run_in_threadpool(
                vector_store.search,
                query_embedding=query_embedding,
                top_k=request.top_k,
                doc_ids=request.doc_ids,
            )

            # Convert results to JSON-serializable format
            response_results = []
            for result in results:
                response_results.append(
                    {
                        "doc_id": result.doc_id,
                        "filename": result.filename,
                        "author": result.author,
                        "paragraph_idx": result.paragraph_idx,
                        "paragraph_text": result.paragraph_text,
                        "excerpt_text": result.excerpt_text,
                        "section_label": result.section_label,
                    }
                )

            return JSONResponse(
                content={
                    "query": request.query,
                    "results": response_results,
                    "total_results": len(response_results),
                    "retrieval_method": "faiss_only",
                }
            )

    except Exception as e:
        raise _internal_error(e)


@app.get("/documents")
async def list_documents():
    """
    List all indexed documents.

    Returns:
        List of document metadata
    """
    try:
        documents = vector_store.list_documents()

        return JSONResponse(
            content={
                "documents": [
                    {
                        "doc_id": doc.doc_id,
                        "filename": doc.filename,
                        "author": doc.author,
                        "file_type": doc.file_type,
                        "total_sentences": doc.total_sentences,
                        "total_paragraphs": doc.total_paragraphs,
                    }
                    for doc in documents
                ],
                "total_documents": len(documents),
            }
        )

    except Exception as e:
        raise _internal_error(e)


@app.delete("/documents/{doc_id}", dependencies=[Depends(require_admin)])
async def delete_document(doc_id: str):
    """
    Delete a document from the index.

    Args:
        doc_id: Document ID to delete

    Returns:
        Success message
    """
    try:
        success = await run_in_threadpool(vector_store.delete_document, doc_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Document with ID {doc_id} not found"
            )

        return JSONResponse(
            content={"message": "Document deleted successfully", "doc_id": doc_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(e)


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the vector store.

    Returns:
        Statistics including total documents, sentences, and paragraphs
    """
    try:
        stats = vector_store.get_stats()
        return JSONResponse(content=stats)

    except Exception as e:
        raise _internal_error(e)


@app.post("/generate", dependencies=[Depends(gate.require)])
async def generate_from_results(request: GenerateFromResultsRequest):
    """
    Generate an answer from pre-fetched search results.

    This endpoint enables a two-step RAG workflow:
    1. User calls /search to get and review results
    2. User clicks a button to generate an answer from those results

    Args:
        query: User's question
        results: List of search results from /search endpoint
        temperature: Generation temperature (default: 0.7)
        max_output_tokens: Maximum response length (default: 8192)

    Returns:
        Server-Sent Events stream with:
        - sources: Initial event with the provided search results
        - token: Streaming text chunks from the model
        - done: Final event with timing information
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG generation not available. Set GEMINI_API_KEY in backend/.env "
            "and restart the server.",
        )

    try:
        if not request.results:
            raise HTTPException(
                status_code=400,
                detail="No search results provided. Please provide results from /search endpoint.",
            )

        overall_start = time.time()

        # Convert the JSON results back to SearchResult objects
        from app.services.vector_store import SearchResult

        search_results = []
        for r in request.results:
            search_results.append(SearchResult(**r.model_dump()))

        # Stream generated answer
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events for streaming response."""
            try:
                # Send initial sources event
                sources_data = [
                    {
                        "filename": r.filename,
                        "author": r.author,
                        "paragraph_idx": r.paragraph_idx,
                        "section_label": r.section_label,
                        "excerpt": r.excerpt_text or r.paragraph_text,
                        "text": r.paragraph_text,
                        "score": (
                            float(r.reranking_score)
                            if r.reranking_score is not None
                            else None
                        ),
                    }
                    for r in search_results
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Stream answer tokens
                generation_start = time.time()
                async for chunk in generator.stream_answer(
                    query=request.query,
                    search_results=search_results,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"

                generation_time = time.time() - generation_start
                total_time = time.time() - overall_start

                # Send completion event with timing
                timing_data = {
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                }
                yield f"data: {json.dumps({'type': 'done', 'data': timing_data})}\n\n"

            except Exception as e:
                error_data = {"message": _generation_error_message(e), "code": "generation_error"}
                yield f"data: {json.dumps({'type': 'error', 'data': error_data})}\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(e)


@app.post("/search-and-generate", dependencies=[Depends(gate.require)])
async def search_and_generate(request: GenerateRequest):
    """
    Retrieve relevant document excerpts and generate an answer using Gemini 3.8 Flash.

    This endpoint implements RAG (Retrieval-Augmented Generation):
    1. Embeds the query
    2. Retrieves top-k relevant passages with optional reranking
    3. Streams an AI-generated answer grounded in the retrieved context

    Args:
        query: User's question
        top_k_context: Number of document excerpts to retrieve (default: 5)
        use_reranking: Enable cross-encoder reranking (default: True)
        top_k_faiss: FAISS candidates for reranking (default: 50)
        context_window: Paragraphs before/after for context (default: 0)
        temperature: Generation temperature (default: 0.7)
        max_output_tokens: Maximum response length (default: 8192)

    Returns:
        Server-Sent Events stream with:
        - sources: Initial event with retrieved document excerpts
        - token: Streaming text chunks from the model
        - done: Final event with timing information
    """
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG generation not available. Set GEMINI_API_KEY in backend/.env "
            "and restart the server.",
        )
    _require_known_doc_ids(request.doc_ids)

    try:
        overall_start = time.time()

        # Step 1: Generate query embedding
        embed_start = time.time()
        query_embedding = (
            await run_in_threadpool(embedding_service.embed_texts, [request.query])
        )[0]
        embed_time = time.time() - embed_start

        # Step 2: Retrieve relevant passages
        retrieval_start = time.time()
        if request.use_reranking:
            results, timing = await run_in_threadpool(
                vector_store.search_with_reranking,
                query_text=request.query,
                query_embedding=query_embedding,
                reranker=reranker,
                top_k_faiss=request.top_k_faiss,
                top_k_paragraphs=request.top_k_context,
                context_window=request.context_window,
                doc_ids=request.doc_ids,
            )
            retrieval_time = timing["total_time"]
        else:
            results = await run_in_threadpool(
                vector_store.search,
                query_embedding=query_embedding,
                top_k=request.top_k_context,
                doc_ids=request.doc_ids,
            )
            retrieval_time = time.time() - retrieval_start
            timing = {"faiss_time": retrieval_time, "reranking_time": 0}

        if not results:
            # (an empty doc_ids list is already rejected by _require_known_doc_ids)
            if request.doc_ids is not None:
                detail = "No matching passages in the selected books."
            else:
                detail = "No relevant documents found. Please index documents first."
            raise HTTPException(status_code=404, detail=detail)

        # Step 3: Stream generated answer
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate Server-Sent Events for streaming response."""
            try:
                # Send initial sources event
                sources_data = [
                    {
                        "filename": r.filename,
                        "author": r.author,
                        "paragraph_idx": r.paragraph_idx,
                        "section_label": r.section_label,
                        "excerpt": r.excerpt_text or r.paragraph_text,
                        "text": r.paragraph_text,
                        "score": (
                            float(r.reranking_score)
                            if r.reranking_score is not None
                            else None
                        ),
                    }
                    for r in results
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Stream answer tokens
                generation_start = time.time()
                async for chunk in generator.stream_answer(
                    query=request.query,
                    search_results=results,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"

                generation_time = time.time() - generation_start
                total_time = time.time() - overall_start

                # Send completion event with timing
                timing_data = {
                    "embedding_time": round(embed_time, 3),
                    "faiss_time": round(timing["faiss_time"], 3),
                    "reranking_time": round(timing["reranking_time"], 3),
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                }
                yield f"data: {json.dumps({'type': 'done', 'data': timing_data})}\n\n"

            except Exception as e:
                error_data = {"message": _generation_error_message(e), "code": "generation_error"}
                yield f"data: {json.dumps({'type': 'error', 'data': error_data})}\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(e)


# --- Serve React Frontend ---

# Mount static assets if frontend is built
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )


@app.get("/", summary="Health check / Frontend", include_in_schema=False)
async def root():
    """Serve frontend index.html in production, health check in dev."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "healthy",
        "message": "API running. Build frontend with: cd frontend && npm run build",
    }


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Serve index.html for any unmatched routes (SPA client-side routing)."""
    # First check if it's a static file inside dist (favicon.svg, etc.). Resolve
    # the path and refuse anything outside dist ("../../backend/.env").
    dist_root = FRONTEND_DIST.resolve()
    try:
        file_path = (dist_root / path).resolve()
        is_static_file = file_path.is_relative_to(dist_root) and file_path.is_file()
    except (ValueError, OSError):  # e.g. a null byte in the URL
        is_static_file = False
    if is_static_file:
        return FileResponse(file_path)
    # Otherwise serve index.html for client-side routing
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
