# PhiloSearch

A semantic search and RAG (Retrieval-Augmented Generation) platform for philosophical texts, featuring a Vite + React frontend and FastAPI backend with 2-stage retrieval and AI-powered answers.

## Features

- 🔍 **Semantic Search**: Sentence embeddings with all-MiniLM-L6-v2 (384-dim)
- 📚 **Book Selection**: Choose which works to search from the Library panel
- 🎯 **2-Stage Retrieval**: Parent-child FAISS + cross-encoder reranking (nDCG@10 +27.6% over flat search, 88% vs 50% accuracy@1)
- 🤖 **RAG Generation**: Gemini 3.8 Flash streaming answers with source citations
- 📄 **Multi-format Support**: PDF, TXT, MD, DOCX
- ⚡ **Ray Distributed Processing**: 3.74x faster embedding than single-threaded CPU (10 workers)
- 📌 **Structured Citations**: Short 1–3 sentence excerpts labelled by book and section (e.g. "Meditations, Book 6, §49")
- 🚀 **Automatic Device Detection**: MPS/CUDA/CPU

## Architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "-apple-system, Segoe UI, Helvetica, Arial, sans-serif", "fontSize": "14px", "lineColor": "#94a3b8", "edgeLabelBackground": "#ffffff", "clusterBkg": "#f8fafc", "clusterBorder": "#cbd5e1", "titleColor": "#334155"}, "flowchart": {"curve": "basis", "nodeSpacing": 40, "rankSpacing": 44, "padding": 14}}}%%
flowchart TB
    UI(["<b>Browser</b><br/>question + selected books"])

    subgraph IDX["INDEXING · OFFLINE"]
        T["Source texts<br/><small>TXT · PDF · MD · DOCX</small>"] --> P["Parse passages<br/><small>section labels, footnotes dropped</small>"]
        P --> S["Split sentences<br/><small>NLTK</small>"]
        S --> E1["Embed sentences<br/><small>all-MiniLM-L6-v2</small>"]
    end

    STORE[("<b>Vector store</b><br/><small>FAISS sentence vectors<br/>passages · books · labels</small>")]

    subgraph API["FASTAPI · /search-and-generate"]
        A["Verify session"] --> Q["Embed question<br/><small>all-MiniLM-L6-v2</small>"]
        Q --> K["Retrieve top 30 sentences<br/><small>within selected books</small>"]
        K --> X["Build excerpts<br/><small>matched sentence + neighbours</small>"]
        X --> R["Rerank excerpts<br/><small>ms-marco-MiniLM-L-6-v2</small>"]
        R --> G["Assemble prompt<br/><small>excerpts to cite, passages as context</small>"]
    end

    LLM["Generate answer<br/><small>Gemini 3.8 Flash</small>"]
    OUT(["<b>Browser</b><br/>streamed answer + citations"])

    UI --> A
    E1 --> STORE
    K -.- STORE
    X -.- STORE
    G --> LLM
    R -- "sources" --> OUT
    LLM -- "tokens" --> OUT

    classDef step fill:#ffffff,stroke:#cbd5e1,stroke-width:1px,color:#0f172a
    classDef model fill:#eff6ff,stroke:#1d4ed8,stroke-width:1.5px,color:#0f172a
    classDef store fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,color:#0f172a
    classDef client fill:#1d4ed8,stroke:#1d4ed8,color:#ffffff

    class T,P,S,A,K,X,G step
    class E1,Q,R,LLM model
    class STORE store
    class UI,OUT client
    linkStyle default stroke:#94a3b8,stroke-width:1.5px
```

## Tech Stack

- **Frontend**: Vite, React 19, TypeScript, Tailwind CSS v4
- **Backend**: FastAPI, Python
- **Vector Store**: FAISS with persistent storage
- **AI/ML**:
  - Embeddings: all-MiniLM-L6-v2 (384-dim, bi-encoder)
  - Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Generation: Gemini 3.8 Flash (1M token context, streaming), falling back to Gemini 3.6 Flash and then Gemini 3.5 Flash when a model is overloaded (503), rate-limited or out of quota (429), unavailable (404), or the connection drops
- **Distributed Processing**: Ray

---

## Quick Start

### Prerequisites

- Node.js 20.19+ or 22.12+ (required by Vite 7)
- Python 3.10+
- A Google AI Studio API key (for Gemini): https://aistudio.google.com/apikey

### 1. Clone & Install

```bash
git clone https://github.com/jessecui/philosearch.git
cd philosearch

# Frontend dependencies
cd frontend
npm install
cd ..

# Backend dependencies
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `backend/.env.example` to `backend/.env` and fill it in:

```bash
GOOGLE_API_KEY=your-google-ai-api-key   # Gemini
CREATOR_NAME=your-name                  # the name the access prompt asks for
# Optional:
# AUTH_SECRET=...     # signs session cookies (derived from the two above if unset)
# ADMIN_TOKEN=...     # enables upload/delete endpoints (disabled if unset)
# MAX_UPLOAD_MB=50    # upload size limit for those endpoints, per file (MiB)
# ALLOWED_ORIGINS=... # browser origins allowed to call the public endpoints (default: http://localhost:5173)
# ENABLE_RAY=true     # enables /index-distributed
```

### 3. Index the Texts

Neither the texts nor the index are committed. Add the texts to `backend/texts/` under the filenames listed in [Texts](#texts), then build the FAISS vector store:

```bash
# From backend/ directory
python -m app.scripts.index_texts
```

This will generate `backend/data/faiss.index` and `backend/data/metadata.json`. The new index is built in a staging folder and only replaces the old one if every file indexes successfully. Restart the server afterwards so it loads the new index.

### 4. Run the Application

```bash
# Terminal 1: Start backend (from backend/)
uvicorn app.main:app --reload
# API at http://localhost:8000

# Terminal 2: Start frontend (from frontend/)
npm run dev
# App at http://localhost:5173
```

In development the frontend calls the API through Vite's `/api` proxy (see `frontend/vite.config.ts`). In production, build the frontend (`npm run build`) and the backend serves it from `frontend/dist`. The frontend and API must share an origin: the session cookie is `SameSite=Lax`, so a frontend hosted on a separate domain cannot authenticate. `VITE_API_URL` is only for a same-site path such as a reverse-proxy prefix.

---

## API Endpoints

| Endpoint | Method | Access | Description |
|----------|--------|--------|-------------|
| `/` | GET | Public | Serves the built frontend (or a health message if `frontend/dist` is missing) |
| `/health` | GET | Public | Health check |
| `/validate-creator` | POST | Public | Checks the access answer; on success sets the `philosearch_session` cookie |
| `/search` | POST | Public | Semantic search with optional reranking |
| `/search-and-generate` | POST | Session | RAG: retrieve and stream a generated answer |
| `/generate` | POST | Session | Stream an answer from results returned by `/search` |
| `/documents` | GET | Public | List indexed documents |
| `/stats` | GET | Public | Vector store statistics |
| `/index` | POST | Admin | Index an uploaded document |
| `/index-distributed` | POST | Admin | Index an uploaded document with Ray (`ENABLE_RAY=true`) |
| `/embed` | POST | Admin | Return sentence embeddings for an uploaded document |
| `/documents/{doc_id}` | DELETE | Admin | Remove a document |

**Session:** an HttpOnly, signed cookie issued by `/validate-creator`, valid for 24 hours. **Admin:** send `X-Admin-Token: <ADMIN_TOKEN>`; these endpoints return 403 when `ADMIN_TOKEN` is not set.

### Search Parameters (`/search`)

- `query` (required): Search text
- `use_reranking` (default: false): Enable 2-stage retrieval
- `top_k` (default: 5, max 50): Results to return
- `top_k_faiss` (default: 30, max 200): FAISS sentence candidates for reranking
- `context_window` (default: 2): Paragraphs before/after to include in each result (only with `use_reranking: true`)
- `doc_ids` (optional): Only search these documents (ids from `GET /documents`); omit to search all. Unknown ids return 404 "Some selected books are no longer indexed."; an empty list returns 404 "Select at least one book to search."

### Generation Parameters (`/search-and-generate`)

- `query` (required)
- `top_k_context` (default: 5, max 20): Passages to cite (the web app uses 10)
- `use_reranking` (default: true), `top_k_faiss` (default: 30)
- `temperature` (default: 0.7), `max_output_tokens` (default: 8192, max 8192)
- `doc_ids` (optional): Only search these documents. The web app sends the books ticked in its Library panel, and omits the field when every book is ticked. Unknown ids return 404 "Some selected books are no longer indexed."; an empty list returns 404 "Select at least one book to search."

---

## Usage Examples

### Index a Document

```bash
curl -X POST "http://localhost:8000/index" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -F "file=@document.pdf"
```

### Search with Reranking

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is consciousness?",
    "use_reranking": true,
    "top_k": 10
  }'
```

### RAG: Search & Generate Answer

```bash
# 1. Get a session cookie
curl -c cookies.txt -X POST "http://localhost:8000/validate-creator" \
  -H "Content-Type: application/json" -d '{"answer": "your-name"}'

# 2. Stream an answer
curl -N -b cookies.txt -X POST "http://localhost:8000/search-and-generate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the nature of the Tao?",
    "top_k_context": 5
  }'
```

### Test Scripts

```bash
# Test retrieval pipeline
python -m app.scripts.test_search "What does the Tao say about water?"

# Test RAG pipeline with pretty output
python -m app.scripts.test_search_and_generate "How should one cultivate virtue?"
```

---

## RAG Response Format

Server-Sent Events (SSE) streaming:

```
// Initial: Retrieved sources
data: {"type":"sources","data":[{"filename":"Tao_Te_Ching__Lao_Tzu.txt","author":"Lao Tzu","paragraph_idx":7,"section_label":"Chapter 8","excerpt":"The supreme good is like water, ...","text":"<full passage>","score":0.97}]}

// Streaming: Tokens as they're generated
data: {"type":"token","data":"The "}
data: {"type":"token","data":"Tao "}

// Final: Timing metrics (/search-and-generate also reports embedding, FAISS and reranking times)
data: {"type":"done","data":{"embedding_time":0.01,"faiss_time":0.002,"reranking_time":0.13,"retrieval_time":0.14,"generation_time":2.34,"total_time":2.52}}

// On failure, instead of "done"
data: {"type":"error","data":{"message":"...","code":"generation_error"}}
```

---

## Performance Benchmarks

### Retrieval Quality

Tested on 50 queries with graded relevance labels over a synthetic 100-paragraph document: 50 closely related utilitarianism variants (the query targets) plus 50 paragraphs on neighbouring topics such as deontology, Kant and Stoicism as distractors:

| Metric | Flat search | Parent-child FAISS | Parent-child + Reranking | vs Flat | vs Parent-child |
|--------|------------|--------------------|--------------------------|---------|-----------------|
| **Accuracy@1** | 50% | 62% | 88% | **+38pp** | **+26pp** |
| **nDCG@10** | 0.704 | 0.760 | 0.899 | **+27.6%** | **+18.3%** |
| **MRR** | 0.659 | 0.740 | 0.930 | **+41.0%** | **+25.6%** |

Reranking adds roughly 30 ms per query on Apple MPS in this eval (see the "Reranked … excerpts in …s" lines in `eval_faiss_cross_encoder_ndcg_output.txt`) (about 0.05–0.5 s on the full corpus, depending on how many candidate excerpts are scored).

### When to Use Reranking

**Use `use_reranking=true` for:**
- User-facing search (quality critical)
- Top-k precision requirements
- Query time up to ~0.5 s acceptable

**Use `use_reranking=false` for:**
- Real-time applications (tens of milliseconds)
- Broad recall needed
- Resource-constrained environments

### Ray Distributed Processing

Embedding 19,900 sentences with all-MiniLM-L6-v2 on a 12-core Mac (CPU unless noted):

| Method | Time | Throughput | vs single-threaded |
|--------|------|------------|--------------------|
| **Single-threaded CPU** (benchmark baseline) | 29.77s | 669 sent/s | 1.0x |
| **Multithreaded CPU** (PyTorch default, no Ray) | 11.55s | 1,723 sent/s | 2.6x |
| **Ray, 10 CPU workers** (excluding 6.6s worker startup) | 7.95s | 2,503 sent/s | **3.74x** (−73.3% latency) |
| **Ray, 10 CPU workers** (including startup) | 14.57s | 1,366 sent/s | 2.04x (−51.1% latency) |
| **Apple GPU (MPS), no Ray** | 5.80s | 3,431 sent/s | 5.1x |

The Ray rows are the median of three runs (latency reduction 72.8%, 73.3%, 74.3%); results vary by about ±1 point between runs. Ray pays off for large CPU-only batches once its workers are running; on a machine with a GPU, a single GPU process is faster. The multithreaded CPU and MPS rows were measured separately on the same input; `eval_ray_ingestion_latency` reports the single-threaded and Ray rows.

The benchmark uses 10 workers; `/index-distributed` runs 8. **Recommendation:** Use `/index-distributed` for documents with 1000+ sentences.

### Run Evaluations

```bash
# Test retrieval quality (flat vs parent-child vs parent-child + reranking)
python -m app.evaluation.eval_faiss_cross_encoder_ndcg

# Test ingestion performance (sequential vs Ray distributed)
python -m app.evaluation.eval_ray_ingestion_latency
```

---

## Project Structure

```
philosearch/
├── frontend/                           # Vite + React frontend
│   ├── src/
│   │   ├── App.tsx                     # Search interface, answer and sources
│   │   ├── main.tsx                    # Entry point
│   │   ├── index.css                   # Global styles
│   │   ├── components/
│   │   │   ├── auth-modal.tsx          # Access prompt
│   │   │   ├── library-panel.tsx       # Book picker (which books to search)
│   │   │   ├── book-cover.tsx          # Generated book covers
│   │   │   └── ui/                     # UI components
│   │   └── lib/
│   │       ├── answer-format.ts        # Renders the answer's Markdown
│   │       ├── books.ts                # Book titles, dates, cover colours, section labels
│   │       ├── api.ts                  # API base URL and error helper
│   │       └── utils.ts                # Utility functions
│   ├── vite.config.ts                  # Dev proxy: /api -> localhost:8000
│   └── package.json
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI endpoints
│   │   ├── services/
│   │   │   ├── embedding_service.py    # Sentence embeddings
│   │   │   ├── reranker_service.py     # Cross-encoder reranking
│   │   │   ├── vector_store.py         # FAISS vector store
│   │   │   ├── generation_service.py   # Gemini 3.8 Flash generation
│   │   │   └── distributed_ingestion.py # Ray parallel processing
│   │   ├── utils/
│   │   │   └── document_processor.py   # Text extraction & splitting
│   │   ├── scripts/
│   │   │   ├── index_texts.py          # Build FAISS index from texts/
│   │   │   ├── test_search.py          # Test retrieval pipeline
│   │   │   └── test_search_and_generate.py # Test RAG pipeline
│   │   └── evaluation/
│   │       ├── eval_faiss_cross_encoder_ndcg.py
│   │       └── eval_ray_ingestion_latency.py
│   ├── data/                           # FAISS index & metadata
│   ├── texts/                          # Source texts (local, not committed)
│   ├── .env.example
│   └── requirements.txt
├── Dockerfile
└── README.md
```

---

## Texts

The library in `backend/texts/` (filenames are `Title__Author_Name.txt`):

| Work | Author | Edition | Filename |
|------|--------|---------|----------|
| Tao Te Ching | Lao Tzu | Tr. Stephen Mitchell (1988) | `Tao_Te_Ching__Lao_Tzu.txt` |
| Nicomachean Ethics | Aristotle | Tr. F. H. Peters, via [Standard Ebooks](https://standardebooks.org/ebooks/aristotle/nicomachean-ethics/f-h-peters) | `Nicomachean_Ethics__Aristotle.txt` |
| Letters from a Stoic | Seneca | Tr. Robin Campbell (Penguin Classics, 1969), a selection of 42 letters | `Letters_from_a_Stoic__Seneca.txt` |
| The Enchiridion | Epictetus | | `The_Enchiridion__Epictetus.txt` |
| Meditations | Marcus Aurelius | | `Meditations__Marcus_Aurelius.txt` |
| Self-Reliance | Ralph Waldo Emerson | | `Self_Reliance__Ralph_Waldo_Emerson.txt` |
| Beyond Good and Evil | Friedrich Nietzsche | | `Beyond_Good_and_Evil__Friedrich_Nietzsche.txt` |
| The Myth of Sisyphus | Albert Camus | Tr. Justin O'Brien (1955) | `The_Myth_of_Sisyphus__Albert_Camus.txt` |

The texts and the index built from them are listed in `.gitignore` so they stay local. The Aristotle text was converted from the Standard Ebooks edition with front matter and endnotes removed. To add a book, save it as plain text with blank lines between paragraphs, run `python -m app.scripts.index_texts`, and add its display details to `frontend/src/lib/books.ts`.

**Deploying:** an image built from a clean checkout has no texts or index, so the library starts empty. Supply the texts at build time and run `python -m app.scripts.index_texts` before starting the server.

---

## Technical Details

**Device Support:**
- Auto-detects MPS (Apple Silicon) / CUDA (NVIDIA) / CPU
- Both embedding and reranker models use same device

**Storage:**
- FAISS index: `backend/data/faiss.index`
- Metadata: `backend/data/metadata.json`
- Persistent across restarts

**Text Processing:**
- Sentence tokenization: NLTK `punkt_tab`
- Passages follow each work's own structure (`DocumentProcessor.parse_document`): numbered sections ("XLIX.", "63.") and chapters start new passages; headings ("THE SIXTH BOOK", "Absurd Walls") become section labels instead of indexed text; numbered footnotes are dropped; short fragments (epigraphs, verse lines) join the passage they belong to
- Citations are the matched sentence plus up to two neighbours from the same passage, within ~300 characters; the reranker scores the candidate windows and each passage is cited once, by its best excerpt. The full passage is available behind "Show full passage"
- Generation cites only the excerpt; the full passage is sent as background

**Pricing (Gemini 3.8 Flash):**
- $0.75/1M input tokens, $3.75/1M output tokens (including thinking) through Dec 31, 2026
- $1.50/1M input tokens, $7.50/1M output tokens from Jan 1, 2027
- Fallback requests to Gemini 3.6 Flash or 3.5 Flash are billed at that model's rates
- On the free tier each model has its own small daily request limit (20/day at the time of writing), so the fallback chain also stretches the daily allowance across three models

---

## Troubleshooting

**"Gemini generator not initialized"**
- Check `backend/.env` file exists with `GOOGLE_API_KEY`
- Get API key from https://aistudio.google.com/apikey

**The access prompt keeps reappearing / "Your session expired"**
- Sessions last 24 hours. Changing `AUTH_SECRET` signs everyone out; if `AUTH_SECRET` is unset, so does changing `GOOGLE_API_KEY` or `CREATOR_NAME`

**Upload or delete returns 403**
- Set `ADMIN_TOKEN` in `backend/.env` and send it as the `X-Admin-Token` header

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
