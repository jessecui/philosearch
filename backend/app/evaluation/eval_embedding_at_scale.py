"""
Embedding latency at scale: single-threaded CPU vs PyTorch's default
multithreading vs Ray CPU workers, with startup counted for every method.

Ray workers take several seconds to start, so on small jobs PyTorch's own
multithreading is as fast or faster; this benchmark shows where that cost
stops mattering. Sentences come from backend/texts/ (repeated to reach each
size), falling back to the bundled evaluation document if texts/ is empty.
Each method runs in its own process so thread settings don't interact.

Usage (from backend/):
    python -m app.evaluation.eval_embedding_at_scale [SIZE ...]
    (default sizes: 20000 300000; 300K sentences takes about 15 minutes)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[2]
METHODS = ("single", "default", "ray")
LABELS = {
    "single": "Single-threaded CPU",
    "default": "PyTorch default multithreading (CPU)",
    "ray": "Ray, 10 CPU workers",
}


def load_sentences(n: int) -> tuple[list[str], str]:
    from app.utils.document_processor import DocumentProcessor

    dp = DocumentProcessor()
    files = sorted((BACKEND / "texts").glob("*.txt"))
    source = "backend/texts"
    if not files:
        files = [Path(__file__).with_name("eval_philosophy_document.txt")]
        source = files[0].name
    base: list[str] = []
    for f in files:
        base += dp.parse_document(f.read_text(encoding="utf-8")).sentences
    return (base * (n // len(base) + 1))[:n], source


def run_method(method: str, n: int) -> dict:
    """Embed n sentences with one method; called in a fresh process."""
    sentences, source = load_sentences(n)
    if method in ("single", "default"):
        if method == "single":
            torch.set_num_threads(1)
        from app.services.embedding_service import EmbeddingService

        start = time.perf_counter()
        service = EmbeddingService(force_cpu=True)
        service.embed_texts(["warm up"])
        startup = time.perf_counter() - start
        start = time.perf_counter()
        service.embed_texts(sentences)
        work = time.perf_counter() - start
    else:
        from app.services.distributed_ingestion import DistributedIngestionPipeline

        start = time.perf_counter()
        pipeline = DistributedIngestionPipeline(num_workers=10, batch_size=128)
        startup = time.perf_counter() - start
        start = time.perf_counter()
        pipeline.embed_sentences_distributed(sentences)
        work = time.perf_counter() - start
        pipeline.shutdown()
    return {"method": method, "sentences": n, "source": source,
            "startup_s": startup, "work_s": work, "total_s": startup + work}


def main(sizes: list[int]) -> None:
    results = []
    for n in sizes:
        for method in METHODS:
            print(f"Running {LABELS[method]} on {n:,} sentences...", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "app.evaluation.eval_embedding_at_scale",
                 "--worker", method, str(n)],
                cwd=BACKEND, capture_output=True, text=True, check=True,
            )
            line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
            results.append(json.loads(line[len("RESULT "):]))

    print(f"\nSentences from {results[0]['source']}; times include startup\n")
    for n in sizes:
        rows = {r["method"]: r for r in results if r["sentences"] == n}
        print(f"=== {n:,} sentences ===")
        for method in METHODS:
            r = rows[method]
            print(f"  {LABELS[method]:40} {r['total_s']:7.1f}s "
                  f"(startup {r['startup_s']:.1f}s, {n / r['work_s']:,.0f} sent/s warm)")
        ray = rows["ray"]["total_s"]
        for method in ("default", "single"):
            other = rows[method]["total_s"]
            change = (1 - ray / other) * 100
            verdict = f"{change:.1f}% faster" if change >= 0 else f"{-change:.1f}% slower"
            print(f"  Ray vs {LABELS[method].split(' (')[0]}: {verdict} ({other / ray:.2f}x)")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print("RESULT " + json.dumps(run_method(sys.argv[2], int(sys.argv[3]))))
    else:
        main([int(a) for a in sys.argv[1:]] or [20000, 300000])
