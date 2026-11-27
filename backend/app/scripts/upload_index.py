"""
Upload the locally built index to the private Hugging Face dataset repo that
deployments download it from.

Usage (from backend/, with INDEX_REPO and a write-access HF_TOKEN set in
backend/.env or the environment):

    python -m app.scripts.upload_index
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from app.services.index_sync import upload_index
from app.services.vector_store import VectorStore


def main() -> int:
    try:
        # Loading it checks the two files belong together, so a broken
        # index is never uploaded
        store = VectorStore()
        documents = len(store.list_documents())
        if not documents:
            print("❌ The index in data/ is empty. Build it first with `python -m app.scripts.index_texts`.")
            return 1
        url = upload_index()
    except Exception as e:
        print(f"❌ {e}")
        return 1
    print(f"✓ Uploaded the index ({documents} documents): {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
