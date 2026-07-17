"""
ingest.py — Build the vector index (run ONCE, or whenever you add new docs).

No FAISS. Uses numpy — avoids the macOS OpenMP bus error entirely.

Usage:
    python ingest.py
"""

import os

# Anaconda ships a broken TensorFlow on Apple Silicon; transformers tries to
# import it and hard-crashes (bus error). We are PyTorch-only, so disable TF.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
STORE_DIR = Path("store")
STORE_DIR.mkdir(exist_ok=True)

# Swap to "pritamdeka/S-PubMedBert-MS-MARCO" for a medical-tuned embedder.
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # medical-tuned, 768-dim
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120     # overlap so sentences aren't cut awkwardly


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def main():
    files = [p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    if not files:
        print("No documents found in ./data — add .txt/.md/.pdf files first.")
        return

    print(f"Loading embedder: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    all_chunks, metadata = [], []
    for f in files:
        chunks = chunk_text(read_file(f))
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"source": f.name, "chunk_id": i, "text": chunk})
        print(f"  {f.name}: {len(chunks)} chunks")

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        normalize_embeddings=True,   # so dot product == cosine similarity
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    np.save(STORE_DIR / "embeddings.npy", embeddings)
    with open(STORE_DIR / "meta.pkl", "wb") as fh:
        pickle.dump(metadata, fh)

    print(f"Done. Indexed {len(all_chunks)} chunks, dim={embeddings.shape[1]} -> ./store")


if __name__ == "__main__":
    main()
