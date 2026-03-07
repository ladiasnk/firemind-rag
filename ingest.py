"""
ingest.py — The "loading dock" of our RAG system.

What this script does, step by step:
  1. Reads all .md (markdown) files from the /data folder
  2. Splits them into smaller chunks (because LLMs have limited context windows
     and smaller chunks make retrieval more precise)
  3. Converts each chunk into an embedding (a list of numbers that captures meaning)
  4. Stores those embeddings in ChromaDB (our local vector database)

Run this once before using the app:
  python ingest.py

If you add new documents to /data, run it again with --reset to rebuild the DB.
"""

import os
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")         # where our documents live
CHROMA_DIR = "chroma_db"        # where ChromaDB will persist on disk
COLLECTION_NAME = "firemind"
CHUNK_SIZE = 500                # characters per chunk
CHUNK_OVERLAP = 80              # overlap between chunks so context isn't lost at boundaries

# We use a small, fast local model for embeddings — no API key needed.
# all-MiniLM-L6-v2 is a solid general-purpose choice: small, fast, good quality.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into overlapping chunks.

    Why overlap? Imagine a sentence that falls right at the boundary between
    two chunks — without overlap, it gets split in half and neither chunk
    has the full context. Overlap prevents this.

    This is a simple character-based splitter. More advanced splitters
    (see: TODO below) respect sentence and paragraph boundaries.

    TODO: Replace this with a smarter splitter like LangChain's
    RecursiveCharacterTextSplitter — it respects paragraph and sentence
    boundaries, which usually improves retrieval quality.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # step forward, but keep some overlap
    return [c for c in chunks if len(c) > 50]  # skip tiny leftover chunks


# ── Document loading ──────────────────────────────────────────────────────────

def load_documents(data_dir: Path) -> list[dict]:
    """
    Load all .md files from the data directory.
    Returns a list of dicts with 'text', 'source', and 'chunk_id'.
    """
    docs = []
    md_files = list(data_dir.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No .md files found in {data_dir}. Add some documents first.")

    for filepath in md_files:
        print(f"  📄 Loading: {filepath.name}")
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "source": filepath.name,
                "chunk_id": f"{filepath.stem}_chunk_{i}"
            })

    print(f"\n  ✅ Loaded {len(md_files)} files → {len(docs)} chunks total\n")
    return docs


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest(reset: bool = False):
    """
    Full ingestion pipeline: load → chunk → embed → store.
    """
    print("=" * 55)
    print("  🔥 FireMind RAG — Ingestion Pipeline")
    print("=" * 55)

    # Step 1: Load documents
    print("\n[1/4] Loading documents from /data ...")
    docs = load_documents(DATA_DIR)

    # Step 2: Set up ChromaDB
    print("[2/4] Setting up ChromaDB ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        print("  ⚠️  Resetting existing collection...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet, that's fine

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity works well for text
    )

    # Skip if already populated (and not resetting)
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(f"  ℹ️  Collection already has {existing_count} chunks. Use --reset to rebuild.")
        return

    # Step 3: Create embeddings
    print(f"[3/4] Loading embedding model ({EMBEDDING_MODEL}) ...")
    print("      (First run downloads ~90MB — subsequent runs are instant)\n")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [doc["text"] for doc in docs]
    print(f"  🔢 Embedding {len(texts)} chunks ...")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    # Step 4: Store in ChromaDB
    print("\n[4/4] Storing in ChromaDB ...")
    collection.add(
        ids=[doc["chunk_id"] for doc in docs],
        documents=[doc["text"] for doc in docs],
        embeddings=embeddings.tolist(),
        metadatas=[{"source": doc["source"]} for doc in docs]
    )

    print(f"\n  ✅ Done! {len(docs)} chunks stored in '{CHROMA_DIR}/'")
    print("  You can now run: streamlit run app.py\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into FireMind RAG")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vector DB")
    args = parser.parse_args()
    ingest(reset=args.reset)
