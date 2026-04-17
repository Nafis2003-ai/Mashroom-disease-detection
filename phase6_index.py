"""
phase6_index.py  —  Build ChromaDB vector index from rag_docs/
==============================================================
Run AFTER phase6_scrape.py.
Chunks all text files, embeds with sentence-transformers, stores in ChromaDB.

Usage:
    python phase6_index.py
"""

import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR  = os.path.join(BASE_DIR, "rag_docs")
DB_DIR   = os.path.join(BASE_DIR, "rag_db")

CHUNK_WORDS   = 350   # target words per chunk
CHUNK_OVERLAP = 50    # overlapping words between consecutive chunks
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, 384-dim, free, runs locally
COLLECTION    = "mushroom_knowledge"


# ── Text chunker ──────────────────────────────────────────────────────────────
def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by word count.
    Tries to split on sentence boundaries when possible.
    """
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        i += chunk_words - overlap
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PHASE 6B — BUILDING VECTOR INDEX")
    print("=" * 65)

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\n  Loading embedding model: {EMBED_MODEL} ...")
    print("  (downloads ~90 MB on first run — cached locally after)")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    print("  Model ready.")

    # ── Init ChromaDB ─────────────────────────────────────────────────────────
    print(f"\n  Initializing ChromaDB at: {DB_DIR}")
    import chromadb
    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh rebuild — delete existing collection if present
    try:
        client.delete_collection(COLLECTION)
        print("  Existing collection deleted (fresh rebuild).")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Process text files ────────────────────────────────────────────────────
    txt_files = sorted(glob.glob(os.path.join(RAG_DIR, "*.txt")))
    if not txt_files:
        print(f"\n  ERROR: No .txt files in {RAG_DIR}")
        print("  Run phase6_scrape.py first.")
        return

    print(f"\n  Found {len(txt_files)} source files.\n")

    total_chunks = 0
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = chunk_text(text)
        if not chunks:
            print(f"  [SKIP] {fname}: empty after chunking")
            continue

        # Embed all chunks for this file
        embeddings = embedder.encode(
            chunks,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        # Unique IDs: filename + chunk index
        ids       = [f"{fname}::{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": fname, "chunk_idx": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"  [OK]  {fname:<52}  {len(chunks):>4} chunks")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(f"  Total chunks indexed : {total_chunks}")
    print(f"  Collection name      : '{COLLECTION}'")
    print(f"  Stored at            : {DB_DIR}")
    print(f"\n  Next step: streamlit run phase6_rag_app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
