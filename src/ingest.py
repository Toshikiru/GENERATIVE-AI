"""
Checkpoint 3 - Ingestion Pipeline

End-to-end document ingestion: parse -> clean -> chunk -> embed -> store,
in one command. This is the automated version of the three manual steps
from Checkpoint 1 (preprocess.py -> generate_embeddings.py -> vector_store.py)
-- drop new syllabus .txt files into data/raw/ and run this single script
to get them cleaned, chunked, embedded, and indexed into Chroma with no
other manual steps.

Usage:
    python src/ingest.py                  # ingest everything in data/raw/
    python src/ingest.py path/to/new.txt  # ingest a single new file
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import split_header, strip_site_chrome, clean_text  # noqa: E402
from generate_embeddings import chunk_text, doc_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "syllabot_chunks"


def ingest_file(path: Path, model, collection) -> int:
    raw_text = path.read_text(encoding="utf-8")
    _, body = split_header(raw_text)
    stripped = strip_site_chrome(body)
    cleaned = clean_text(stripped)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / path.name).write_text(cleaned, encoding="utf-8")

    label = doc_label(path.name)
    chunks = [f"Course: {label}\n{c}" for c in chunk_text(cleaned)]
    if not chunks:
        print(f"  {path.name}: no usable chunks, skipped")
        return 0

    embeddings = model.encode(chunks, convert_to_numpy=True)
    ids = [f"{path.stem}_{i}" for i in range(len(chunks))]

    # Remove any previous chunks for this document so re-ingesting the same
    # file updates it instead of duplicating entries.
    existing = collection.get(where={"source": path.name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{"source": path.name} for _ in chunks],
    )
    print(f"  {path.name}: {len(chunks)} chunks ingested")
    return len(chunks)


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        targets = sorted(RAW_DIR.glob("*.txt"))

    print(f"Ingesting {len(targets)} document(s) ...")
    total = 0
    for path in targets:
        total += ingest_file(path, model, collection)

    print(f"\nDone. {total} chunks now indexed across the collection "
          f"({collection.count()} total chunks in '{COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
