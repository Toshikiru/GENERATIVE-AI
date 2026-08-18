"""
Checkpoint 2 - Vector Database Setup

Loads the Checkpoint 1 embeddings into a persistent Chroma collection
(cosine distance space, per docs/distance_metric_comparison.md) and runs a
few example similarity-search queries to prove retrieval works end-to-end.

We pass our own precomputed embeddings (from sentence-transformers/
all-MiniLM-L6-v2) rather than letting Chroma pick a default embedding
function, so the model used for indexing stays identical to the one used
for querying and to the one justified in Checkpoint 1.
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "syllabus_embeddings.json"
CHROMA_DIR = ROOT / "data" / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "syllabot_chunks"

EXAMPLE_QUERIES = [
    "How much is the final exam worth?",
    "What textbook is required for this course?",
    "Can I collaborate with classmates on problem sets?",
]


def build_collection(client, model):
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    records = data["records"]

    # Recreate fresh each run so this script is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(records))],
        embeddings=[r["embedding"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[{"source": r["source"]} for r in records],
    )
    print(f"Indexed {len(records)} chunks into Chroma collection '{COLLECTION_NAME}' "
          f"({data['dimensions']}-dim, cosine space) at {CHROMA_DIR}")
    return collection


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    model = SentenceTransformer(MODEL_NAME)
    collection = build_collection(client, model)

    for query in EXAMPLE_QUERIES:
        q_vec = model.encode(query).tolist()
        results = collection.query(query_embeddings=[q_vec], n_results=3)

        print(f"\nQuery: \"{query}\"")
        for rank, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
        ):
            snippet = doc[:140].replace("\n", " ")
            print(f"  {rank}. [cosine_dist={dist:.3f}] {meta['source']} — {snippet}...")


if __name__ == "__main__":
    main()
