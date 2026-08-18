"""
Checkpoint 1 - Embedding Generation Demo

Chunks the cleaned syllabus text (data/processed/) into short passages and
embeds them with a Hugging Face sentence-transformer model. Saves the
resulting vectors plus a small sanity check (cosine similarity between a
same-topic pair and a different-topic pair) to demonstrate the embeddings
are semantically meaningful, not just random vectors.

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - Purpose-built for semantic similarity / retrieval (unlike a generic
    causal LM), which is exactly what a RAG pipeline needs.
  - 384-dim output, ~80MB, runs fast on CPU -- no GPU required for a
    dataset this size (11 documents).
  - Widely used as the default embedding model in LangChain/LlamaIndex RAG
    tutorials, so it keeps Checkpoint 2/3 (vector DB + orchestration
    framework) straightforward to build on top of.
  - Open-source / free via Hugging Face, no API key needed.
"""
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
EMBED_DIR = ROOT / "data" / "embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _is_section_header(line: str) -> bool:
    """
    The cleaned syllabus text still carries one structural signal from the
    original page: short section-title lines ("Exams", "Grading Policy",
    "Textbook"...) sit on their own line immediately before the prose that
    belongs to them. A blind fixed-size sentence window ignores this and
    can slice a section's prose in half -- e.g. splitting "There will also
    be a three-hour final... This will count as the remaining 50% of the
    grade" across two unrelated chunks. Detecting headers lets us chunk by
    section instead, so a fact and the sentence naming it stay together.
    """
    words = line.split()
    return 1 <= len(words) <= 5 and not line.rstrip().endswith((".", ",", ";", ":"))


def _split_long_section(section: str, max_words: int = 120):
    words = section.split()
    if len(words) <= max_words:
        return [section]
    sentences = re.split(r"(?<=[.!?])\s+", section)
    chunks, current = [], []
    for sent in sentences:
        current.append(sent)
        if sum(len(s.split()) for s in current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def doc_label(filename: str) -> str:
    """
    Turn '11_database_systems.txt' into 'Database Systems'. Once a document
    is split into chunks, each chunk's course identity would otherwise only
    live in a sidecar metadata field -- invisible to both the embedding
    model and BM25 keyword search. A query like "...in the database
    systems course" then can't out-rank a same-topic chunk from a
    different course (e.g. every syllabus has near-identical
    "Collaboration Policy" wording). Prefixing every chunk with its course
    label bakes that identity into the searchable text itself.
    """
    stem = re.sub(r"^\d+_", "", Path(filename).stem)
    return stem.replace("_", " ").title()


def chunk_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections = []
    header, body = None, []

    def flush():
        if body:
            content = (f"{header}\n" if header else "") + " ".join(body)
            sections.append(content.strip())

    for line in lines:
        if _is_section_header(line):
            flush()
            header, body = line, []
        else:
            body.append(line)
    flush()

    chunks = []
    for section in sections:
        if len(section.split()) >= 8:  # skip near-empty fragments
            chunks.extend(_split_long_section(section))
    return chunks


def main():
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    all_sources = []
    for path in sorted(PROCESSED_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        label = doc_label(path.name)
        chunks = [f"Course: {label}\n{c}" for c in chunk_text(text)]
        all_chunks.extend(chunks)
        all_sources.extend([path.name] * len(chunks))

    print(f"Embedding {len(all_chunks)} chunks from {len(set(all_sources))} documents ...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    out = {
        "model": MODEL_NAME,
        "dimensions": int(embeddings.shape[1]),
        "num_chunks": len(all_chunks),
        "records": [
            {"source": src, "text": chunk, "embedding": vec.tolist()}
            for src, chunk, vec in zip(all_sources, all_chunks, embeddings)
        ],
    }
    out_path = EMBED_DIR / "syllabus_embeddings.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {len(all_chunks)} embeddings ({embeddings.shape[1]}-dim) -> {out_path}")

    # Sanity check: a same-topic pair FROM DIFFERENT DOCUMENTS should score
    # higher than a pair of chunks on unrelated topics. Restricting to
    # different source documents matters -- two chunks from the same page
    # will look similar just because they share the same course, prose
    # style, and vocabulary, which isn't evidence the embeddings capture
    # topic meaning rather than just "same document."
    def first_match_from_each_doc(keyword: str, min_docs: int = 2):
        seen_sources = {}
        for src, chunk in zip(all_sources, all_chunks):
            if keyword in chunk.lower() and src not in seen_sources:
                seen_sources[src] = chunk
            if len(seen_sources) >= min_docs:
                break
        return list(seen_sources.items())

    grading_pairs = first_match_from_each_doc("grading")
    if len(grading_pairs) >= 2:
        (src_a, chunk_a), (src_b, chunk_b) = grading_pairs[0], grading_pairs[1]
        idx_a, idx_b = all_chunks.index(chunk_a), all_chunks.index(chunk_b)

        # An unrelated chunk: pick one from a third, topically distant
        # document (not src_a or src_b) that does NOT mention grading.
        idx_c = next(
            i for i, (src, c) in enumerate(zip(all_sources, all_chunks))
            if src not in (src_a, src_b) and "grading" not in c.lower()
        )

        sim_same_topic = util.cos_sim(embeddings[idx_a], embeddings[idx_b]).item()
        sim_diff_topic = util.cos_sim(embeddings[idx_a], embeddings[idx_c]).item()
        print("\nSanity check (cosine similarity):")
        print(f"  '{src_a}' grading chunk <-> '{src_b}' grading chunk: {sim_same_topic:.3f}")
        print(f"  '{src_a}' grading chunk <-> '{all_sources[idx_c]}' unrelated chunk: {sim_diff_topic:.3f}")
        print("  (same-topic-different-document score should be higher than the unrelated pair)")


if __name__ == "__main__":
    main()
