"""
Checkpoint 2 - Distance Metric Comparison

Runs the same test queries against the Checkpoint 1 embeddings using three
similarity/distance metrics -- Cosine, Euclidean, and Dot Product -- and
writes a comparison report with example results and a recommendation to
docs/distance_metric_comparison.md.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "syllabus_embeddings.json"
REPORT_PATH = ROOT / "docs" / "distance_metric_comparison.md"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_QUERIES = [
    "What percentage of my grade comes from the final exam?",
    "What programming language does this course use?",
    "Is there a penalty for submitting an assignment late?",
]

TOP_K = 3


def cosine_sim(query_vec, matrix):
    q = query_vec / np.linalg.norm(query_vec)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m @ q  # higher = more similar


def euclidean_dist(query_vec, matrix):
    return np.linalg.norm(matrix - query_vec, axis=1)  # lower = more similar


def dot_product(query_vec, matrix):
    return matrix @ query_vec  # higher = more similar


def top_k_indices(scores, k, higher_is_better):
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    return order[:k]


def main():
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    records = data["records"]
    matrix = np.array([r["embedding"] for r in records], dtype=np.float32)

    print(f"Loading model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    report = [
        "# Checkpoint 2 - Distance Metric Comparison\n",
        f"Comparing Cosine similarity, Euclidean distance, and Dot Product on "
        f"the {len(records)} syllabus chunks embedded in Checkpoint 1 "
        f"(`sentence-transformers/all-MiniLM-L6-v2`, 384-dim), using the "
        f"same {len(TEST_QUERIES)} test queries against each metric.\n",
    ]

    for query in TEST_QUERIES:
        q_vec = model.encode(query, convert_to_numpy=True)

        cos_scores = cosine_sim(q_vec, matrix)
        euc_scores = euclidean_dist(q_vec, matrix)
        dot_scores = dot_product(q_vec, matrix)

        cos_top = top_k_indices(cos_scores, TOP_K, higher_is_better=True)
        euc_top = top_k_indices(euc_scores, TOP_K, higher_is_better=False)
        dot_top = top_k_indices(dot_scores, TOP_K, higher_is_better=True)

        report.append(f"## Query: \"{query}\"\n")
        for label, top_idx, scores in [
            ("Cosine similarity (higher = better)", cos_top, cos_scores),
            ("Euclidean distance (lower = better)", euc_top, euc_scores),
            ("Dot Product (higher = better)", dot_top, dot_scores),
        ]:
            report.append(f"**{label}**\n")
            for rank, idx in enumerate(top_idx, 1):
                src = records[idx]["source"]
                snippet = records[idx]["text"][:140].replace("\n", " ")
                report.append(f"{rank}. [{scores[idx]:.3f}] `{src}` — {snippet}...")
            report.append("")

        overlap_cos_euc = len(set(cos_top) & set(euc_top))
        overlap_cos_dot = len(set(cos_top) & set(dot_top))
        report.append(
            f"*Top-{TOP_K} overlap: Cosine/Euclidean = {overlap_cos_euc}/{TOP_K}, "
            f"Cosine/Dot = {overlap_cos_dot}/{TOP_K}*\n"
        )

    norms = np.linalg.norm(matrix, axis=1)
    report.append("## Recommendation\n")
    report.append(
        f"All {len(records)} chunk embeddings from `all-MiniLM-L6-v2` have norms "
        f"ranging {norms.min():.3f}-{norms.max():.3f} (mean {norms.mean():.3f}), i.e. "
        f"they are effectively unit-length already. When vectors are unit-normalized, "
        f"Cosine similarity and Dot Product become mathematically equivalent ranking-wise "
        f"(their top-k overlap above should be at or near {TOP_K}/{TOP_K}), while Euclidean "
        f"distance ranks differently because it is sensitive to the vectors' absolute "
        f"position, not just their angle.\n"
    )
    report.append(
        "**We recommend Cosine similarity** for SyllaBot's retrieval step: it's the "
        "standard choice for sentence-transformer embeddings, is scale-invariant (robust "
        "if we ever swap in an embedding model that doesn't output unit vectors), and is "
        "natively supported as the default distance space in Chroma, our chosen vector "
        "database (see `src/vector_store.py`).\n"
    )

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
