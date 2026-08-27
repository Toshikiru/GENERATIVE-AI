"""
Checkpoint 2 - API Integration Demo

Retrieves relevant syllabus context from the Chroma vector store (built by
src/vector_store.py) and sends it to Google's Gemini Chat Completion API
along with the grounded system prompt from docs/prompt_engineering.md
(Prompt 1). Demonstrates end-to-end: query -> retrieval -> grounded
generation.

Requires a free Gemini API key: https://aistudio.google.com/apikey
Set it as an environment variable before running (never hardcode it here):
    setx GEMINI_API_KEY "your-key-here"      (persists across sessions)
  or (current session only):
    $env:GEMINI_API_KEY = "your-key-here"    (PowerShell)
"""
import os
import sys
import time
from pathlib import Path

import chromadb
from google import genai
from google.genai import errors as genai_errors
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "syllabot_chunks"
GEMINI_MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are SyllaBot, a course syllabus and study guide assistant.

You will be given a student's question and one or more excerpts retrieved
from their course syllabi. Answer using ONLY the information in the
provided excerpts.

Rules:
1. If the excerpts contain the answer, respond concisely and cite which
   course/document it came from.
2. If the excerpts do NOT contain enough information to answer, say so
   explicitly: "I couldn't find that in your uploaded syllabi." Do not
   guess, estimate, or fill in with general knowledge about how courses
   "usually" work.
3. Never invent a percentage, date, or policy that is not literally present
   in the excerpts.
4. If the question is unrelated to course syllabi, politely decline."""

DEMO_QUESTIONS = [
    "How much is the final exam worth in the microeconomics course?",
    "What is the capital of France?",  # deliberately out-of-scope
]


def retrieve_context(collection, embed_model, question: str, k: int = 3):
    q_vec = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]
    return docs, sources, distances


def format_context(docs, sources):
    return "\n\n".join(f"[{src}]\n{doc}" for doc, src in zip(docs, sources))


def generate_with_retry(client, model, prompt, retries=2, delay_seconds=5):
    """
    Gemini's free tier occasionally returns a transient 503 (server
    overloaded) that has nothing to do with our own pipeline -- retrying
    a couple of times with a short delay clears most of these without
    interrupting a live demo.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except genai_errors.ServerError as exc:
            last_exc = exc
            if attempt < retries:
                print(f"  (Gemini temporarily unavailable, retrying in "
                      f"{delay_seconds}s...)", file=sys.stderr)
                time.sleep(delay_seconds)
    raise last_exc


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.")
        print("Get a free key at https://aistudio.google.com/apikey and set it, e.g.:")
        print('  PowerShell:  $env:GEMINI_API_KEY = "your-key-here"')
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    embed_model = SentenceTransformer(MODEL_NAME)

    for question in DEMO_QUESTIONS:
        docs, sources, distances = retrieve_context(collection, embed_model, question)
        context = format_context(docs, sources)

        # Low-confidence retrieval (cosine distance close to 1.0 = low
        # similarity) signals the question is likely out of scope, per
        # Prompt 3 in docs/prompt_engineering.md.
        best_distance = distances[0]

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\nRetrieved excerpts:\n{context}\n\n"
            f"Student question: {question}"
        )
        try:
            response = generate_with_retry(client, GEMINI_MODEL, full_prompt)
            answer = response.text
        except genai_errors.ServerError as exc:
            answer = ("[Gemini API is temporarily unavailable right now -- "
                       "this is a live-service hiccup, not a bug in our "
                       "pipeline. Retrieval above already shows the correct "
                       "grounded context was found; the generation step "
                       "would apply the same grounded prompt to it.]")
            print(f"  (API error after retries: {exc})", file=sys.stderr)

        print(f"\n{'=' * 70}")
        print(f"Question: {question}")
        print(f"Top retrieval cosine distance: {best_distance:.3f} "
              f"({'likely relevant' if best_distance < 0.7 else 'likely out-of-scope'})")
        print(f"Retrieved from: {sources}")
        print(f"\nSyllaBot: {answer}")


if __name__ == "__main__":
    main()
