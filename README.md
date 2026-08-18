# SyllaBot

RAG-based course syllabus and study guide assistant. IS Professional
Elective #4 (Generative AI Systems) capstone project.

## Team

| Member | Component |
|---|---|
| Sugar Rey R. Garcia | Data Pipeline & Embeddings |
| Rudelyn Joy Abanador | RAG Architecture & Prompting |
| Joel Jr. A. Madula | Deployment & Interface |

## Repository Structure

```
data/
  raw/         raw scraped syllabus pages (data collection)
  processed/   cleaned text + tokens (preprocessing output)
  embeddings/  generated embedding vectors
  chroma_db/   persistent vector database (regenerable, gitignored)
src/
  fetch_raw_syllabi.py       downloads raw data (Checkpoint 1, deliverable 2)
  preprocess.py               cleaning/normalization/tokenization (deliverable 3)
  generate_embeddings.py      section-aware chunking + embedding generation (deliverable 4)
  vector_store.py             Chroma vector DB setup + similarity search (Checkpoint 2, deliverable 8)
  compare_distance_metrics.py Cosine/Euclidean/Dot Product comparison (deliverable 9)
  api_integration_demo.py     Gemini API + grounded RAG demo (deliverable 7)
  ingest.py                   automated parse->chunk->embed->store pipeline (Checkpoint 3, deliverable 10)
  rag_app.py                  LangChain RAG app w/ hybrid retrieval + conversational memory (deliverables 11-12)
docs/
  SyllaBot_Proposal.docx           project proposal (deliverable 1)
  preprocessing_report.md          before/after cleaning examples
  reflection.md                     short reflection (deliverable 5)
  prompt_engineering.md            prompt design + hallucination-prevention (deliverable 6)
  distance_metric_comparison.md    metric comparison report (deliverable 9)
  framework_comparison.md          LangChain vs. LlamaIndex note (Checkpoint 3, deliverable 13)
```

## Setup

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For `api_integration_demo.py` and `rag_app.py`, also set a free Gemini API
key as an environment variable (never hardcode it in a file):
```
$env:GEMINI_API_KEY = "your-key-here"     # PowerShell, current session
setx GEMINI_API_KEY "your-key-here"       # persists across sessions
```
Get a key at https://aistudio.google.com/apikey

## Running the pipeline

Checkpoint 1-2 pipeline (manual, step by step):
```
python src/fetch_raw_syllabi.py           # (re)download raw data into data/raw/
python src/preprocess.py                  # clean + tokenize -> data/processed/
python src/generate_embeddings.py         # embed chunks -> data/embeddings/
python src/vector_store.py                # index into Chroma + example queries
python src/compare_distance_metrics.py    # Cosine/Euclidean/Dot Product report
python src/api_integration_demo.py        # grounded Q&A via Gemini (needs GEMINI_API_KEY)
```

Checkpoint 3 pipeline (automated ingestion + RAG app):
```
python src/ingest.py                      # parse+chunk+embed+store everything in data/raw/ (or: python src/ingest.py path/to/new.txt)
python src/rag_app.py                     # interactive multi-turn RAG chat (needs GEMINI_API_KEY)
python src/rag_app.py --demo              # scripted demo: 5 grounded queries + 1 out-of-scope refusal
```

## Data Source

12 real course syllabus pages from MIT OpenCourseWare (Creative Commons
licensed, publicly downloadable), covering Computer Science, Mathematics,
Statistics, Economics, Psychology, Data Science/ML, Linear Algebra, and
Database Systems -- chosen to mirror a student managing several different
subjects in one semester, which is SyllaBot's target use case. See
`data/raw/*.txt` for source URLs.
