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
src/
  fetch_raw_syllabi.py   downloads raw data (Checkpoint 1, deliverable 2)
  preprocess.py          cleaning/normalization/tokenization (deliverable 3)
  generate_embeddings.py embedding generation demo (deliverable 4)
docs/
  SyllaBot_Proposal.docx      project proposal (deliverable 1)
  preprocessing_report.md     before/after cleaning examples
  reflection.md                short reflection (deliverable 5)
```

## Setup

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Checkpoint 1 pipeline

```
python src/fetch_raw_syllabi.py       # (re)download raw data into data/raw/
python src/preprocess.py              # clean + tokenize -> data/processed/
python src/generate_embeddings.py     # embed chunks -> data/embeddings/
```

## Data Source

11 real course syllabus pages from MIT OpenCourseWare (Creative Commons
licensed, publicly downloadable), covering Computer Science, Mathematics,
Statistics, Economics, Psychology, Data Science/ML, and Database Systems --
chosen to mirror a student managing several different subjects in one
semester, which is SyllaBot's target use case. See `data/raw/*.txt` for
source URLs.
