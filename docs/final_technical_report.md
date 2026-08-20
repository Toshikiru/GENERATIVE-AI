# SyllaBot — Final Technical Report

IS Professional Elective #4 (Generative AI Systems) — Mini Generative AI
Capstone Project

**Status note:** this report covers Checkpoints 1-4, including verified
Docker build/run evidence (§7.3).

---

## 1. Problem & Motivation

A student juggling several courses in one semester has to track grading
weights, late-day policies, collaboration rules, and textbook
recommendations spread across several separately formatted syllabus
documents. SyllaBot is a Retrieval-Augmented Generation (RAG) assistant
that answers natural-language questions about a student's own course
syllabi, grounded strictly in the syllabus text itself — it is designed to
explicitly refuse rather than guess when an answer isn't present in the
indexed documents, which matters because a plausible-sounding but wrong
answer about a real deadline or grading weight is worse than no answer.

**Target users:** students managing multiple courses in one term.
**Success criteria:** grounded, cited answers to realistic syllabus
questions; correct refusal on out-of-scope questions; a working multi-turn
conversational interface; a deployable, containerized application.

## 2. Architecture Overview

```
data/raw (scraped syllabi)
    -> preprocess.py        clean + strip site chrome
    -> generate_embeddings.py   section-aware chunk + course-label prefix + embed
    -> Chroma (data/chroma_db)  persistent vector store, cosine space
    -> rag_app.py            hybrid retrieval (dense + BM25) -> Gemini, with
                              per-session conversational memory
    -> web_app.py             Streamlit chat interface
    -> Docker                 multi-stage image, ingest baked in at build time
```

`src/ingest.py` automates the top three stages (parse -> chunk -> embed ->
store) into one command for new documents, replacing the manual
step-by-step Checkpoint 1/2 pipeline for ongoing ingestion.

## 3. Data Pipeline (Checkpoint 1)

**Source:** 12 real course syllabus pages from MIT OpenCourseWare
(Creative Commons licensed), spanning Computer Science, Mathematics,
Statistics, Economics, Psychology, Data Science/ML, Linear Algebra, and
Database Systems — chosen to mirror a student's actual mixed course load.

**Cleaning (`src/preprocess.py`):** raw scraped pages carry repeated
navigation menus, a duplicated course-info sidebar, icon-label artifacts,
and a global footer/cookie modal — in some cases as much boilerplate text
as actual content. Rather than writing per-page rules, cleaning exploits a
structural pattern consistent across the whole platform: real syllabus
prose always sits between the *last* occurrence of the line "Syllabus" and
the *last* occurrence of "Course Info". This generalized across all 12
documents despite them spanning pages from 1999-2024 with very different
lengths and structures (see `docs/reflection.md` for the specific edge
cases, e.g. a paginated psychology syllabus that required extra filtering).
HTML entities are decoded and whitespace is normalized after chrome
stripping.

**Tokenization:** NLTK `sent_tokenize`/`word_tokenize` with a regex
fallback if NLTK data is unavailable. Before/after examples and per-document
word/sentence/token counts are in `docs/preprocessing_report.md`.

## 4. Embeddings & Vector Store (Checkpoints 1-2)

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — used
consistently for both indexing and querying throughout the project via a
shared `Embeddings` wrapper class, so retrieval never mismatches embedding
spaces.

**Chunking (`src/generate_embeddings.py`):** chunking is section-aware
rather than a fixed-size sliding window — it detects short header-like
lines (e.g. "Late Days", "Grading Policy") and groups the prose that
follows into one chunk per section, splitting only sections that exceed
~120 words. This keeps a fact and the sentence that names it in the same
chunk (a fixed window can otherwise slice a policy sentence across two
unrelated chunks). Every chunk is also prefixed with its course label
(`doc_label()`, e.g. "Course: Database Systems") — this became important
in Checkpoint 3 (§6).

**Vector store (`src/vector_store.py`):** Chroma, persistent, cosine
distance space.

**Distance metric decision (`docs/distance_metric_comparison.md`):**
Cosine, Euclidean, and Dot Product were compared on the same 3 test
queries. Because `all-MiniLM-L6-v2` outputs effectively unit-normalized
embeddings, Cosine and Dot Product ranked identically (3/3 top-k overlap)
while Euclidean distance is sensitive to absolute vector position, not just
angle. **Cosine similarity** was selected — the standard choice for
sentence-transformer embeddings, scale-invariant, and Chroma's default.

## 5. Prompt Design (Checkpoint 2)

Three structured prompts (full text and hallucination-prevention examples
in `docs/prompt_engineering.md`):

1. **Grounded Q&A** — the default system prompt; answers only from
   retrieved excerpts, explicit refusal phrasing when the answer isn't
   present, never invents a percentage/date/policy.
2. **Multi-Course Aggregation** — used for queries spanning multiple
   courses at once; addresses each course separately rather than silently
   dropping ones with no matching data.
3. **Out-of-Scope Refusal Filter** — a cheap pre-classification pass for
   low-similarity retrieval results, catching off-topic questions (e.g.
   "What's the capital of France?") before the more expensive grounded
   generation prompt runs.

The shared design principle: the model is told exactly what counts as a
valid basis for an answer, and is given an explicit, low-friction way to
say "I don't know" instead of being implicitly rewarded for always sounding
confident.

## 6. RAG Implementation (Checkpoint 3)

**Ingestion (`src/ingest.py`):** automates parse -> clean -> chunk -> embed
-> store end-to-end for `data/raw/*.txt` (or a single new file), replacing
Checkpoint 1/2's manual multi-script process; re-ingesting a document
replaces its old chunks instead of duplicating them.

**Framework: LangChain** (full comparison against LlamaIndex in
`docs/framework_comparison.md`). SyllaBot's requirement isn't routing
between multiple index types — it's a single flat collection needing
retrieval + grounding + multi-turn memory composed into one pipeline, which
maps directly onto LangChain's LCEL chain composition (`RunnablePassthrough
.assign(...) | prompt | llm`) and `RunnableWithMessageHistory`.

**Hybrid retrieval:** dense vector search alone struggles to distinguish
two courses' near-identical boilerplate (e.g. every syllabus's
"Collaboration Policy" reads almost the same). SyllaBot combines dense
retrieval with BM25 keyword search (`EnsembleRetriever`, 0.5/0.5 weights) —
BM25 reliably catches distinctive terms like a named course that dense
embeddings under-weight, and this is reinforced by the course-label
chunk-prefixing from §4.

**Conversational memory:** `RunnableWithMessageHistory` with a per-session
`InMemoryChatMessageHistory`, keyed by session ID (CLI: a fixed session
string; web UI: a per-browser-session UUID).

**Query condensing and a bug found + fixed during testing:** because
retrieval runs on the raw question text before the LLM sees it, a follow-up
like "what textbook does *it* recommend?" needs its pronoun resolved before
retrieval, not just at generation time — otherwise retrieval search on the
literal word "it" fails regardless of what the final LLM could have
inferred. A small auxiliary LLM call condenses each follow-up into a
standalone question using conversation history.

While live-testing the Checkpoint 3 demo, this surfaced a real grounding
bug: the question *"What's the late policy for the Python programming
course?"* was answered using the **Data Science** syllabus instead of the
actual **Intro CS Programming Python** course, because an earlier turn had
mentioned "the Data Science course uses Python," and the condense step
over-eagerly rewrote the already-specific follow-up into a *different*
course pulled from that history. Raw retrieval (bypassing the condense
step) already ranked the correct course first for the literal question —
the fault was isolated to the condense prompt substituting a course name it
shouldn't have. The fix tightened `CONDENSE_PROMPT` to only fill in
genuinely vague references (bare pronouns/bare phrases) and explicitly
forbid substituting a different course when the follow-up already names or
describes one. Verified against the exact failure case (condensing
returned the question unchanged, as required) and against a legitimate
pronoun-resolution case (still correctly resolved) to confirm the fix
didn't regress the feature it was patching.

## 7. Application Interface & Deployment (Checkpoint 4)

### 7.1 Model Customization

Full reasoning in `docs/model_customization.md`. Summary: the generator
(`gemini-3.5-flash`) is a closed-weight API model, so LoRA/QLoRA isn't
applicable to it directly — customization there is limited to prompting and
grounding, which SyllaBot already leans on. The concrete PEFT opportunity
identified is a LoRA fine-tune of the open-weight embedding model
(`all-MiniLM-L6-v2`) with a contrastive objective, targeting the same
near-duplicate-boilerplate retrieval weakness that BM25 + course-label
prefixing currently compensate for at the data layer. This wasn't performed
for this checkpoint — the cheaper data/architecture-level fix already
resolves the practical problem — but is documented as the next concrete
step if the project continued.

### 7.2 Application Interface

`src/web_app.py`: a Streamlit chat interface wrapping `rag_app.build_chain()`
directly — all retrieval/grounding/memory logic stays in `rag_app.py`, the
web layer is presentation only. Uses `st.cache_resource` so the embedding
model and vector index load once per server process rather than on every
interaction, and a per-browser-session UUID for conversational memory.
Verified locally: the chat renders, round-trips through the real RAG chain,
and (after a fix made during testing) degrades to a clean user-facing error
message instead of a raw stack trace when the underlying API call fails
(e.g. on a rate limit).

### 7.3 Containerization & Deployment

`Dockerfile`: multi-stage build — a `builder` stage installs dependencies
into an isolated prefix, and the runtime stage copies over only that
prefix plus source code, keeping build tools out of the final image. The
build also runs `src/ingest.py` against `data/raw/`, baking a populated
vector index into the image so the container is immediately queryable on
start with no cold-start ingestion step.

`GEMINI_API_KEY` is never baked into the image — it's read from the
environment at container start (`docker run -e GEMINI_API_KEY=...`, or an
env file passed via `--env-file`), matching the same runtime-only check
`rag_app.build_chain()` already enforces for local runs. For an actual
hosted deployment this would come from the platform's secret store instead
of a command-line flag.

**Deployment evidence:** `docker build -t syllabot .` completes cleanly
end-to-end, including the build-time `RUN python src/ingest.py` step,
which indexed all 12 source documents into 145 chunks inside the image.
`docker run -p 8501:8501 -e GEMINI_API_KEY=... syllabot` starts the
container and serves the Streamlit UI immediately (no cold-start ingest),
confirmed via `docker ps` (container up, port mapped) and an HTTP 200
response from the running container. Full command output is captured in
`docs/evidence/docker_deployment_log.txt`.

## 8. Limitations

- **Closed-weight generator:** no fine-tuning is possible on
  `gemini-3.5-flash`; all behavioral customization goes through prompting.
- **Free-tier API quota:** `gemini-3.5-flash`'s free tier caps at 20
  requests/day per project, which throttles live testing and demoing more
  than the application logic itself.
- **In-memory conversational history:** `InMemoryChatMessageHistory` does
  not persist across server restarts; a production deployment would need a
  persistent session store.
- **Retrieval disambiguation is a workaround, not a root fix:** BM25 +
  course-label prefixing compensates for near-duplicate syllabus boilerplate
  across courses; a fine-tuned embedding model (§7.1) would address this
  more directly.
- **Single flat document collection:** the corpus is homogeneous
  (syllabus chunks only); a mixed-content corpus (e.g. structured calendar
  data alongside free-text policy) would likely need LlamaIndex-style
  routing, per `docs/framework_comparison.md`.

## 9. Lessons Learned

- Live end-to-end testing (not just unit-level checks) surfaced a real bug
  (§6) that static review of the retrieval logic alone would have missed —
  the fault was in how conversation history biased query rewriting, not in
  retrieval itself.
- Workarounds applied at the data layer (course-label prefixing,
  section-aware chunking) can resolve a retrieval-quality problem more
  cheaply than reaching for fine-tuning, and are worth exhausting first.
- Keeping the web interface (`web_app.py`) as a thin wrapper around the
  same `build_chain()` used by the CLI (`rag_app.py`) meant the interface
  checkpoint required no changes to RAG logic — only presentation and error
  handling — which kept the surface area for new bugs small.
