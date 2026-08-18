# Checkpoint 3 — Framework Comparison Note: LangChain vs. LlamaIndex

SyllaBot's RAG application (`src/rag_app.py`) is built on **LangChain**.
This note compares how LangChain and LlamaIndex approach data
routing/node indexing, and explains what we actually used and why.

## How each framework approaches indexing and routing

**LlamaIndex** is built around the idea of a **document → Node** pipeline
feeding purpose-built **Index** structures (`VectorStoreIndex`,
`SummaryIndex`, `TreeIndex`, `KeywordTableIndex`...), each optimized for a
different retrieval pattern. On top of that sit **Query Engines** and
**Routers** — a `RouterQueryEngine` can inspect an incoming question and
dispatch it to whichever index/engine is best suited (e.g., a summary
question goes to a `SummaryIndex`, a specific-fact question goes to a
`VectorStoreIndex`). Routing and node structure are first-class citizens
of the framework; indexing strategy is the main thing you configure.

**LangChain** treats retrieval as one composable step in a general
**chain/graph of Runnables** (LCEL). There's a `VectorStore` /
`Retriever` abstraction, but no equivalent "pick the right index type"
router built into the core retrieval path — if you want query routing,
you build it yourself as an explicit branch in the chain (e.g., an LLM
call that classifies the query, then a conditional edge to a different
retriever). What LangChain is comparatively stronger at is everything
*around* retrieval: prompt templates, output parsers, chat memory,
agents/tools, and chaining multiple LLM calls together into one
pipeline — which is exactly what SyllaBot needs (grounded system prompt +
retrieval + multi-turn memory in one chain).

## What we used and why

SyllaBot's core requirement for Checkpoint 3 isn't "route between multiple
index types" — our data is a single flat collection of syllabus chunks in
one Chroma vector store — it's "retrieve relevant chunks, ground a
response in them, and remember the conversation across turns." That maps
directly onto LangChain's strengths:

- **`langchain_chroma.Chroma` + `.as_retriever()`** — thin wrapper around
  our existing Chroma collection from Checkpoint 2 (`src/vector_store.py`).
  We supply our own `Embeddings` subclass wrapping the same
  `sentence-transformers/all-MiniLM-L6-v2` model used throughout the
  project, so indexing and querying stay on the same model chosen and
  justified in Checkpoint 1.
- **LCEL (`RunnablePassthrough.assign(...)  | prompt | llm`)** — composes
  retrieval, prompt formatting, and generation into a single runnable
  chain. Retrieval isn't a separate "index type" decision here; it's just
  one step that runs on every query.
- **`RunnableWithMessageHistory`** — gives us per-session conversational
  memory (Checkpoint 3 deliverable 12) without hand-rolling a message
  buffer, and integrates directly with the same chain object.

If SyllaBot's scope grows to include genuinely different content types
that need different retrieval strategies — e.g., a separate index for
structured data like exam dates and calendars vs. free-text policy
prose — LlamaIndex's router/index-type model would become the more
natural fit, since that's precisely the "which index should handle this
query" problem it's designed to solve. For a single homogeneous document
collection with a conversational interface on top, LangChain's
general-purpose chain composition was the better match.
