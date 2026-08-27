# SyllaBot Presentation Guide
### Checkpoint 1 & 2 — Data Pipeline + Vector Search

**How to use this manual:** Follow it top to bottom during the actual
presentation. Anything in <span style="color:#16a34a">**green**</span> is
what you say out loud — read it naturally, in your own words if you're
comfortable. **Result:** lines tell you what the output on screen means,
so you can point at it confidently. Plain bullets are backup facts, only
use those if a panelist asks for more detail.

---

## PART A — Explain the Proposal (before touching the keyboard)

<span style="color:#16a34a">**SAY:** "Good [morning/afternoon]. We're presenting Checkpoint 1 and 2 of SyllaBot — a RAG chatbot, which stands for Retrieval-Augmented Generation, that answers questions about your own course syllabi using only what's actually written in them, never a guess."</span>

<span style="color:#16a34a">**SAY:** "The problem: a student with several classes has to track grading weights, late policies, and textbooks scattered across separate documents. SyllaBot lets them just ask instead of digging through PDFs — and it's built to say 'I don't know' rather than confidently give a wrong deadline or grade percentage."</span>

<span style="color:#16a34a">**SAY:** "Today we're covering Checkpoint 1, preparing the data, and Checkpoint 2, making that data searchable and keeping the AI honest — and we'll run the actual code live."</span>

---

## PART B — Live Run-Through

Run each command in order: say the **before** line, run it, then say the
**after** line and use **Result** to explain what's on screen.

### Command 1 — Download the syllabi

```
python src/fetch_raw_syllabi.py
```

<span style="color:#16a34a">**SAY (before):** "First we collect our data — 12 real course syllabi scraped from MIT OpenCourseWare, MIT's free, legal-to-use course archive."</span>

<span style="color:#16a34a">**SAY (after):** "Here's what we got."</span>

**Result:** raw `.txt` files saved in `data/raw/` — one per course, still
full of menu/footer junk, exactly as scraped, nothing cleaned yet.

- Covers CS, Math, Stats, Economics, Psychology, Data Science, Linear
  Algebra, Database Systems
- Pages span 1999–2024, so formatting isn't consistent — a real test for
  our cleaning code

### Command 2 — Clean the text

```
python src/preprocess.py
```

<span style="color:#16a34a">**SAY (before):** "Scraped pages come with navigation menus, cookie pop-ups, and sidebar junk mixed into the real content, so this step strips all of that out."</span>

<span style="color:#16a34a">**SAY (after):** "This is the same document, before and after."</span>

**Result:** cleaned `.txt` files in `data/processed/` — only real syllabus
prose remains (grading policy, schedule, readings), plus a word/sentence
count report confirming how much junk was removed per document.

- One rule handled all 12 pages: real text sits between the *last*
  "Syllabus" line and the *last* "Course Info" line
- Edge case: the Psychology page was split across sub-pages, so we
  separately filtered a stray "Next" link

### Command 3 — Chunk + embed

```
python src/generate_embeddings.py
```

<span style="color:#16a34a">**SAY (before):** "A whole syllabus is too long to search at once, so this step cuts each one into smaller chunks by section, then converts each chunk into a numeric fingerprint called an embedding."</span>

<span style="color:#16a34a">**SAY (after):** "Every chunk is also tagged with its course name so it's never ambiguous which class it came from."</span>

**Result:** 145 chunks total across all 12 syllabi, each turned into a
384-number vector, saved to `data/embeddings/`.

- Chunked by section (e.g. "Late Days"), not fixed word count — avoids
  splitting a rule from its explanation
- Model used: `all-MiniLM-L6-v2`

### Command 4 — Index into the vector database

```
python src/vector_store.py
```

<span style="color:#16a34a">**SAY (before):** "All 145 embeddings now go into Chroma, a database built specifically for fast similarity search."</span>

<span style="color:#16a34a">**SAY (after):** "This also runs a couple of example searches so you can see it actually pulling back relevant chunks."</span>

**Result:** a persistent index saved to `data/chroma_db/`, plus sample
query output showing the top-matching chunks for a couple of test
questions.

- Persistent means we don't have to re-embed everything on every run

### Command 5 — Compare distance metrics

```
python src/compare_distance_metrics.py
```

<span style="color:#16a34a">**SAY (before):** "There's more than one way to measure 'how similar' two chunks are, so we tested three — Cosine similarity, Euclidean distance, and Dot product — on the same test questions."</span>

<span style="color:#16a34a">**SAY (after):** "All three agreed on the best matches, but we chose Cosine going forward."</span>

**Result:** a side-by-side ranking table per query; Cosine and Dot product
matched 3/3 on top results, confirming Cosine is a safe, standard choice
— it's also Chroma's default.

- Reason they matched: our embeddings are unit-length, so Cosine and Dot
  product rank almost identically; Euclidean is more sensitive to length

### Command 6 — Grounded Q&A demo (needs `GEMINI_API_KEY`)

```
python src/api_integration_demo.py
```

<span style="color:#16a34a">**SAY (before):** "This last command runs the full pipeline live — your question goes in, we retrieve matching chunks, and hand them to Gemini with a strict prompt telling it exactly how to answer."</span>

<span style="color:#16a34a">**SAY (after):** "Watch it cite its source — and if I ask something off-topic, watch it refuse instead of guessing."</span>

**Result:** a grounded answer citing the source course for a real
question, and an explicit refusal ("I couldn't find that in your
uploaded syllabi") for an off-topic one — proving the honesty rule
actually works, not just in theory.

- Rules given to the AI: answer only from retrieved excerpts, cite the
  course, never invent a percentage/date/policy
- Multi-course questions get answered per-course, never merged or
  skipped
- Off-topic questions are flagged before the AI even attempts an answer

---

## PART C — Anticipated Questions

| Question | Answer |
|---|---|
| Why clean the data first? | Junk text would get embedded too and confuse search results |
| Why chunk by section, not fixed size? | Avoids splitting a rule from its explanation across two chunks |
| Why Cosine over Euclidean? | Our embeddings are unit-length, so Cosine measures what matters — direction — and it's the standard for this model |
| Why does the bot refuse sometimes? | A wrong guess about a real deadline/grade is worse than no answer |
| Why MIT OpenCourseWare? | Free, legal, and realistically messy — a real test for our cleaning code |
| What about a totally unrelated question? | A low-confidence check flags it out-of-scope before the AI tries to answer |

---

## Closing line

<span style="color:#16a34a">**SAY:** "So — Checkpoint 1 turned messy raw syllabi into clean, well-organized chunks. Checkpoint 2 made those chunks searchable and made sure the AI only answers with what's actually true, citing sources and saying 'I don't know' when it should. That's Checkpoint 1 and 2 — thank you, happy to take questions."</span>
