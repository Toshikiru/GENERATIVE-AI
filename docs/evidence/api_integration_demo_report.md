# API Integration Demo — Report

Verified run of `src/api_integration_demo.py` (Checkpoint 2, deliverable
7): retrieves relevant syllabus context from Chroma, then generates a
grounded answer via the Gemini API using the Prompt 1 system prompt from
`docs/prompt_engineering.md`.

Kept as a backup in case the live API is temporarily unavailable during
presentation — this is a real, successful run, not a mock-up.

---

## Test 1 — In-scope grounded question

**Question:** "How much is the final exam worth in the microeconomics
course?"

**Retrieval result:**
- Top cosine distance: `0.216` (likely relevant — well under the 0.7
  out-of-scope threshold)
- Retrieved from: `08_principles_of_microeconomics.txt` (top 3 matches)

**SyllaBot's answer:**
> According to the Principles of Microeconomics syllabus, the final exam
> counts as 50% of your grade.
>
> *(Source: `08_principles_of_microeconomics.txt`)*

**What this proves:** retrieval correctly found the right course document,
and the answer is grounded and cited — no invented numbers.

---

## Test 2 — Out-of-scope question

**Question:** "What is the capital of France?"

**Retrieval result:**
- Top cosine distance: `0.867` (likely out-of-scope — above the 0.7
  threshold)
- Retrieved from: `10_intro_cs_programming_python.txt` (top 3 matches —
  the least-bad matches available, since vector search always returns
  *something*)

**SyllaBot's answer:**
> I am sorry, but I can only answer questions related to your course
> syllabi. If you have a question about your course schedule, policies,
> or assignments, please feel free to ask!

**What this proves:** even though retrieval returned *some* chunks (as it
always does), the model correctly refused to answer using unrelated
syllabus content instead of guessing "Paris" from general knowledge.

---

## Reliability note

The script was hardened after this run: `client.models.generate_content`
now retries automatically (up to 2 extra attempts, 5s apart) on a
transient `ServerError` (e.g. `503 UNAVAILABLE` — Gemini's free tier
occasionally reports high demand), and prints a graceful fallback message
instead of crashing if the API is still unavailable after retries. This
report's transcript is the full, unedited output of that hardened script.

Full raw console output: `docs/evidence/api_integration_demo_backup.txt`
