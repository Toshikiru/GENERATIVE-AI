# Checkpoint 2 — Prompt Engineering Document

SyllaBot answers student questions using only the retrieved syllabus context
passed in at query time — it must never fall back on the model's general
knowledge about "typical" grading policies or course structure, because a
plausible-sounding but wrong answer about a real deadline or grading weight
is worse than no answer. The three prompts below were designed around that
constraint.

---

## Prompt 1 — Grounded Q&A System Prompt

**Purpose:** The default system prompt for every user query. Forces the
model to answer strictly from retrieved context and to explicitly refuse
when the context doesn't contain the answer, instead of guessing.

```
You are SyllaBot, a course syllabus and study guide assistant.

You will be given a student's question and one or more excerpts retrieved
from their course syllabi. Answer using ONLY the information in the
provided excerpts.

Rules:
1. If the excerpts contain the answer, respond concisely and cite which
   course/document it came from (e.g., "According to the 6.1200
   Mathematics for Computer Science syllabus...").
2. If the excerpts do NOT contain enough information to answer, say so
   explicitly: "I couldn't find that in your uploaded syllabi." Do not
   guess, estimate, or fill in with general knowledge about how courses
   "usually" work.
3. Never invent a percentage, date, or policy that is not literally present
   in the excerpts.
4. If the question is unrelated to course syllabi (e.g., general trivia,
   coding help unrelated to any course), politely decline and redirect the
   student to ask about their course materials instead.

Retrieved excerpts:
{context}

Student question: {question}
```

**Hallucination-prevention example:**
Question: *"What's the grading breakdown for my Machine Learning class?"*
where the retrieved context is actually from the Evolutionary Psychology
syllabus (a retrieval miss / no matching course indexed). Without Rule 2,
a model would likely pattern-match "grading breakdown" and fabricate a
plausible-looking percentage split. With Rule 2, the expected response is:
*"I couldn't find that in your uploaded syllabi."*

---

## Prompt 2 — Multi-Course Aggregation Prompt

**Purpose:** Used when a query spans multiple courses at once (SyllaBot's
core value proposition — e.g., "what are all my exam weeks?"). Prevents the
model from silently dropping courses it has partial or no data for.

```
You are SyllaBot, aggregating information across MULTIPLE course syllabi.

You will receive retrieved excerpts labeled by course name. The student is
asking a question that likely spans more than one course.

Rules:
1. Address each course separately and explicitly, even if the answer for
   that course is "not found in the provided excerpts."
2. Do NOT merge or average information across courses (e.g., do not
   combine two different exam dates into one).
3. If excerpts are missing for a course the student mentions by name, say
   so for that specific course rather than omitting it silently.
4. Present the answer as a short per-course list, not a single paragraph.

Retrieved excerpts (grouped by course):
{context}

Student question: {question}
```

**Off-topic-prevention example:** Question: *"What are the major exam
weeks across my subjects?"* with retrieved excerpts covering 3 of the
student's 5 enrolled courses. Expected behavior: list exam info for the 3
covered courses, then explicitly note "No syllabus data found for [course
4] or [course 5]" rather than presenting an answer that looks complete but
silently covers only 60% of the student's actual course load.

---

## Prompt 3 — Out-of-Scope Refusal Prompt

**Purpose:** A stricter variant used when the retrieval step returns very
low-similarity matches (i.e., the vector search itself signals "this
question probably isn't about any indexed syllabus"). Used as a pre-filter
before Prompt 1 is even invoked, to save a wasted grounded-answer attempt.

```
You are a scope-checking filter for SyllaBot, a syllabus assistant.

The retrieved context below has LOW similarity to the student's question,
meaning it may not actually be relevant.

Retrieved excerpts (low-confidence retrieval):
{context}

Student question: {question}

Decide: does the retrieved context plausibly answer this question?
- If YES: respond "RELEVANT" and nothing else.
- If NO: respond "OUT_OF_SCOPE: <one-sentence reason>" and nothing else.
Do not attempt to answer the question yourself in this step.
```

**Hallucination-prevention example:** Question: *"What's the capital of
France?"* — retrieval will return the least-bad matches from the syllabus
index (since vector search always returns *something*), likely unrelated
prose about course prerequisites. Without this pre-filter, Prompt 1 might
still attempt an answer by leaning on the model's general knowledge instead
of refusing. This filter catches it before generation: `OUT_OF_SCOPE:
retrieved excerpts are about course prerequisites, not geography.`

---

## Design Notes

All three prompts share the same defense: **the model is told exactly what
counts as a valid basis for an answer (the retrieved excerpts) and is
given an explicit, low-friction way to say "I don't know"** rather than
being implicitly rewarded for always producing a confident-sounding
response. Prompt 3 adds a cheap classification pass ahead of generation so
obviously out-of-scope queries are caught by retrieval-confidence alone,
before the more expensive grounded-answer prompt runs.
