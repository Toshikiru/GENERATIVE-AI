# SyllaBot Defense Guide (Explain-Like-I'm-5 Edition)
### Checkpoint 1 & Checkpoint 2 only

This is your cheat sheet for defense. Every part is explained in the
simplest way possible first, then followed by the "real talk" version with
the actual technical words, so you can match your answer to how technical
the question sounds.

---

## 0. The One-Sentence Story

> **Imagine you have a huge pile of messy class syllabi. SyllaBot is a
> robot librarian that reads all of them, remembers where everything is,
> and can answer your questions honestly — and says "I don't know" instead
> of making stuff up.**

Checkpoint 1 = teaching the robot to **read and organize** the syllabi.
Checkpoint 2 = teaching the robot **how to search** its memory and
**how to talk** without lying.

---

## PART 1 — CHECKPOINT 1: Getting the Papers Ready

### Step 1.1 — Downloading the syllabi (`fetch_raw_syllabi.py`)

**Like I'm 5:** Imagine 12 different classes each gave you a syllabus
paper. Instead of you walking around collecting them by hand, we built a
robot arm that visits each class's website and photocopies the syllabus
page for us. We got 12 real ones from MIT's website (MIT OpenCourseWare),
covering things like Math, Psychology, Data Science, and Databases —
kind of like a student who's taking a bunch of different subjects at once.

**Real talk:** We scraped 12 real course syllabus pages from MIT
OpenCourseWare (Creative Commons licensed, so it's legal to use), spanning
CS, Math, Statistics, Economics, Psychology, Data Science/ML, Linear
Algebra, and Database Systems.

> **If asked "why MIT OCW?"** — it's free, real, legally reusable, and
> varied enough (1999–2024, different formats) to stress-test our cleaning
> code instead of using fake, too-clean sample text.

---

### Step 1.2 — Cleaning the mess (`preprocess.py`)

**Like I'm 5:** Photocopying a webpage doesn't just give you the syllabus —
it also grabs the website's menu buttons, the cookie pop-up, the little
picture icons, and the footer at the bottom, over and over. It's like
copying a page from a coloring book but the crayon scribbles from other
kids are still on it. We had to erase all the scribbles and keep only the
real syllabus writing.

**How we erased the scribbles (the smart trick):** Instead of writing
separate cleanup rules for each of the 12 pages (which would take forever
and break easily), we noticed **every single page follows the same
pattern**: the real syllabus text always sits between the *last* time the
word "Syllabus" appears and the *last* time "Course Info" appears on the
page. So we just say: "grab everything between those two markers" — one
simple rule that works on all 12 pages, even though they look totally
different.

We also had to fix leftover weird symbols like `&rsquo;` (which is just a
fancy computer way of writing an apostrophe `'`) so the text reads
normally, and we removed extra blank spaces.

**Real talk:** Cleaning removes site navigation, duplicated course-info
sidebars, icon-label artifacts (e.g. `theaters`, `auto_stories`), and the
global footer/cookie modal. The extraction rule exploits a structural
pattern common to the whole platform — real syllabus prose sits between the
*last* "Syllabus" line and the *last* "Course Info" line — which
generalized across all 12 documents despite them spanning pages from
1999–2024 with very different lengths and structures. HTML entities are
decoded and whitespace is normalized afterward.

> **Curveball they might ask:** "What if a page didn't follow that
> pattern?" — One page (Introduction to Psychology) was actually spread
> across multiple sub-pages, so our one photocopy only grabbed one part
> plus a stray "Next" pagination link, which we filtered out separately.
> We didn't assume one fixed template — we designed around structural
> markers common to the platform, which is why it survived that edge case.

---

### Step 1.3 — Splitting sentences (Tokenization)

**Like I'm 5:** Now that the writing is clean, we need to teach the
computer to see where one sentence ends and the next begins, and where one
word ends and the next begins — like teaching a little kid to find the
spaces between words and the periods between sentences.

**Real talk:** We used NLTK's `sent_tokenize`/`word_tokenize` (a standard
toolkit for splitting text into sentences and words), with a regex-based
fallback if NLTK's data isn't available. Results (word/sentence/token
counts per document) are documented in `docs/preprocessing_report.md` —
e.g. `01_mathematics_for_computer_science.txt` went from 1131 raw words to
854 cleaned words after junk removal.

---

### Step 1.4 — Cutting into bite-sized pieces + giving each piece a
"fingerprint" (`generate_embeddings.py`)

**Like I'm 5:** A whole syllabus is too big to search through quickly, so
we cut it into smaller chunks — like cutting a big pizza into slices. But
we're not cutting randomly! We cut **at the natural section breaks**, like
"Grading Policy" or "Late Days," so a whole rule stays together on one
slice instead of getting sliced in half and losing its meaning.

Then, for every slice, we create a **fingerprint** — a list of numbers that
represents what that slice is *about*. Two slices that talk about similar
things (like two "late policy" sections from different classes) get
fingerprints that are close to each other, so the computer can quickly tell
"these two are similar" without literally reading every word every time.

We also stamped each slice with a little label at the top, like "Course:
Database Systems," so it never gets confused about which class a slice
belongs to.

**Real talk:** Chunking is section-aware rather than a fixed-size sliding
window — it detects short header-like lines (e.g. "Late Days", "Grading
Policy") and groups the prose that follows into one chunk per section,
splitting only sections that exceed ~120 words. This keeps a fact and the
sentence that names it in the same chunk. Every chunk is prefixed with its
course label via `doc_label()`. The fingerprint (embedding) is generated by
`sentence-transformers/all-MiniLM-L6-v2` (384 numbers per fingerprint), the
same model used consistently for indexing and querying.

> **If asked "why chunk by section instead of just every N words?"** — A
> fixed-size window can slice a policy sentence across two unrelated
> chunks (e.g. cutting "late days" from its penalty percentage into
> separate pieces). Section-aware chunking keeps a fact and its context
> together.

---

## PART 2 — CHECKPOINT 2: Searching and Talking

### Step 2.1 — Putting fingerprints in a smart filing cabinet (`vector_store.py`, Chroma)

**Like I'm 5:** Now we have hundreds of little fingerprint cards (one per
slice). We put them all into a special filing cabinet called **Chroma**
that's really good at one job: given a *new* fingerprint (like your
question), instantly find the cards whose fingerprints look the most
similar — without checking every card one by one like a regular filing
cabinet would.

**Real talk:** `src/vector_store.py` sets up a persistent Chroma vector
database (saved to disk in `data/chroma_db/`, so we don't have to
re-embed everything every time) that indexes all 145 syllabus chunks for
similarity search.

---

### Step 2.2 — How do we measure "similar"? (`compare_distance_metrics.py`)

**Like I'm 5:** Imagine every fingerprint is an arrow pointing in some
direction in space. There are different ways to decide if two arrows are
"close":

- **Cosine similarity** — do the two arrows point in **roughly the same
  direction**? (Doesn't care how long the arrows are, just the direction.)
- **Euclidean distance** — how far apart are the **tips** of the two
  arrows, like measuring with a ruler?
- **Dot product** — a math shortcut that mixes direction *and* length
  together.

We tested all three on the same 3 sample questions (like "What percentage
of my grade comes from the final exam?") and checked which one gave the
best top-3 matches.

**Real talk:** All 145 chunk embeddings from `all-MiniLM-L6-v2` are
effectively unit-length (norm ≈ 1.000). When vectors are unit-normalized,
Cosine similarity and Dot Product become mathematically equivalent for
ranking purposes (confirmed: 3/3 top-k overlap on every test query), while
Euclidean distance ranks slightly differently because it's sensitive to
absolute vector position, not just angle.

**We chose Cosine similarity** because: it's the standard choice for
sentence-transformer embeddings, it's scale-invariant (robust if we ever
swap embedding models), and it's Chroma's default distance space anyway.

> **If asked "why not just use Euclidean since it's simpler?"** — Because
> once embeddings are normalized to length 1, Euclidean distance is really
> just measuring the same "angle" information in a roundabout way, but
> Cosine is the metric actually designed for that job and is what the
> whole embedding ecosystem (Chroma, sentence-transformers) is built
> around.

---

### Step 2.3 — Teaching the robot to be honest (`prompt_engineering.md`)

**Like I'm 5:** Now the fun part — the robot found the right slices, but
we still need to teach it **how to answer**. A robot that just makes stuff
up when it's not sure is dangerous — imagine it confidently telling you
the wrong exam date! So we wrote very strict instructions (called
"prompts") for it to follow, kind of like classroom rules taped to the
robot's forehead:

**Rule Card 1 — Answer honestly:** "Only answer using the slices I gave
you. If you don't see the answer in there, say 'I couldn't find that in
your uploaded syllabi' instead of guessing."

**Rule Card 2 — Handle many classes at once:** "If the student asks about
several classes at the same time, answer for *each* class separately —
don't quietly skip a class just because you don't have much information
about it, and don't mix answers together."

**Rule Card 3 — Know when a question isn't even about class stuff:** "If
the search results barely match the question at all (like someone asking
'What's the capital of France?'), just say this is out of scope instead of
trying to force an answer out of unrelated syllabus text."

**Real talk:** Three structured prompts were designed, each enforcing the
same core defense — **the model is told exactly what counts as a valid
basis for an answer, and is given an explicit, low-friction way to say "I
don't know."**

1. **Grounded Q&A prompt** — the default; answers only from retrieved
   excerpts, must cite the source course, and must use the exact refusal
   phrase when the excerpts don't contain the answer. Never invents a
   percentage, date, or policy.
2. **Multi-Course Aggregation prompt** — used when a query spans multiple
   courses; addresses each course separately, never silently drops a
   course with no matching data, never merges two courses' numbers
   together.
3. **Out-of-Scope Refusal prompt** — a cheap pre-check run *before* the
   expensive grounded-answer prompt, for cases where retrieval confidence
   is low (e.g. an off-topic question). Classifies `RELEVANT` vs.
   `OUT_OF_SCOPE: <reason>` without attempting to answer.

> **Hallucination-prevention example to have ready:** Question: *"What's
> the grading breakdown for my Machine Learning class?"* where retrieval
> actually pulls up the Evolutionary Psychology syllabus (a retrieval
> miss). Without Rule 1, the model would likely pattern-match "grading
> breakdown" and invent a plausible-looking percentage split anyway. With
> Rule 1 in place, the required response is: *"I couldn't find that in
> your uploaded syllabi."*

---

### Step 2.4 — Actually talking to the AI brain (`api_integration_demo.py`)

**Like I'm 5:** All the pieces come together here: take the student's
question → find the closest fingerprint slices in the filing cabinet →
hand those slices plus the rule cards to the AI (Google's Gemini) → get
back a grounded, honest answer.

**Real talk:** `src/api_integration_demo.py` wires the vector search
output into a real Gemini API call using the Grounded Q&A prompt from
§2.3, demonstrating an end-to-end grounded Q&A round trip: retrieve →
inject context into the prompt → generate → return a cited or
explicitly-refused answer.

---

## Quick-Fire Q&A Cheat Sheet

| If they ask... | 5-year-old answer | Real-talk backup |
|---|---|---|
| Why clean the data at all? | The robot can't learn from a page full of menu buttons and cookie pop-ups mixed into the real syllabus text | Boilerplate would otherwise dominate the embedding signal and pollute retrieval |
| Why chunk instead of embedding the whole document? | A whole syllabus is too big and mixes too many topics into one fingerprint; slices let us find just the relevant part | Smaller, topic-coherent chunks give more precise retrieval; a full-doc embedding would average away specific facts |
| Why section-aware chunking, not fixed-size? | So we don't accidentally cut a rule in half between two pieces | Keeps a fact and the sentence naming it in the same chunk, avoiding mid-policy splits |
| Why Cosine over Euclidean/Dot? | We tested all 3 arrow-measuring tricks and Cosine is the one built for this job | Embeddings are unit-normalized, so Cosine ≈ Dot Product ranking-wise; Cosine is Chroma's default and the sentence-transformer standard |
| Why does the robot say "I don't know" sometimes? | Because a wrong guess about a real deadline is worse than no answer at all | Rule 2 of the Grounded Q&A prompt explicitly forbids inventing facts not present in retrieved excerpts |
| Why MIT OpenCourseWare data? | It's free, real, and legal to use, and it's messy enough to be a real test | Creative Commons licensed, spans 1999–2024 with varied structure, good stress test for the cleaning pipeline |

---

## 30-Second Elevator Summary (memorize this one)

> "Checkpoint 1 is about getting clean, well-organized data: we scraped
> real MIT syllabi, stripped out all the website junk using one consistent
> structural rule, split the text into sentences, then cut it into
> meaningful chunks and gave each chunk a numeric fingerprint. Checkpoint 2
> is about making that data *usable and safe*: we stored the fingerprints
> in a Chroma vector database, proved Cosine similarity was the right way
> to measure 'closeness' between them, and wrote strict prompts that force
> the AI to only answer from real retrieved content — and to honestly say
> 'I don't know' instead of making things up."
