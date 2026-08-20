# Checkpoint 4 — Model Customization Note

SyllaBot currently uses two models, and they sit on opposite sides of the
"can we fine-tune this" line:

| Role | Model | Weights |
|---|---|---|
| Generation (answers + query condensing) | `gemini-3.5-flash` (Google Gemini API) | Closed — API-only |
| Embeddings (retrieval) | `sentence-transformers/all-MiniLM-L6-v2` | Open-source, self-hosted |

This split determines where PEFT/LoRA/QLoRA is even applicable, so it's the
starting point for this note.

## Why the generator is used as-is

`gemini-3.5-flash` is served entirely behind Google's API — we send text,
we get text back, and there is no mechanism to attach a LoRA adapter or
otherwise alter its weights. Any customization has to happen outside the
model: prompt engineering (`docs/prompt_engineering.md`) and grounding via
retrieval are the only levers available for a closed-weight model, which is
exactly what SyllaBot already leans on. Fine-tuning is simply not on the
table here, and treating "we didn't fine-tune Gemini" as a gap would be a
category error — it's a constraint of using a hosted, closed model, not a
skipped step.

## Where PEFT would plausibly apply, if we had more time

### 1. LoRA on the embedding model (highest-value target)

`all-MiniLM-L6-v2` is a small, open-weight BERT-family encoder — a
realistic LoRA target, and it points directly at a real weakness we've
already had to work around. As documented in `src/rag_app.py`, near-identical
boilerplate across syllabi (e.g. every course's "Collaboration Policy" reads
almost the same) makes dense retrieval alone unreliable at telling two
courses apart; we currently patch this at the *data* layer, by prefixing
every chunk with its course label (`doc_label()` in
`src/generate_embeddings.py`) and blending in BM25 keyword search
(`EnsembleRetriever` in `rag_app.py`).

A LoRA fine-tune of MiniLL's attention projections (`query`/`value`), trained
with a contrastive loss (e.g. `MultipleNegativesRankingLoss` via
`sentence-transformers`) on pairs like *("late policy for the database
systems course", the matching chunk)* vs. *(same query, the near-duplicate
chunk from a different course)*, would push the encoder itself to separate
these embeddings — addressing the root cause instead of compensating for it
with keyword search and text prefixing. Because LoRA only trains small
rank-decomposition matrices injected into the attention layers (base weights
frozen), this is cheap: MiniLL is ~22M parameters, so even full fine-tuning
would fit on a single consumer GPU, and LoRA makes it trivially so (a handful
of MB of trainable parameters, no gradient checkpointing needed).

### 2. QLoRA on the generator, if we self-hosted instead of using the Gemini API

If SyllaBot's generator were swapped for an open-weight instruction-tuned
model (e.g. Llama 3.1 8B Instruct or Mistral 7B Instruct, run locally via
something like vLLM or Ollama instead of a hosted API), QLoRA becomes the
relevant technique: 4-bit NF4 quantization of the frozen base weights plus
LoRA adapters on the attention projections (`q_proj`, `k_proj`, `v_proj`,
`o_proj`) makes it possible to fine-tune a 7-8B model on a single consumer
GPU with 16GB VRAM, since only the small adapter matrices (typically rank
8-16) need full-precision gradients.

The target for that fine-tune would be narrow and specific, not general
"teach it about syllabi" — the retrieval step already supplies course
content, so the model doesn't need new facts memorized. What it *would*
benefit from is tighter adherence to SyllaBot's output contract: always
citing the source course, always using the exact refusal phrasing ("I
couldn't find that in your uploaded syllabi") instead of a paraphrase, and
resisting the temptation to fill gaps with general knowledge under
adversarial phrasing. A few hundred synthetic (question, grounded-context,
ideal-answer) triples generated from our own syllabus corpus would be enough
supervision for that kind of stylistic/behavioral tightening.

## What we actually did instead, and why that's the right call for now

We did not perform either fine-tune for this checkpoint. Two reasons:

1. **The cheaper fix already works.** The BM25 + course-label-prefixing
   workaround for retrieval ambiguity is verified working (see the
   Checkpoint 3 demo transcript and the condense-prompt fix applied after
   testing) without needing a training run, labeled contrastive data, or
   GPU time. Reaching for LoRA before exhausting prompt/architecture-level
   fixes would be solving the problem at the more expensive layer first.
2. **Self-hosting the generator to enable QLoRA would trade a working,
   free-tier-accessible API for GPU infrastructure we don't have,** in
   exchange for a behavioral tightening (citation/refusal consistency) that
   the system prompt already handles adequately for a semester project's
   scope.

If we were to extend SyllaBot beyond this course, LoRA-tuning the embedding
model is the concrete next step we'd prioritize — it's the smaller, cheaper
fine-tune, it targets a weakness we can already point to precisely, and it
doesn't require abandoning the hosted Gemini API for generation.
