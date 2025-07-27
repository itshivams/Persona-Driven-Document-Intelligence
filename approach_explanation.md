# Approach Explanation — Persona‑Driven Document Intelligence&#x20;

This document outlines the design and implementation of a **generic document intelligence system** that extracts and ranks the most relevant sections from a collection of PDFs, tailored to any persona and their specified task.

---

## 1. Ingestion & Chunking

1. **Text extraction**: Use `extract_text_by_page()` to convert each PDF page into raw text.
2. **Section chunking**: Each page is split at the first blank line into `(heading, body)` pairs via `chunk_page()`. Headings are detected via regex, with a fallback to “Untitled Section” when no clear heading exists.

**Purpose**: Defines logical content units for downstream semantic analysis.

---

## 2. Embedding

* **Model**: Sentence-Transformers `all-MiniLM-L6-v2` provides 384‑dim dense vectors.
* **Query formation**: Concatenate persona description, job objective, and a generic stem of domain‑relevant keywords.
* **Section embeddings**: Compute vector for each `(heading + body)` to represent semantic content.

**Purpose**: Enables measuring semantic similarity between the user’s need and each section.

---

## 3. Robust Section Ranking

Each section is scored by combining several factors:

| Component                  | Weight | Role                                                          |
| -------------------------- | ------ | ------------------------------------------------------------- |
| **Cosine similarity**      | 0.55   | Core semantic relevance to the user’s query                   |
| **Topical boost**          | 0.20   | +static themes (e.g. domain buckets) + dynamic corpus tokens  |
| **Brevity bonus**          | 0.05   | Rewards concise, high‑information sections                    |
| **Heading penalty**        | -0.25  | Demotes generic headings (“Untitled Section” or “Conclusion”) |
| **Soft diversity penalty** | -0.05× | Slightly penalizes multiple top slots from the same document  |

### Static & Dynamic Boosts

* **Static**: Predefined keyword buckets matching high‑value themes (configurable per domain).
* **Dynamic**: Auto-extract top N frequent tokens from the corpus at runtime; sections containing these terms receive an adaptive boost.

**Purpose**: Balances precision (static) with adaptability (dynamic) across any document collection.

---

## 4. Sub‑section Refinement (Abstractive Summarization)

* **Model**: `t5-small` via a `Summarizer` wrapper for CPU-only abstractive summaries.
* **Process**: Feed each top section’s body text to T5 with a “summarize:” prompt to generate a fluent paragraph summary of key points.

**Purpose**: Produces cohesive, human‑readable snippets that distill the essence of each section.

---

## 5. Performance & Constraints

* **CPU‑only**: No GPU dependency.
* **Memory**: Total image size ≤ 1 GB (includes MiniLM and T5‑small models).
* **Latency**: Complete end‑to‑end for 3–5 documents in ≤ 60 seconds on standard hardware.

**Purpose**: Ensures feasibility for offline or resource‑constrained environments.

---

## 6. Generalization

* **Persona‑agnostic**: Change `persona.json` and `job.json` to any role and task (e.g. analyst, researcher, student).
* **Domain‑agnostic**: Update static keyword buckets to match your corpus (e.g. financial, scientific, legal) without code changes.
* **Adaptive**: Dynamic token boost automatically surfaces new themes in any PDF set.
* **Pluggable summarization**: Swap or fine‑tune the summarizer model for domain‑specific style.
