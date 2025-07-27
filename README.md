# Persona‑Driven Document Intelligence

This repository implements a **generic, offline, CPU‑only** pipeline to extract, rank, and summarize the most relevant sections from a collection of PDFs, customized to any persona and their specific task.

## Features

* **Modular stages**: ingestion → chunking → embedding → ranking → summarization → output
* **Semantic embeddings**: uses Sentence‑Transformers `all‑MiniLM‑L6‑v2` for lightweight, 384‑dim vectors
* **Robust ranking**:

  * **Cosine similarity** for semantic relevance
  * **Static keyword boosts** (configurable per domain)
  * **Dynamic corpus boosts** (auto‑learned top tokens)
  * **Brevity bonus** for concise sections
  * **Heading penalty** to de‑prioritize generic titles
  * **Soft diversity** to balance coverage across documents
* **Abstractive summaries**: integrates `t5‑small` for fluent, paragraph‑style `refined_text`
* **High performance**: < 1 GB image, CPU‑only, end‑to‑end < 60 s for 3–5 PDFs (20 pages each)
* **Domain‑agnostic**: easily swap static buckets and persona/task definitions via JSON; no code changes

## Getting Started

### Prerequisites

* Docker (Engine ≥ 20.10)
* Linux/macOS/Windows WSL2

### Project Structure

```
├── Dockerfile
├── requirements.txt
├── README.md                # This file
├── approach_explanation.md  # Detailed methodology
├── src/
│   ├── main.py              # Entry point
│   ├── ingestion/pdf_loader.py
│   ├── chunker/chunker.py
│   ├── models/embedder.py
│   ├── models/ranker.py
│   ├── models/summarizer.py
│   └── output/formatter.py
└── sample_input/
    ├── docs/                # PDF files
    ├── persona.json         # Persona metadata
    └── job.json             # Job‑to‑be‑done metadata
```

### Building the Docker Image

```bash
docker build -t persona_doc_intel .
```

### Running the Pipeline

1. **Prepare input**:

   * Place your PDF files under `my_input/docs/`
   * Define `my_input/persona.json`:

     ```json
     {"persona": "Your Persona Title"}
     ```
   * Define `my_input/job.json`:

     ```json
     {"job_to_be_done": "Specific task description for the persona."}
     ```

2. **Run**:

```bash
docker run --rm \
  -v "$PWD/my_input:/input" \
  -v "$PWD/my_results:/output" \
  persona_doc_intel \
  --input /input --output /output/results.json --top_k 10
```

3. **Output**:

   * Check `my_results/results.json` for the final structured JSON:

     ```json
     {
       "metadata": {...},
       "extracted_sections": [...],
       "subsection_analysis": [...]
     }
     ```

## Customization

* **Persona/Job**: edit `persona.json` and `job.json` to any role and task.
* **Static buckets**: modify `STATIC_BUCKETS` in `src/models/ranker.py` to tune domain themes.
* **Summary length**: tweak `max_len` and `min_len` in `src/models/summarizer.py`.

## Performance & Limitations

* Designed for **small to medium** PDF collections (3–10 docs, up to \~100 pages total).
* **Scalability**: embedding and summarization are batchable but CPU‑bound; expect linear time with document size.
* **Robustness**: non‑PDF or corrupted files are skipped with a warning.

## License

MIT License

---

*Built for Round 1B: Persona‑Driven Document Intelligence.*
