"""
Persona‑Driven Document Intelligence – v5 + T5 summaries

• dynamic corpus boosts
• soft diversity
• PDF parsing errors are caught and skipped
"""

import argparse, json, math, os, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from pdfminer.pdfparser import PDFSyntaxError

from src.ingestion.pdf_loader import extract_text_by_page
from src.chunker.chunker import chunk_page
from src.models.embedder import Embedder
from src.models.ranker import build_query, compute_boosts
from src.models.summariser import Summarizer
from src.output.formatter import write_output


def safe_to_text(p) -> str:
    if isinstance(p, tuple):
        for itm in p:
            if isinstance(itm, str):
                return itm
            if isinstance(itm, bytes):
                return itm.decode("utf-8", "ignore")
        return ""
    if isinstance(p, bytes):
        return p.decode("utf-8", "ignore")
    return p if isinstance(p, str) else ""


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / den if den else 0.0


def gather_sections(docs: Path, files: List[str], emb: Embedder) -> List[Dict]:
    sections = []
    for pdf in files:
        pdf_path = docs / pdf
        try:
            pages = extract_text_by_page(pdf_path)
        except PDFSyntaxError:
            print(f"[WARNING] Skipping unreadable PDF: {pdf}")
            continue

        for p_no, raw in enumerate(pages, start=1):
            page = safe_to_text(raw)
            if not page.strip():
                continue
            for head, body in chunk_page(page):
                vec = emb.embed(f"{head}\n{body}".strip())
                sections.append({
                    "document": pdf,
                    "page_number": p_no,
                    "section_title": head,
                    "text": body,
                    "vector": vec,
                })
    return sections


def rank_sections(q_vec: List[float], secs: List[Dict], top_k: int = 10) -> List[Dict]:
    boosts = compute_boosts([s["text"] for s in secs])
    seen = defaultdict(int)
    scored: List[Tuple[float, Dict]] = []

    for idx, s in enumerate(secs):
        base = cosine(q_vec, s["vector"])
        boost = boosts[idx]
        brev = 1 / math.sqrt(max(1, len(s["text"].split())))
        heading_pen = -0.25 if s["section_title"].lower() in {"untitled section", "conclusion"} else 0
        diversity_pen = -0.05 * seen[s["document"]]

        score = 0.55 * base + 0.20 * boost + 0.05 * brev + heading_pen + diversity_pen
        scored.append((score, s))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = []
    for _, sec in scored:
        seen[sec["document"]] += 1
        top.append(sec)
        if len(top) == top_k:
            break

    for i, sec in enumerate(top, 1):
        sec["importance_rank"] = i
    return top


def refine(top_secs: List[Dict], summarizer: Summarizer) -> List[Dict]:
    refined = []
    for sec in top_secs:
        summary = summarizer.summarize(sec["text"], max_len=64, min_len=20)
        refined.append({
            "document": sec["document"],
            "page_number": sec["page_number"],
            "refined_text": summary,
        })
    return refined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    inp = Path(args.input)
    docs_dir = inp / "docs"
    pdfs = sorted(os.listdir(docs_dir))

    persona = json.load((inp / "persona.json").open())
    job_meta = json.load((inp / "job.json").open())

    embedder = Embedder()
    summarizer = Summarizer(device="cpu")

    query_vec = embedder.embed(build_query(persona, job_meta))

    sections = gather_sections(docs_dir, pdfs, embedder)
    top_secs = rank_sections(query_vec, sections, top_k=args.top_k)
    sub_secs = refine(top_secs, summarizer)

    metadata = {
        "input_documents": pdfs,
        "persona": persona["persona"],
        "job_to_be_done": job_meta.get("job_to_be_done") or next(iter(job_meta.values())),
        "processing_timestamp": datetime.utcnow().isoformat(),
    }

    write_output(metadata, top_secs, sub_secs, args.output)


if __name__ == "__main__":
    main()
