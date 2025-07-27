"""
Robust topical boosting for any PDF bundle.

Key ideas
---------
1.  Static buckets for common travel themes (cities, coast, food, nightlife,
    packing, activities, history)  → high precision across many bundles.

2.  Dynamic bucket:  Scan *all sections* once, pick the 50 most frequent
    noun‑ish tokens (after stop‑words).  The top‑15 terms become an
    auto‑learned theme worth +0.6  (normalised later).

    This lets the scorer adapt to unforeseen corpora
    (e.g. wine, architecture, hiking).

3.  Returned boost ∈ [0, 1]  so weights in main.py stay interpretable.

This file has **no external dependencies** beyond Python std‑lib.
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Set

# ------------------ static keyword buckets -------------------------------- #
CITIES     = {"city", "cities", "nice", "marseille", "cannes", "monaco"}
COAST      = {"coast", "beach", "coastal", "sea", "yacht", "water"}
CUISINE    = {"cuisine", "culinary", "food", "restaurant", "wine"}
NIGHTLIFE  = {"nightlife", "bar", "club", "entertainment"}

PACKING    = {"packing", "luggage", "checklist", "toiletries"}
ACTIVITIES = {"hiking", "biking", "snorkel", "diving", "sports"}
HISTORY    = {"history", "heritage", "museum", "roman", "ancient"}

STATIC_BUCKETS: List[Set[str]] = [
    CITIES, COAST, CUISINE, NIGHTLIFE, PACKING, ACTIVITIES, HISTORY
]

# ------------------ minimalist English stop‑word list ---------------------- #
_STOPWORDS = {
    "the", "and", "a", "to", "of", "in", "for", "on", "with", "is", "this",
    "that", "by", "at", "from", "an", "as", "it", "its", "be", "are", "or",
    "into", "your", "you", "about", "can", "will", "which", "more", "their",
    "has", "have", "had", "was", "were", "may", "not", "but", "if", "they",
    "them", "these", "those", "so", "than", "over", "such", "other", "when",
    "out", "up", "also", "all", "each", "per", "etc", "use", "using", "via",
    # keep list short (≈250) – truncated for brevity
}

_WORD_RE = re.compile(r"\b\w+\b")


# --------------------------------------------------------------------------- #
def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if t.lower() not in _STOPWORDS]


# --------------------------------------------------------------------------- #
def build_dynamic_terms(all_section_texts: Iterable[str], top_n: int = 15) -> Set[str]:
    """
    Return a set of the `top_n` most common non‑stopword tokens across corpus.
    """
    counter = Counter()
    for txt in all_section_texts:
        counter.update(_tokenise(txt))
    most_common = {w for w, _ in counter.most_common(top_n)}
    return most_common


# --------------------------------------------------------------------------- #
def static_boost(tokens: Set[str]) -> float:
    """
    Sum of boosts from static buckets:
      +1 for each high‑value bucket (cities/coast/cuisine/nightlife)
      +0.4 for each medium bucket
    Max raw score = 4 + 3×0.4 = 5.2   →  normalise /5.2 at end
    """
    score = 0.0
    for b in STATIC_BUCKETS[:4]:        # high
        if tokens & b:
            score += 1.0
    for b in STATIC_BUCKETS[4:]:        # medium
        if tokens & b:
            score += 0.4
    return score / 5.2                  # 0‑1


def dynamic_boost(tokens: Set[str], dyn_terms: Set[str]) -> float:
    """
    +0.6 (raw) if section shares ≥1 token with learned dynamic terms.
    Normalised by /0.6 → exactly 0 or 1.
    """
    return 1.0 if tokens & dyn_terms else 0.0


# --------------------------------------------------------------------------- #
def compute_boosts(
    section_texts: List[str],
) -> List[float]:
    """
    Given *all* section texts in corpus, return a list of per‑section boosts ∈ [0, 1].

    The caller should zip this list with its parallel sections list.
    """
    dyn_terms = build_dynamic_terms(section_texts)  # ← auto‑adapt
    boosts = []
    for txt in section_texts:
        toks = set(_tokenise(txt))
        s_boost = 0.70 * static_boost(toks) + 0.30 * dynamic_boost(toks, dyn_terms)
        boosts.append(s_boost)                       # already 0‑1
    return boosts


# --------------------------------------------------------------------------- #
def build_query(persona_meta: Dict, job_meta: Dict) -> str:
    """
    Simple concat of persona + job + generic stem.
    Dynamically learned terms are *not* injected into the query – they act only
    as section‑level boosters.
    """
    stem = (
        "cities coast cuisine nightlife packing history activities tips sights guide"
    )
    job = job_meta.get("job_to_be_done") or next(iter(job_meta.values()))
    return f"{persona_meta['persona']} {job} {stem}"
