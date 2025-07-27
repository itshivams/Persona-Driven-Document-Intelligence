"""
Heading‑aware chunker.

A heading is any first line that starts with an optional bullet (• – —),
optionally some whitespace, then ≥ 4 alphanumeric characters (allows mixed
case).  Fallback → "Untitled Section".
"""

import re
from typing import Iterator, Tuple

# Matches e.g.   "Tips and Tricks", "- Water Sports", "•  City Exploration"
HEADING_RE = re.compile(r"^[-–—•*]?\s*([A-Za-z][A-Za-z0-9 &\\-]{3,})$")


def detect_heading(first_line: str) -> str:
    m = HEADING_RE.match(first_line.strip())
    return m.group(1).strip() if m else "Untitled Section"


def chunk_page(page_text: str) -> Iterator[Tuple[str, str]]:
    """
    Yields (heading, body) tuples for a single PDF page.
    Very simple: split on first blank line.
    """
    parts = page_text.strip().split("\n\n", 1)
    if len(parts) == 2:
        heading_raw, body = parts
    else:
        heading_raw, body = "", parts[0]

    heading = detect_heading(heading_raw.splitlines()[0] if heading_raw else "")
    yield heading, body.strip()
