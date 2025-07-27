"""
Lightweight wrapper around SentenceTransformers for CPU‑only embedding.

Usage
-----
embedder = Embedder()                     # loads model once
vec = embedder.embed("some text")         # list[float], length 384
vecs = embedder.embed(["t1", "t2"], batch=True)  # list[list[float]]
"""

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Tiny wrapper so the rest of the code stays framework‑agnostic."""

    def __init__(self) -> None:
        # download once; HF_HOME is persisted inside Docker layer
        self._model = SentenceTransformer(_MODEL_NAME, device="cpu")

    # ------------------------------------------------------------------ #
    def embed(self, text: Union[str, List[str]], batch: bool = False):
        """
        Parameters
        ----------
        text : str or list[str]
        batch : bool
            * If False (default) → `text` must be str; returns list[float].
            * If True  → `text` must be list[str]; returns list[list[float]].

        Returns
        -------
        list[float] or list[list[float]]
        """
        if batch:
            if not isinstance(text, list):
                raise TypeError("batch=True requires a list[str]")
            emb = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return emb.tolist()

        # single sentence
        if not isinstance(text, str):
            raise TypeError("batch=False requires text: str")
        emb = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return emb.tolist()
