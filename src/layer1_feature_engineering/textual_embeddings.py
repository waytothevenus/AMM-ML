"""
Textual Embeddings – Feature Family 4/6  (RECOMMENDED ADDITION #5: swappable backend).

Produces dense vector embeddings from vulnerability text (descriptions,
advisories, exploit code snippets).

Supports two backends (configurable in settings.yaml → layer1.embedding_backend):
  - "tfidf"        : TF-IDF with SVD truncation – no GPU, no extra deps
  - "transformer"   : sentence-transformers model (loaded from local path)

Both backends expose the same interface:  embed(texts) → ndarray[N, D].
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from config import get_config

logger = logging.getLogger(__name__)


# ---- abstract interface ----

class EmbeddingBackend(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...


# ================================================================
#  TF-IDF + Truncated SVD backend
# ================================================================
class TFIDFBackend:
    """Lightweight, offline-safe text embedding."""

    def __init__(self, dim: int = 64, max_features: int = 5000) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        tfidf = self._vectorizer.fit_transform(corpus)
        n_features = tfidf.shape[1]
        if n_features < self._svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(
                n_components=max(1, n_features - 1), random_state=42
            )
        self._svd.fit(tfidf)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, dim) float32 array."""
        if not self._fitted:
            # Auto-fit on first call (cold start)
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf)
        # Pad to original dim if SVD was clamped
        if vecs.shape[1] < self.dim:
            pad = np.zeros((vecs.shape[0], self.dim - vecs.shape[1]), dtype=np.float32)
            vecs = np.hstack([vecs, pad])
        return vecs.astype(np.float32)


# ================================================================
#  Sentence-Transformer backend (optional heavy dependency)
# ================================================================
class TransformerBackend:
    """Uses a local sentence-transformers model for rich embeddings."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with:  pip install sentence-transformers torch"
            )

        cfg = get_config()
        from config import resolve_path
        path = model_path or str(resolve_path(cfg.models.embeddings_dir) / cfg.models.embedding_model_name)
        logger.info("Loading transformer model from %s", path)
        self._model = SentenceTransformer(str(path))
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


# ================================================================
#  Factory
# ================================================================
_BACKEND_CACHE: dict[str, EmbeddingBackend] = {}


def get_embedding_backend() -> EmbeddingBackend:
    cfg = get_config()
    name = cfg.layer1.embedding_backend  # "tfidf" | "transformer"
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]

    if name == "transformer":
        backend: EmbeddingBackend = TransformerBackend()
    else:
        dim = cfg.layer1.tfidf_svd_components
        max_features = cfg.layer1.tfidf_max_features
        backend = TFIDFBackend(dim=dim, max_features=max_features)

    _BACKEND_CACHE[name] = backend
    return backend


# ================================================================
#  High-level feature extraction
# ================================================================

def compute_textual_embeddings(
    vuln_id: str,
    graph,
    backend: EmbeddingBackend | None = None,
) -> dict[str, float]:
    """
    Return embedding features as {embed_0: ..., embed_1: ..., ..., embed_D-1: ...}.
    """
    if backend is None:
        backend = get_embedding_backend()

    node = graph.get_node(vuln_id) if hasattr(graph, "get_node") else {}
    if node is None:
        node = {}

    cwe_raw = node.get("cwe_ids", "")
    if isinstance(cwe_raw, list):
        cwe_text = " ".join(cwe_raw)
    elif isinstance(cwe_raw, str):
        cwe_text = " ".join(c.strip() for c in cwe_raw.split(";") if c.strip())
    else:
        cwe_text = ""
    texts_parts = [
        node.get("description", ""),
        node.get("title", ""),
        cwe_text,
    ]
    combined = " ".join(t for t in texts_parts if t).strip()
    if not combined:
        combined = vuln_id  # fallback to ID string

    vec = backend.embed([combined])[0]  # shape (D,)
    return {f"embed_{i}": float(v) for i, v in enumerate(vec)}
