"""MedCPT-Article-Encoder embedder (Q6).

Turns chunk text into dense vectors for the FAISS index.
``embed_chunks`` is text-in / ``(N, 768)`` float32-out — no FAISS here.

Two seams keep the torch path out of unit tests:
  - ``_load_model`` lazily loads the HF model on MPS (loud CPU fallback).
  - ``_forward`` runs one batch through the model + CLS pooling.
Tests monkeypatch both; the real code only runs in the manual smoke.
"""

from __future__ import annotations

import numpy as np
import structlog

log = structlog.get_logger()

MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
EMBED_DIM = 768
MAX_TOKENS = 512  # model limit; chunker keeps us under, defensive truncation

_model = None
_tokenizer = None


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    log.warning("mps_unavailable_cpu_fallback", model=MODEL_NAME)
    return "cpu"


def _load_model():
    """Lazily load (tokenizer, model) onto the best device. Cached."""
    global _model, _tokenizer
    if _model is None:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).to(_device()).eval()
    return _tokenizer, _model


def _forward(tokenizer, model, batch: list[str]) -> np.ndarray:
    """One batch → (len(batch), 768). CLS-token pooling per MedCPT usage."""
    import torch

    device = model.device
    enc = tokenizer(
        batch,
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        embeds = model(**enc).last_hidden_state[:, 0, :]
    return embeds.cpu().numpy()


def embed_chunks(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed chunk texts → float32 ``(N, 768)``."""
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    tokenizer, model = _load_model()
    blocks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        blocks.append(_forward(tokenizer, model, batch))
    return np.vstack(blocks).astype(np.float32)
