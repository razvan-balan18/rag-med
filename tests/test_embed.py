"""MedCPT-Article-Encoder embedder tests — mock model load + forward.

No real model, no torch needed. The `_load_model` / `_forward` seams are the
explicit mock points (week2 day2: "mock the model load + forward").
"""

from __future__ import annotations

import numpy as np

from rag_med.indexing import embed as embed_mod
from rag_med.indexing.embed import EMBED_DIM, embed_chunks


def _fake_forward_factory(calls: list[list[str]]):
    """Record each batch; return a deterministic (len, 768) float64 block."""

    def _forward(tokenizer, model, batch):
        calls.append(list(batch))
        return np.ones((len(batch), EMBED_DIM), dtype=np.float64)

    return _forward


def test_returns_n_by_768(monkeypatch):
    monkeypatch.setattr(embed_mod, "_load_model", lambda: ("tok", "model"))
    monkeypatch.setattr(embed_mod, "_forward", _fake_forward_factory([]))

    out = embed_chunks(["a", "b", "c"])
    assert out.shape == (3, 768)


def test_dtype_is_float32(monkeypatch):
    monkeypatch.setattr(embed_mod, "_load_model", lambda: ("tok", "model"))
    monkeypatch.setattr(embed_mod, "_forward", _fake_forward_factory([]))

    out = embed_chunks(["a"])
    assert out.dtype == np.float32


def test_empty_input_returns_empty_no_model_load(monkeypatch):
    def _boom():
        raise AssertionError("_load_model must not run on empty input")

    monkeypatch.setattr(embed_mod, "_load_model", _boom)

    out = embed_chunks([])
    assert out.shape == (0, 768)
    assert out.dtype == np.float32


def test_batches_at_batch_size(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(embed_mod, "_load_model", lambda: ("tok", "model"))
    monkeypatch.setattr(embed_mod, "_forward", _fake_forward_factory(calls))

    texts = [str(i) for i in range(5)]
    out = embed_chunks(texts, batch_size=2)

    assert out.shape == (5, 768)
    assert [len(b) for b in calls] == [2, 2, 1]
