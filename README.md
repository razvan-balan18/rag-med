# rag-med

Verifier-gated retrieval-augmented QA over open-access pneumology literature (COPD, asthma, ILD, PH). Hybrid retrieval (FAISS + BM25 + RRF + MedCPT cross-encoder rerank) feeds a Claude Sonnet generator; an NLI + Haiku judge verifies each sentence against retrieved evidence before streaming the answer.

## Install

```bash
uv venv
uv pip install -e .
cp .env.example .env  # fill in NCBI + Anthropic keys
```
