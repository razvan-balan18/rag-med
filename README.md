# rag-med

Verifier-gated retrieval-augmented QA over open-access pneumology literature (COPD, asthma, ILD, PH). Hybrid retrieval (FAISS + BM25 + RRF + MedCPT cross-encoder rerank) feeds a Claude Sonnet generator; an NLI + Haiku judge verifies each sentence against retrieved evidence before streaming the answer.

# Note from creator
Agentic AI has played a big role in the making of this project. For transparency, I decided to include the .claude folder, with all the settings and decisions. The files in the .claude/research have been created with the /grill-me skill. For more details check Mat Pocock's set of skills, which helped a lot during the development of this project. I tried to review the ai generated code to the best of my abilities, being a learning project. So, if necessary, kindly review. Cheers.

## Install

```bash
uv venv
uv pip install -e .
cp .env.example .env  # fill in NCBI + Anthropic keys
```
