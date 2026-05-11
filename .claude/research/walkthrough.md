# Pneumology RAG — End-to-End Walkthrough

Companion to `architecture.md` (how it runs), `decisions.md` (what we picked), and `glossary.md` (vocabulary).

This document is a **plain-English narrative**. It walks one question from the moment brother types it in the browser to the moment he reads a verified, color-coded answer. Every technical term is explained the first time it shows up.

If something in here is still confusing, that's a bug in this doc — flag it and we'll add another analogy.

Last updated: 2026-05-10

---

## How to read this document

- **First time through:** read §0 (concepts primer) carefully. Every later section assumes you know those terms.
- **Second time through:** skim §0, focus on §3 (the request lifecycle) and §13 (the worked example).
- **Reference:** sections are self-contained. Jump straight to what you need.

---

## Table of contents

0. [Concepts you need first (the primer)](#0-concepts-you-need-first-the-primer)
1. [Three phases, three programs](#1-three-phases-three-programs)
2. [What brother sees: the website](#2-what-brother-sees-the-website)
3. [The request lifecycle, step by step](#3-the-request-lifecycle-step-by-step)
4. [The verification mechanism in depth](#4-the-verification-mechanism-in-depth)
5. [What renders in the browser, and when](#5-what-renders-in-the-browser-and-when)
6. [Clicking a citation](#6-clicking-a-citation)
7. [What happens when things go wrong](#7-what-happens-when-things-go-wrong)
8. [Phase 1: how the bundle was built](#8-phase-1-how-the-bundle-was-built)
9. [Phase 3: how quality is measured](#9-phase-3-how-quality-is-measured)
10. [The data — what lives where](#10-the-data--what-lives-where)
11. [What runs locally vs. what leaves the machine](#11-what-runs-locally-vs-what-leaves-the-machine)
12. [Latency and cost budget per query](#12-latency-and-cost-budget-per-query)
13. [A worked example end-to-end](#13-a-worked-example-end-to-end)
14. [Glossary of the moving parts](#14-glossary-of-the-moving-parts)

---

## 0. Concepts you need first (the primer)

Read this section before anything else. Each idea is one short paragraph plus an analogy.

### 0.1 What is RAG?

**RAG = Retrieval-Augmented Generation.** A two-step recipe:

1. **Retrieval:** when the user asks a question, search a private collection of documents for the most relevant passages.
2. **Generation:** paste those passages into a prompt and ask an LLM (Claude) to answer the question *using only those passages*.

Why bother? Because LLMs alone "hallucinate" — they invent plausible-sounding but wrong facts. By forcing the LLM to ground every claim in real retrieved text, we cut hallucinations dramatically. RAG is how you turn a general chatbot into a specialist that cites sources.

**Analogy:** A research assistant who can only quote from books on the desk in front of them, not from memory.

### 0.2 What is an embedding?

An **embedding** is a list of numbers — typically 384, 768, or 1024 of them — that represents the *meaning* of a piece of text. Two pieces of text with similar meaning produce similar lists.

Example: the sentences "the cat sat on the mat" and "a feline rested on the rug" have very different words but nearly identical embeddings, because their meanings are close.

How? A neural network is trained on huge amounts of text to map sentences into this number-space such that semantically related things land near each other.

We use 768-dim embeddings from a model called **MedCPT** — a model that was trained on PubMed click logs (which papers people clicked together for the same query), so it specifically understands medical-paper similarity.

**Analogy:** Each sentence gets coordinates on a giant 768-dimensional map. "Asthma symptoms" lands near "wheezing", far from "concrete pouring". Then "find similar sentences" becomes "find nearby points on the map".

### 0.3 What is a vector index? (FAISS)

A vector index is a database optimized for one question: *given this embedding, find the K embeddings closest to it.*

Doing this naively on 250,000 chunks would mean comparing the user's question against every chunk every time — slow. **FAISS** (Facebook AI Similarity Search) is a library that builds a clever data structure so the search runs in milliseconds.

We give FAISS one job: store all 250k chunk embeddings, return the 50 nearest when given a question embedding.

**Analogy:** A library where books are shelved by topic similarity, not alphabetically. The librarian can fetch the 50 most-related books in a second.

### 0.4 What is BM25?

**BM25** is a 30-year-old keyword-matching algorithm. It scores documents by how well their *words* match the query's words — accounting for word frequency, document length, and rare-word weight.

It does *not* understand meaning. "Asthma" and "wheezing" are unrelated to BM25. But it's brilliant at exact-token matches: rare medical terms, drug names like `MEPOLIZUMAB`, dose strings like `100mg`, lab values like `FEV1`.

We run BM25 *alongside* the embedding-based search because they have complementary failure modes — embeddings miss rare exact tokens, BM25 misses synonyms.

**Analogy:** A search engine from 2005. Doesn't understand what you mean, but if you type the exact right keyword it finds it instantly.

### 0.5 What is hybrid retrieval, and what is RRF?

**Hybrid retrieval** = use both an embedding-based search (FAISS) *and* a keyword search (BM25), then combine their results.

The challenge: FAISS gives scores like `0.87` (cosine similarity), BM25 gives scores like `12.4` (a totally different scale). You can't just add them.

**RRF = Reciprocal Rank Fusion.** Instead of using the raw scores, use the *rank* (1st, 2nd, 3rd...) each document got in each list. The combined score is:

```
final_score(doc) = sum over each list of:  1 / (60 + rank_in_that_list)
```

A document ranked 1st in both lists scores high. A document only in one list still gets credit. Score scales don't matter — only ranks do. The constant 60 is from the original RRF paper; we don't tune it.

**Analogy:** Two judges with different scoring scales. Instead of trying to normalize their scores, just see who they each ranked in the top 10, and weight by ranking position.

### 0.6 What is a cross-encoder, and how is it different from an embedding model?

Two types of model for "is this question related to this chunk?":

**Bi-encoder (what MedCPT-Query and MedCPT-Article do, what FAISS uses):**
- Encode question → vector A.
- Encode chunk → vector B.
- Compare A and B with a fast math operation (dot product).
- *The model never sees question and chunk together.* Fast — you can pre-encode all chunks once.

**Cross-encoder (what MedCPT-Cross-Encoder does):**
- Glue question and chunk into one input: `"[question] [SEP] [chunk text]"`.
- Run the whole thing through a model.
- Output a single relevance score.
- Far more accurate, *because the model attends to question and chunk together*.
- Far slower — you can't pre-encode anything; every (question, chunk) pair is a fresh forward pass.

We use both. Cheap bi-encoder + BM25 finds 50 candidates fast. Expensive cross-encoder re-scores those 50 to pick the top 10. This is called **two-stage retrieval**.

**Analogy:** First a junior librarian skims titles to grab 50 books. Then a senior librarian actually reads the relevant chapter of each to pick the best 10.

### 0.7 What is an LLM, and what is Claude?

**LLM = Large Language Model.** A neural network trained on internet-scale text to predict the next word. You give it a prompt; it generates a continuation.

**Claude** is Anthropic's family of LLMs. We use two:
- **Claude Sonnet 4.6** — strong, expensive. We use it as the **generator** that writes the answer.
- **Claude Haiku 4.5** — smaller, ~3x cheaper, ~3x faster. We use it as the **judge** that double-checks borderline claims.

The LLM doesn't run on brother's machine. We make HTTP calls to Anthropic's API and pay per token.

**Analogy:** Two cloud-hosted writers we rent by the word. Sonnet is the senior writer; Haiku is a fact-checker.

### 0.8 What is NLI?

**NLI = Natural Language Inference.** A model that takes two short texts and decides their logical relationship:

- **Premise:** "Patients on triple therapy showed FEV1 decline of 33 mL/year."
- **Hypothesis:** "Triple therapy slows lung function loss in COPD."

The NLI model outputs three probabilities that sum to 1:
- **entailment** — premise supports hypothesis
- **neutral** — premise neither supports nor contradicts
- **contradiction** — premise refutes hypothesis

We use `microsoft/deberta-v3-large-mnli` — a model trained on hundreds of thousands of human-labeled (premise, hypothesis, label) triples.

It's our cheap, fast, deterministic second opinion. It runs on CPU, no API calls, no per-query cost. It tells us, sentence-by-sentence, whether Claude's claim is actually backed by the chunk Claude cited.

**Analogy:** A junior fact-checker that's fast and free but not always confident. When confident, we trust them. When unsure, we escalate to the senior fact-checker (Haiku judge).

### 0.9 What is SSE? (live streaming from server to browser)

**SSE = Server-Sent Events.** A web standard for one-way real-time messaging. The browser opens a connection to the server. The server keeps it open and pushes messages whenever it wants. The browser receives each message as a discrete event.

Each SSE message looks like:
```
event: token
data: {"text": "Patients with COPD..."}
```

We use SSE to stream Claude's answer word-by-word as it's generated, so brother sees text appear immediately instead of waiting 8 seconds for the full answer.

**Analogy:** A radio. The browser tunes in; the server broadcasts.

### 0.10 What does "async" / "parallel" mean in our code?

Two ideas, often confused:

**Async (asyncio):** Python doesn't block while waiting for slow things. While we wait for Claude's API to send the next token, the same thread can handle a different task. Useful for I/O-heavy work (network calls).

**Parallel (multiple threads):** Two CPU-bound tasks run on two CPU cores at the same time. We use this for FAISS search + BM25 search — they're independent, so why do them sequentially?

Both show up in our code:
- `await asyncio.gather(...)` — run multiple async tasks concurrently.
- `asyncio.to_thread(some_function)` — run a CPU-bound function on a separate thread without blocking the event loop.

**Analogy:** Async = a cook stirring one pot while waiting for water to boil. Parallel = two cooks each at their own stove.

### 0.11 What is FastAPI?

**FastAPI** is a Python web framework. It lets us write code like:

```python
@router.post("/ask")
async def ask(req: AskRequest):
    return {"answer": "..."}
```

…and FastAPI handles all the HTTP plumbing: parsing the request, validating it, serializing the response, generating Swagger docs at `/docs`. We picked it because it's async-native (good for SSE streaming), has built-in validation via Pydantic, and is the de-facto standard for Python ML APIs.

**Analogy:** A receptionist that takes web requests, hands them to your function, and ships your function's return value back as JSON.

### 0.12 What is Pydantic?

**Pydantic** is a Python library for typed data validation. You declare a class with fields and types; Pydantic checks incoming data matches the schema.

```python
class AskRequest(BaseModel):
    question: str
    top_k: int = 10
```

If the request body has `question` as a number or `top_k` as a string, Pydantic raises a clear validation error → FastAPI turns it into a 400 response automatically. No manual `if isinstance(...)` checks scattered everywhere.

**We also use it for config**: `pydantic-settings` reads `.env` and `config.yaml`, validates them, exposes a typed `settings` object.

### 0.13 What is Docker / a container?

**Docker** packages an application + all its dependencies + a stripped-down Linux into a single file (an "image"). You run that image as a "container" — an isolated process tree.

Why? Brother gets one command (`docker compose up -d`) instead of: install Python 3.12, install 47 pip packages, install FAISS native libraries, set up paths, ... Container = "it works on my machine" actually works on his machine too.

**Analogy:** A shipping container. Same box ships from any factory to any port; doesn't matter what's inside.

### 0.14 What does "loaded in RAM" mean for the models?

Each model (MedCPT, DeBERTa, etc.) is a file of numerical weights — hundreds of megabytes to a couple gigabytes. When the FastAPI server boots, it reads those files and holds the weights in RAM (memory) for the lifetime of the process.

This means:
- **Boot is slow** (~12–15 seconds while everything loads).
- **Per-query is fast** — no disk reads, no cold start.
- **RAM cost is fixed** — ~3–4 GB consumed forever, but brother has 32 GB so we don't care.

We never lazy-load (load on first use) because the first user would pay a 15-second penalty.

**Analogy:** A chef puts all the ingredients on the counter at the start of service, not when each order arrives.

### 0.15 What is a chunk, and why do we chunk?

A research paper is too long to feed an LLM directly (Sonnet has limits on input size, costs scale with input length). So we break each paper into smaller pieces called **chunks** — about 350 tokens each, or roughly one paragraph.

Each chunk:
- Comes from one section of one paper (Introduction / Methods / Results / Discussion).
- Gets its own embedding.
- Has a unique ID like `12345678_methods_03` (paper 12345678, methods section, third chunk).
- Is what gets retrieved, ranked, cited, and verified.

The unit of search is the chunk. The unit of "where did this come from" is the paper.

### 0.16 What is a "token"?

Three different "tokens" show up in this project. Don't conflate them.

- **LLM tokens** — Claude breaks text into ~3-4-character pieces. Costs and context limits are measured in these. ~750 words ≈ 1000 tokens.
- **Embedding tokens** — each model (MedCPT, DeBERTa) has its own tokenizer that breaks text differently. We measure chunks in **DeBERTa tokens** because the verifier (DeBERTa-NLI) has a 512-token cap and is the strictest.
- **BM25 tokens** — words after our biomedical regex tokenizer (keep `IL-4` whole, drop stopwords). These are the "words" the keyword search indexes.

When we say "350-token chunks", we always mean DeBERTa tokens. Glossary forbids the bare term `n_tokens` — always specify which.

### 0.17 What is "top-K"?

**Top-K** = the K highest-scoring results from some ranking. K is a number we choose.

- "FAISS returns top-50" means the 50 nearest chunks by embedding similarity.
- "Cross-encoder reranks to top-10" means out of those 50, the 10 best after smarter scoring.
- "top-K=10 chunks go into the prompt" means Claude sees 10 chunks.

K trades quality vs. cost. More chunks in the prompt = more context for Claude = more $ + more chance of hallucination from noisy chunks. K=10 is our balance.

### 0.18 What is the "verifier"?

This is the project's signature feature. After Claude writes the answer, the verifier:

1. Splits the answer into sentences.
2. For each sentence, parses out the `[n]` citations.
3. For each (sentence, cited chunk) pair, runs NLI (the fast cheap fact-checker).
4. For borderline NLI results (model unsure), escalates to Haiku (the smart cheap fact-checker).
5. Labels each sentence: **green** (supported), **yellow** (unclear), **red** (unsupported), **gray** (verifier crashed).

This is what makes the project different from "ChatGPT over PDFs" — every claim gets a second opinion before brother trusts it.

---

## 1. Three phases, three programs

The codebase has three entry points. They share libraries but run at different times, in different places, for different reasons.

| Phase | Program | When it runs | Where |
|---|---|---|---|
| **Indexing (Phase 1)** | `python -m rag_med.indexing.pipeline` | Once, plus refreshes | Dev machine |
| **Serving (Phase 2)** | FastAPI process inside Docker | Always-on | Brother's machine |
| **Eval (Phase 3)** | `python -m eval` | On demand during dev | Dev machine |

**Why three?** Phase 1 *builds* the search index from scratch (slow, ~5 hours). Phase 2 *uses* the prebuilt index to answer questions (fast). Phase 3 *measures* how good Phase 2 is (developer tool).

This walkthrough centers on **Phase 2** — that's what brother actually uses. Phases 1 and 3 are covered in §8 and §9 because you need them to understand where the data comes from and how we know the system works.

The three phases share `src/rag_med/shared/` — tokenizer, schemas, DB helpers. Each has its own folder otherwise (`indexing/`, `serving/`, `eval/`).

---

## 2. What brother sees: the website

The frontend is intentionally minimal. Plain HTML + vanilla JavaScript + ES modules + SSE. **No React, no build step.** Files live in `static/` and FastAPI itself serves them.

**Why so plain?** Building a polished React UI would burn 1–2 weeks. The project's value is RAG quality + verification, not frontend craft. Plain HTML gets us to working in a day.

### 2.1 The page

One page, one URL: `http://localhost:8000/`.

```
┌────────────────────────────────────────────────────────────┐
│  Pneumology RAG                          [⚙] [docs]        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   ┌──────────────────────────────────────────────────┐     │
│   │ Ask a question…                                  │     │
│   └──────────────────────────────────────────────────┘     │
│                                          [ Ask ]           │
│                                                            │
│   ── Answer ───────────────────────────────────────────    │
│   (streamed text appears here as Claude writes)            │
│                                                            │
│   ── Retrieved chunks (collapsible) ───────────────────    │
│   [1] Calverley 2007, NEJM, Methods    snippet…            │
│   [2] Vestbo 2013, Lancet, Results     snippet…            │
│   …                                                        │
│                                                            │
│   ── Citation detail (slides in on click) ─────────────    │
│   (full chunk text + paper metadata)                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 2.2 The state model (how the page tracks what's going on)

`static/state.js` holds one JavaScript object that describes everything visible:

```js
let state = {
  question: '',
  chunks: [],            // top 10 retrieved
  answerSentences: [],   // [{text, citations:[1,3], status:'pending'|...}]
  streaming: false,
  error: null
};
```

Every UI change goes through `setState({...})`, which updates the object and re-runs the rendering. **Why?** This avoids the bug class where the DOM and your mental model disagree. If state says `streaming=true`, the spinner is visible. Always. No exceptions.

This is the React idea (declarative UI from state) reimplemented in 30 lines without any framework.

### 2.3 What happens on first page load

1. Brother opens `http://localhost:8000/` in his browser.
2. FastAPI serves `static/index.html`.
3. Browser parses HTML, sees `<script type="module" src="app.js">`, loads JS modules.
4. `app.js` calls `GET /health`. The backend is still loading models — responds with `503 {status:"loading"}`.
5. The input field stays grayed out. Frontend retries `/health` every 2 seconds.
6. Once the backend finishes loading models (~15 s after Docker started), `/health` returns `200 {status:"ok", paper_count, chunk_count, ...}`.
7. Input field unlocks. Brother types a question and clicks Ask.

There's no login, no session, no history. Every `/ask` is independent.

---

## 3. The request lifecycle, step by step

This is the heart of the system. Brother clicked Ask. Here's what happens, with the file/function each step lives in.

### 3.1 Step 0 — Browser opens an SSE stream

Frontend (`static/sse.js`) sends a POST request and tells the browser to read the response as a live stream of events:

```
POST /ask
Content-Type: application/json
Accept: text/event-stream
{"question": "What is the FEV1 decline rate in COPD on triple therapy?", "top_k": 10}
```

The browser holds this connection open until the server closes it. The server pushes events into the connection as the pipeline runs.

(Aside: browsers' built-in `EventSource` only does GET. We use `fetch()` + a stream reader because we want to POST a JSON body. Same effect.)

### 3.2 Step 1 — FastAPI accepts and validates

`src/rag_med/serving/api.py`:

```python
@router.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    return StreamingResponse(pipeline(req.question, req.top_k),
                             media_type="text/event-stream")
```

What this does:
- `AskRequest` is a Pydantic model. If `question` is missing or wrong type, Pydantic auto-rejects with `400 Bad Request`.
- `StreamingResponse` tells FastAPI: don't wait for `pipeline()` to finish, just send each event as it's produced.
- `pipeline()` is an `async def` function that `yield`s SSE-formatted strings — that's what makes the stream work.

### 3.3 Step 2 — Embed the question (the dense path)

`serving/retrieve.py::embed_question()`.

What this does in plain English: turn the question text into a list of 768 numbers that represents its meaning, so we can compare it to chunks.

How:
1. The MedCPT-Query-Encoder model is already in RAM (loaded at boot, see §0.14).
2. Tokenize the question with MedCPT's tokenizer.
3. Run the question through the model — one forward pass.
4. Output: `query_vector`, a 768-dimensional float32 array.

Because this is CPU-bound, we wrap the call in `asyncio.to_thread(...)` so it doesn't block the event loop (§0.10).

We also cache the result with `functools.lru_cache(maxsize=1000)` — if the same question gets asked twice, the embedding is reused. Tiny win but free.

### 3.4 Step 3 — Tokenize the question (the lexical path)

`shared/tokenize.py::biomedical_tokens()`.

What this does in plain English: break the question into "words" the way BM25 expects — but with custom rules so medical terms survive intact.

```python
"What is the FEV1 decline rate in COPD on triple therapy?"
↓
["fev1", "decline", "rate", "copd", "triple", "therapy"]
```

Rules (locked Q7):
- Keep hyphens, plus signs, slashes inside tokens: `IL-4`, `CD8+`, `FEV1/FVC` stay whole. (A naive tokenizer would split `IL-4` into `il` and `4`, killing recall on every cytokine query.)
- Keep digit-letter combos: `FEV1`, `25mg`.
- Drop English stopwords: "the", "is", "of", "and", "in".
- Lowercase.

Output: `bm25_tokens: list[str]`.

### 3.5 Step 4 — Search FAISS and BM25 in parallel

`serving/retrieve.py::retrieve()`:

```python
dense_hits, lexical_hits = await asyncio.gather(
    asyncio.to_thread(faiss_index.search, query_vector, k=50),
    asyncio.to_thread(bm25.get_top_n, bm25_tokens, n=50),
)
```

What this does in plain English: run both search engines at the same time, on different threads, and wait for both.

- **FAISS** returns the 50 chunks with embeddings closest to `query_vector`. Each result is `(chunk_id, similarity_score)`.
- **BM25** returns the 50 chunks with the strongest keyword overlap. Each result is `(chunk_id, bm25_score)`.

`asyncio.gather` waits for both to finish; total time = max(faiss_time, bm25_time) ≈ 0.3 seconds (not the sum).

**Why both?** They have complementary strengths. FAISS catches "dyspnea" matching "shortness of breath" (semantic). BM25 catches the exact word `MEPOLIZUMAB` even if the embedder hasn't seen it (lexical). Using only one would miss roughly 20% of relevant chunks.

### 3.6 Step 5 — Fuse the two ranked lists with RRF

`serving/retrieve.py::rrf_fuse()`.

We have two ranked lists. Each list has the same 50-ish chunks but in different orders, with different score scales. We want one merged list.

We use Reciprocal Rank Fusion (§0.5):

```
For each chunk that appears in either list:
  rrf_score(chunk) = sum:
    1 / (60 + rank_in_dense)    if it appears in dense list
    1 / (60 + rank_in_lexical)  if it appears in lexical list

Sort chunks by rrf_score, descending.
```

Output: `fused_candidates`, ~50 unique chunks ordered by combined relevance. The constant 60 comes from the original RRF paper; we don't tune it.

### 3.7 Step 6 — Cross-encoder rerank

`serving/rerank.py::rerank()`.

What this does in plain English: take the 50 candidates and run a smarter (slower) model on each to get a more accurate ranking. Pick the top 10.

Recall §0.6: a cross-encoder reads `(question, chunk)` jointly. It's more accurate than the bi-encoder embeddings used in Step 4, but too slow to run on the whole corpus. We use it as a refinement pass.

```python
pairs = [(question, chunk_text) for chunk_text in fused_candidates_texts]
rerank_scores = cross_encoder.predict(pairs)  # one batched forward pass
reranked_chunks = sorted(zip(fused_candidates, rerank_scores), key=score, reverse=True)
top_10 = reranked_chunks[:10]
```

Time: ~1 second on i7-155H for 50 pairs.

### 3.8 Step 7 — Apply the rerank floor (refusal trigger or per-chunk drop)

`serving/retrieve.py::apply_rerank_floor()`. Implements `decisions.md §Q9b`.

There's a config knob `rerank_floor` (a single number, hardcoded in v1, calibrated empirically in M4). It does two things:

**A. Hard refusal trigger.** If the *best* rerank score is below the floor, none of the chunks are good enough to answer — the question is out of scope.

In plain English: brother asked something the corpus doesn't cover. Don't even bother calling Claude. Send back a fixed "I can't answer this from the evidence" response.

```python
if reranked_chunks[0].rerank_score < rerank_floor:
    yield sse_event("token", {"text": "The retrieved evidence does not address this question."})
    yield sse_event("verified", {...one supported sentence...})
    yield sse_event("done", {})
    log_trace(refusal="hard")
    return
```

**B. Per-chunk drop.** Any chunks below the floor that survived top-K get dropped from `final_chunks` before prompting. Fewer noisy chunks = less hallucination opportunity.

Survivors → `final_chunks: list[chunk_id]`. In M1–M3, `rerank_floor = 0.0` (everything passes); calibrated empirically in M4 against the gold set.

### 3.9 Step 8 — Emit the `retrieved` SSE event

```
event: retrieved
data: {"chunks": [
  {"chunk_id": "12345678_methods_03", "paper_pmid": "12345678",
   "section_type": "methods", "text_preview": "Patients were randomized…"},
  …  // top 10
]}
```

Frontend stores this in `state.chunks` and renders the collapsible "Retrieved chunks" panel below where the answer will appear.

**Why send this before the answer?** Brother sees the evidence list before Claude's first word. Useful for trust ("oh good, it found the TORCH trial"), and useful for catching obvious mistakes ("wait, why is it pulling cardiology papers?").

### 3.10 Step 9 — Build the prompt for Claude

`serving/generate.py::build_prompt()`.

In plain English: paste the chunks and the question into a string template, with strict rules about how Claude should answer.

Each chunk gets a numbered header with metadata, then the chunk text:

```
[1] (Calverley et al. 2007, NEJM, Methods)
"Patients were randomized to salmeterol/fluticasone…"

[2] (Vestbo et al. 2013, Lancet, Results)
"Annual FEV1 decline was…"
…

Question: What is the FEV1 decline rate in COPD on triple therapy?
```

The system prompt enforces five rules (`decisions.md §Q12c`):
1. Only use information from the chunks. No background knowledge.
2. Every factual sentence must end with a citation marker like `[1]` or `[1][3]`.
3. For numbers (doses, p-values, FEV1, sample sizes), quote the chunk verbatim before citing.
4. No medical advice voice. "Study X found…" not "You should…".
5. If the chunks don't address the question, refuse with a fixed phrase.

This prompt is the *contract* Claude must obey. The verifier (Step 11+) exists because Claude sometimes breaks the contract — and we'd rather catch it than hide it.

### 3.11 Step 10 — Stream the answer from Claude Sonnet

`serving/generate.py::generate_stream()`.

```python
async with generator_client.messages.stream(
    model="claude-sonnet-4-6",
    system=SYSTEM_PROMPT,
    messages=[{"role":"user", "content": prompt}],
) as stream:
    async for delta in stream.text_stream:
        token = delta  # rename Anthropic's "delta" → our "token" at the boundary
        accumulated_text += token
        yield sse_event("token", {"text": token})
```

What this does in plain English:
1. Open a streaming connection to Anthropic's API.
2. As Claude generates each chunk of text, immediately forward it to the browser as an SSE `token` event.
3. Also accumulate the text locally — we'll need the full answer for the verifier.

Brother sees the answer typing itself in real time, starting ~0.5 s after the first event. He's reading along by t = 4 s.

When Claude finishes, the SDK reports total token cost. We stamp that into `query_traces.generator_cost_usd` (~$0.018 per query).

### 3.12 Step 11 — Split the answer into sentences

`serving/verify.py::split_sentences()`.

The full answer is one string. The verifier needs to check claim-by-claim. We split on `.!?` with a regex (v1; may upgrade to a smarter splitter later if `Fig. 1`-style edge cases cause noise).

Each sentence becomes a `Sentence` object:

```python
class Sentence:
    idx: int                          # 0, 1, 2, ...
    text: str                         # "Patients on triple therapy showed..."
    citations: list[int]              # [1] or [1, 3] etc., parsed from regex
    cited_chunk_ids: list[str]        # resolved via final_chunks
    status: 'supported'|'unclear'|'unsupported'|'unknown'
    failure_kind: str | None
    nli_entailments: list[float]
    judge_confidences: list[float]
```

`citations` come from a regex on `\[(\d+)\]`. If the sentence is `"...rate of 33 mL/year [1][2]."`, we get `citations = [1, 2]`.

`cited_chunk_ids` come from looking each `[n]` up against the `final_chunks` map (1-indexed → chunk_id):
- `[1] → final_chunks[0] → "12345_methods_03"`
- `[2] → final_chunks[1] → "67890_results_01"`

If a sentence has `[7]` but only chunks `[1]` through `[5]` were in the prompt, then `cited_chunk_ids` for that `[7]` is `''` and we tag the sentence with `failure_kind = 'fabricated_citation'` (Claude invented a citation number — see §4).

### 3.13 Step 12 — NLI batch pass (the cheap fact-check)

`serving/verify.py::nli_check()`.

The DeBERTa-NLI model (§0.8) is in RAM. We need to score every (sentence, cited_chunk) pair.

Here's the multi-citation rule (locked Q1, AND-of-singles): a sentence with citations `[1][3]` produces *two* pairs to check:
- `(sentence, chunk_1)`
- `(sentence, chunk_3)`

We collect all pairs across all sentences into one big batch and run a single forward pass through DeBERTa:

```python
inputs = [(sentence.text, chunks[cid].text)
          for s in sentences
          for cid in s.cited_chunk_ids]
nli_outputs = nli_model(inputs)  # batched
```

For each pair, the model outputs three probabilities: `[entailment, neutral, contradiction]`. We keep the entailment probability per pair.

Time: ~3–5 seconds for ~10 sentences × ~1.5 cited chunks each. CPU only, batched.

### 3.14 Step 13 — Triage and judge escalation (the smart fact-check, only when needed)

`serving/verify.py::triage_and_judge()`.

For each pair, decide based on the NLI entailment score:

- `entailment ≥ 0.9` → confident **supported**. Done.
- `contradiction ≥ 0.9` → confident **unsupported**. Done.
- Otherwise (entailment in the borderline 0.3–0.9 band, ~30% of pairs in practice) → **escalate to the judge**.

The judge is Claude Haiku 4.5. We use a separate API client (`judge_client`, distinct from `generator_client`). We send a one-shot prompt:

```
You evaluate whether a passage supports a single claim.

Passage: "<chunk text>"
Claim: "<sentence text>"

Reply ONLY with JSON:
{"verdict": "supported"|"unsupported"|"unclear", "confidence": 0.0-1.0}
```

All escalated calls are dispatched in parallel via `asyncio.gather()` — they don't depend on each other, so we don't wait for one before sending the next.

Cost: ~$0.003–0.005 per query in total judge spend. Stamped into `query_traces.judge_cost_usd`.

**Why not use Haiku for everything?** Because NLI is free and handles ~70% of pairs confidently. Pure-judge would cost ~3x more.

**Why not use NLI alone?** Because NLI is unreliable in the borderline band — the model knows it's unsure, and we'd be flipping coins on 30% of claims. Haiku resolves those.

### 3.15 Step 14 — Finalize the per-sentence verdict

`serving/verify.py::finalize()`.

Each sentence may have multiple cited chunks → multiple pairs → multiple verdicts. We collapse them with the AND-of-singles rule:

- All pairs `supported` → sentence is **supported** (green).
- Any pair `unsupported` → sentence is **unsupported** (red), `failure_kind = 'nli_contradiction'` or `'judge_inconclusive'` depending on which stage decided.
- Any pair `unclear`, none `unsupported` → sentence is **unclear** (yellow).

Edge cases (forensics):
- Sentence with `[7]` but only 5 chunks existed → status='unsupported', `failure_kind='fabricated_citation'`. NLI/judge skipped for that pair.
- Sentence with no `[n]` at all → status='unsupported', `failure_kind='no_citation'`. Strict by design — we want to see this in the dots and fix the prompt if it happens often.
- Vacuous-but-cited ("These results are interesting [3]") → NLI returns neutral → judge returns "unclear" → status='unclear' (yellow). Honest signal that the claim is too vague to verify.
- NLI/judge crash on a sentence → status='unknown' (gray), `failure_kind='verifier_crash'`. Pipeline continues for the other sentences.

### 3.16 Step 15 — Emit the `verified` SSE event

```
event: verified
data: {"sentences": [
  {"idx": 0, "text": "Patients with COPD on triple therapy…",
   "citations": [1], "cited_chunk_ids": ["12345_results_02"],
   "status": "supported", "failure_kind": null,
   "nli_entailments": [0.94], "judge_confidences": [null]},
  {"idx": 1, "text": "These findings are exciting [3].",
   "citations": [3], "cited_chunk_ids": ["67890_discussion_01"],
   "status": "unclear", "failure_kind": "judge_inconclusive",
   "nli_entailments": [0.45], "judge_confidences": [0.5]},
  …
]}
```

The frontend re-renders the answer block with colored dots inline. This single event arrives ~5–8 s after the last `token` event — that's the verifier latency, a known v1 limitation. (Post-v1 path: incremental verification per sentence.)

### 3.17 Step 16 — Emit `done` and close the stream

```
event: done
data: {}
```

Server closes the SSE connection. Frontend flips `state.streaming = false`. Server writes one row to `query_traces` (forensics) and one structured-JSON log line to stdout.

Total time from Ask click to done event: typically 10–15 seconds, p95 budget ≤20 seconds.

---

## 4. The verification mechanism in depth

Verification is the project's signature feature. It deserves its own section.

### 4.1 Why verify at all?

Even Sonnet sometimes:
- Slaps `[3]` on a sentence that only chunk 1 supports (over-citation).
- Writes a sentence with no citation when it ran out of grounded content (citation drift).
- Fabricates a citation number — `[7]` when only 5 chunks exist.
- Paraphrases a number wrong: "FEV1 declined 40 mL/year [1]" when chunk 1 actually says 33 mL/year.

Without verification, brother would have to open every cited paper to check — defeating the tool. With verification, the system surfaces failures honestly via colored dots. The original answer stays untouched; the dots tell brother where to be skeptical.

### 4.2 Why two stages (NLI then judge)?

Pure-LLM-judge would cost ~$0.05/query × 30 q/day = ~$45/month. Pure-NLI would let many borderline cases through with random verdicts.

Hybrid:
- **NLI is fast** (~5 s for 10 sentences batched on CPU), free, deterministic. Handles the obvious 70% of pairs (clear entailment or contradiction).
- **Judge handles the hard 30%** at ~$0.005/query average. Net cost ~70% lower than LLM-only.

### 4.3 Why DeBERTa specifically?

`microsoft/deberta-v3-large-mnli` is one of the strongest publicly available NLI models. Trained on the MNLI dataset (huge collection of human-labeled premise/hypothesis pairs). 1.5 GB; runs at acceptable speed on CPU.

The 512-token cap is *the* reason we measure chunks in DeBERTa tokens (§0.16). If `(chunk + sentence)` exceeded 512 tokens, DeBERTa would silently truncate, making it fact-check against a chunk *fragment* — potentially wrong verdict. Sizing chunks at 350 ± 50 DeBERTa tokens leaves comfortable headroom: `(350-token chunk + ~30-token sentence) = ~380 tokens`, well under 512.

The `truncation_assert_failed` `failure_kind` exists as a safety net to fire if this invariant ever breaks.

### 4.4 Why Haiku for the judge, not Sonnet?

The judge task is *binary entailment over short text* — much simpler than open-ended generation. Haiku 4.5 is plenty for this. Sonnet would cost ~3x with no quality lift. This is the Q16 cost decision.

### 4.5 What the verifier deliberately does NOT do

- It does **not** edit Claude's answer. The original streamed text stays. Dots get added next to it.
- It does **not** re-prompt Claude with "fix this". That hides failures.
- It does **not** strike through unsupported sentences. Brother reads them and sees the red dot.
- It does **not** chain (e.g., "if sentence 1 is unsupported, also flag sentence 2"). Each sentence is verified in isolation.

The whole posture: **surface failures, don't paper over them.**

---

## 5. What renders in the browser, and when

The frontend's job is to translate SSE events into DOM updates.

### 5.1 Event handlers

`static/sse.js` parses each event and dispatches to handlers in `app.js`:

| SSE event | Handler | What it does |
|---|---|---|
| `retrieved` | `onRetrieved(chunks)` | `setState({chunks})`. Renders the "Retrieved chunks" panel. |
| `token` | `onToken(text)` | Appends text to the answer area. No state mutation; pure DOM op. |
| `verified` | `onVerified(sentences)` | `setState({answerSentences})`. Re-renders the answer with sentence dots. |
| `done` | `onDone()` | `setState({streaming: false})`. Stops "thinking" indicator. |
| `error` | `onError(payload)` | `setState({error})`. Shows red banner + retry button. |

### 5.2 How the streaming text renders without flicker

During `token` events, we directly mutate a `Text` node — *not* `innerHTML`:

```js
const answerEl = document.getElementById('answer-text');
const textNode = answerEl.firstChild;  // a Text node
function onToken(text) {
  textNode.appendData(text);  // O(1) append, no HTML re-parse
}
```

If we used `innerHTML += text`, the browser would re-parse the entire answer string on every token — flicker, sluggishness, ruined UX. `appendData` on a Text node is fast.

On `verified`, we accept one full re-render of the answer block (split into sentence spans, attach colored dots, attach `[n]` click handlers). One re-render is fine.

### 5.3 The colored dots

Each sentence is rendered as a span with a colored dot at the end:

```html
<span class="sentence sentence--supported" data-sentence-idx="0">
  Patients with COPD on triple therapy showed reduced exacerbation rates [1].
  <span class="dot dot--green" title="Verified by NLI 0.94"></span>
</span>
```

Color mapping (from glossary):
- `supported` → green
- `unclear` → yellow
- `unsupported` → red
- `unknown` → gray (verifier crashed)

Hover tooltip shows the verifier's evidence: NLI score, judge verdict + confidence, `failure_kind` if any.

### 5.4 Footer summary

Below the answer, a one-line summary:

> 8/9 claims verified · 1 unclear · 0 unsupported

Quick-glance trust signal. If the right-side numbers aren't 0, brother slows down and reads more carefully.

---

## 6. Clicking a citation

Brother sees `[3]` in the answer. He clicks it.

1. Click handler reads `state.answerSentences[i].cited_chunk_ids[3]` → `"67890_results_02"`.
2. Frontend calls `GET /chunks/67890_results_02`.
3. Backend reads from SQLite `chunks` table:
   ```json
   {"chunk_id": "67890_results_02", "paper_pmid": "67890",
    "section_type": "results", "text": "<full chunk text>",
    "prev_chunk_id": "67890_results_01", "next_chunk_id": "67890_results_03"}
   ```
4. Frontend calls `GET /papers/67890`.
5. Backend reads from SQLite `papers` table:
   ```json
   {"pmid": "67890", "title": "Long-term FEV1 decline in COPD",
    "authors": "Vestbo et al.", "journal": "Lancet", "year": 2013,
    "abstract": "…", "doi": "10.1016/…", "source_type": "full_text"}
   ```
6. Frontend renders the citation panel: paper title, authors, journal/year, section type, full chunk text, link to PubMed (`https://pubmed.ncbi.nlm.nih.gov/67890`).
7. Brother reads the actual evidence and decides whether the answer's claim holds up.

`prev_chunk_id` / `next_chunk_id` exist so a "show more context" button can fetch the surrounding chunks.

These two routes (`/chunks`, `/papers`) are why the API has four endpoints, not one. Without them, brother would have to manually search PubMed for every reference.

---

## 7. What happens when things go wrong

Every failure mode is catalogued in `architecture.md §5.1`. Here's the brother-facing experience for each.

### 7.1 No chunks pass the rerank floor

Brother asks "What's the capital of France?". Cross-encoder scores everything below `rerank_floor`.

- `event: retrieved` may still fire (low-quality chunks; useful for debugging).
- Server skips Claude entirely.
- `event: token` fires with the canonical phrase: `"The retrieved evidence does not address this question."`.
- `event: verified` with one supported sentence (the refusal is trivially true).
- `event: done`.

Logged: `refusal: "hard"`. No Claude API cost. This is the behavior the Q15 acceptance gate measures against the 10-question adversarial slice.

### 7.2 Claude API rate-limited or down

Anthropic's SDK retries 3x with exponential backoff automatically. If still failing:

- `event: error  data: {stage: "generate", code: "anthropic_5xx", message: "...", retryable: true}`.
- `event: done`.
- Frontend shows red banner: "Claude is having trouble. Try again."
- Retry button re-issues the same `/ask`.

### 7.3 Monthly cap reached

Brother queries past the $15/month default cap (`config.yaml::cost.monthly_cap_usd`).

- `event: error  data: {code: "monthly_cap", message: "Monthly cap reached. Raise in config.yaml.", retryable: false}`.
- Brother edits `config.yaml`, restarts container.

This is preferable to silent 429 errors from Anthropic — the failure is predictable and the fix is one config edit (Q16).

**Soft signal before this hard fail (Q21).** When MTD spend > 0.8 × cap, the system logs loud + sets `/health.cost_warning=true`, frontend shows yellow banner. Brother decides whether to raise the cap before getting blocked rather than discovering at the moment of need.

### 7.3b Per-query ceiling exceeded (Q21)

A single `/ask` somehow accrues more than `cost.per_query_ceiling_usd` ($0.10 default) before completing — almost always a code bug (e.g. judge stuck in a tight loop, runaway tool use).

- Pipeline aborts mid-query.
- `event: error  data: {code: "per_query_ceiling", message: "single-query cost exceeded $0.10 ceiling — likely a bug, check logs", retryable: false}`.
- `event: done`.
- Logged with full per-call cost breakdown for forensics.

**Why no retry:** retrying a buggy code path just spends more money. This is a code-bug signal, not a transient.

### 7.3c Anthropic console hard limit hit (Q21)

Console-side cap on the API key (dashboard, not `config.yaml`) is reached. Defaults: $50/mo dev key, $30/mo brother's key. Anthropic returns a quota error.

- Same UX as 429 path: `event: error  data: {code: "console_quota", ...}`.
- Distinguishes from 429 in logs via response body — useful when triaging "did we burn through the budget or is Anthropic just throttling?"
- Backstop layer — defends against any code bug that bypasses `monthly_cap_usd`.

### 7.4 Verifier crash mid-pipeline

NLI model raises an error (e.g., out-of-memory). The generator already finished; brother already saw the answer.

- `event: verified` still fires, but each affected sentence has `status='unknown'`, `failure_kind='verifier_crash'`.
- All dots render gray.
- Footer: "Verification unavailable for this answer."
- Brother knows to treat the answer with extra skepticism.

The answer is **not** retracted. The system is honest about the failure.

### 7.5 Browser closes mid-stream

User closes the tab while Claude is still streaming.

- Server-side async generator detects the closed connection on its next yield.
- Pipeline cancelled (Claude stream abandoned, NLI never runs).
- Logged with `aborted: true`.
- Brother only sees what made it to the browser before he closed.

(Anthropic still bills for tokens that did stream — outside our control.)

### 7.6 Bundle out of date

Brother's local bundle is `v1.0.0`, dev pushed `v1.1.0` to HuggingFace.

- On boot, app fetches `manifest.json` from the public HF URL, compares versions.
- `/health` returns `{... update_available: true, latest_version: "v1.1.0"}`.
- Frontend banner: "A newer index is available. Run ./scripts/download_index.sh."
- **No auto-update** — too disruptive (10 GB download). Brother runs the script when convenient.

---

## 8. Phase 1: how the bundle was built

This runs once on the dev machine before brother ever sees the system. Output: a tarball brother downloads.

### 8.1 The five staged commands

```
python -m rag_med.indexing.pipeline fetch     # PubMed + PMC → SQLite
python -m rag_med.indexing.pipeline chunk     # split papers → SQLite chunks
python -m rag_med.indexing.pipeline embed     # MedCPT → FAISS file
python -m rag_med.indexing.pipeline bm25      # tokenize → BM25 index file
python -m rag_med.indexing.pipeline manifest  # bundle manifest + tar.gz
```

Each writes to disk before the next runs. Re-running a stage is free; re-running everything from scratch is ~4–6 hours.

### 8.2 Fetch

In plain English: download every paper that matches the filter, parse the XML, store in SQLite.

- Query PubMed E-utilities `esearch` for the filter (Q4 lock: MeSH Respiratory Tract Diseases tree + journal whitelist + date ≥ 2015) → list of PMIDs.
- For each PMID, `efetch` → XML → parse → row in `papers` table.
- For PMIDs in the PMC Open Access subset, fetch full XML from PMC → store full text alongside abstract.
- Network errors: 3-attempt exponential backoff. Parse errors: log to `failed_papers`, no retry, continue. Pipeline never crashes on one bad paper.
- Resumability: SQLite is the progress log. Re-run skips already-fetched PMIDs (`INSERT OR IGNORE`).

**Day-1 mechanics (Q22b–c).**
- HTTP via `httpx` directly. Not Biopython — too heavyweight for one module, hides headers we want to see.
- NCBI API key required day 1 (free, ~5 min signup). `.env` carries `NCBI_API_KEY` + `NCBI_EMAIL`. With key: 10 req/s. Without: 3 req/s. NCBI requires `NCBI_EMAIL` on every E-utilities request as a politeness contract.
- XML parsing via `pubmed_parser` library — returns paragraphs tagged by `section_name`, exact shape the IMRaD chunker wants. ~50 lines of integration vs ~300 of hand-rolled XPath. M5 escape hatch: swap `indexing/ingest/parse.py` for hand-rolled `lxml` if quality assessment forces it.
- M1 toy uses E-utilities exclusively (no PMC OA Bulk FTP / OAI-PMH); 100 papers fits well within rate limit. M5 full corpus (~150k) introduces the bulk path.

**Salvage rule for malformed XML (Q22d) — minimum viable record.** Keep paper iff `pmid` present + `title` present + (`abstract` present OR ≥1 body section parsed). Per-chunk forgiveness: if one section/table fails, drop that chunk only. Failed papers row in `failed_papers` with `failure_reason` enum (`missing_title` | `no_content` | `xml_parse_error` | `encoding_error`).

**M1 toy corpus shape (Q22a).** Narrow topic, full-text only — NOT random sample from the full filter. Topic = COPD (highest density of post-2020 PMC OA full-text in pneumology + landmark trials make toy questions write themselves). Query: `("Pulmonary Disease, Chronic Obstructive"[MeSH] OR "COPD"[Title/Abstract]) AND ("2020"[Date - Publication] : "3000"[Date - Publication]) AND "open access"[filter]`. The `open access` filter guarantees every result has a PMCID, so the full-text path is always exercised in M1.

### 8.3 Chunk

In plain English: cut each paper into ~350-DeBERTa-token pieces, tagged with which section they came from.

`indexing/chunk.py`:
1. Parse the XML body into IMRaD sections (Introduction / Methods / Results / Discussion).
2. Inside each section, split at sentence boundaries.
3. Greedy-pack sentences into chunks targeting 350 ± 50 DeBERTa tokens.
4. Tag each chunk with `section_type`.
5. Tables → own chunks (caption + cell text). Figure captions → own chunks. References list stripped entirely.
6. Abstract → own chunk with `section_type='abstract'`.
7. Write to `chunks` table: `chunk_id = "{pmid}_{section_type}_{ordinal:02d}"`, plus `n_deberta_tokens` and `n_medcpt_tokens` for sanity.

Output: ~200–250k chunks across ~150k papers (numbers re-measured during M5).

### 8.4 Embed

In plain English: turn each chunk's text into a 768-dim vector, save them all to a FAISS index file.

- Load `MedCPT-Article-Encoder` (the article-side counterpart of the query encoder).
- Forward pass over each chunk text → 768-dim vector.
- Batched, ~hours on CPU for 250k chunks.
- Write all vectors into a FAISS index. v1 default: `IndexFlatIP` for accuracy. (Q18 deferred: maybe HNSW for speed if needed.)
- Save to `faiss.index` (~750 MB).

### 8.5 BM25 index build

- For each chunk, tokenize with the biomedical regex tokenizer.
- Build a `rank_bm25` inverted index over the tokens.
- Pickle to `bm25.pkl` (~200 MB).

(An "inverted index" is just a dictionary: word → list of chunks containing that word. That's how keyword search runs in milliseconds instead of scanning every chunk.)

### 8.6 Manifest + bundle

- Insert one row in `index_manifest` SQLite table: `bundle_version`, `built_at`, `embed_model`, `embed_model_revision`, `chunker_git_sha`, `paper_count`, `chunk_count`.
- Tar `(sqlite.db, faiss.index, bm25.pkl, manifest.json)` → `bundle.tar.gz`.
- Upload to HuggingFace Datasets via `huggingface-cli upload`.
- Also upload a standalone `manifest.json` at the dataset root (for cheap version polling without downloading the tarball).

### 8.7 Brother's install ritual

Once per release:

```bash
git clone https://github.com/<dev>/rag-med.git
cd rag-med
cp .env.example .env  # then edit ANTHROPIC_API_KEY
./scripts/download_index.sh   # fetches bundle.tar.gz from HF, untars to data/
docker compose up -d
```

Total: ~25 minutes. App boots, models load, `/health` returns `200`, browser unlocks input. He's ready to ask questions.

---

## 9. Phase 3: how quality is measured

Eval is how we know whether a code change made things better or worse.

### 9.1 The gold set

290 questions, organized in `data/gold_set/`:

| Subset | Count | Source | Purpose |
|---|---|---|---|
| Brother | 50 | hand-written from his real research | Headline metrics |
| Synthetic | 150 | LLM-generated from corpus, 20 spot-checked | Volume |
| BioASQ | 80 | external pneumology slice | External validity |
| Adversarial | 10 | out-of-scope ("capital of France") | Refusal honesty gate |

Each `gold_item`:

```python
class GoldItem:
    question_id: str            # "g042"
    question: str
    tags: dict                  # {section_focus, topic, difficulty}
    relevance: dict[str, str]   # PMID → "relevant"|"partial"|"not_relevant"
    author: str                 # "brother" | "synthetic" | "bioasq" | "adversarial"
```

**Paper-level (PMID), not chunk-level relevance** (locked Q3). Brother labels at PMID granularity in ~25 minutes total. Chunk-level labels would have taken ~3 hours and would break every time we changed the chunker. PMID labels survive chunker changes.

### 9.2 How eval calls the system

Direct Python imports — not HTTP. `from rag_med.serving.retrieve import retrieve` etc. Same process. No SSE parsing. Eval tests the *pipeline* quality, not the HTTP plumbing.

### 9.3 The three modes

| Mode | Flag | What runs | Cost / run |
|---|---|---|---|
| Retrieval-only | (default) | Steps 1–7 (embed, search, fuse, rerank) | $0 |
| Full | `--full` | Full pipeline including Claude generator + Haiku judge, **via Anthropic batch API** (50% off, 24h SLA) | ~$28 |
| Mock LLM | `--mock-llm` | Full pipeline with cached Claude responses on a 30-q subset | $0 |

Default is retrieval-only because most metrics (Recall@10, nDCG, MRR) don't need generation. Fast iteration.

`--full` is gated to milestone runs: M2 baseline, M4 first calibration, M5 post-scale, M6 acceptance, +1 slack = **5 runs total budget across M1–M6 (~$140)**.

### 9.4 What gets measured

- **Retrieval:** Recall@10 (paper-level, strict), Recall@50, nDCG@10, MRR. Implemented via `pytrec_eval`.
  - **Recall@10 plain English:** "out of all the papers labeled relevant for this question, what fraction does the system surface in the top-10 chunks?"
  - **nDCG plain English:** like Recall but with bonus credit for ranking the relevant ones higher.
  - **MRR plain English:** "where in the ranked list is the first relevant paper, on average?" Reciprocal of that rank.
- **End-to-end (when `--full`):** Faithfulness (% sentences `supported`), Citation accuracy, Answer relevance.
- **Honesty:** % `hard_refusal` on the 10-question adversarial slice.
- **Engineering:** Latency p50, p95.
- **Slices:** Faithfulness conditional on `section_type` of cited chunks. Tells us if abstract chunks are dominating retrieval inappropriately (Q7 instrumentation).

### 9.5 Output and comparison

Per-run Parquet file: `results/run_<git_sha>_<timestamp>.parquet`. Columns: `(run_id, question_id, retrieved_chunk_ids, status_per_sentence, latency_ms, tags…, bundle_version)`.

`python -m eval compare <runA.parquet> <runB.parquet>` outputs:
- Per-metric delta table (Recall@10 +0.03, faithfulness +0.05, ...).
- Questions where verdict flipped (regression debug).
- Top 20 failing questions with full traces.

Comparing across `bundle_version` is a feature, not a bug — it shows whether re-indexing helped or hurt.

### 9.6 The acceptance gates (Q15)

Five gates. M6 ships only when **all five** are met or honestly marked "below target" in the writeup.

| Gate | Metric | Target |
|---|---|---|
| Retrieval | Recall@10, paper-level, strict | ≥ 0.65 on brother's 50-q set |
| Faithfulness | % sentences `supported` | ≥ 0.80 |
| Refusal honesty | % `hard_refusal` on 10-q adversarial slice | ≥ 0.80 |
| Latency | p95 end-to-end on i7-155H | ≤ 20 s |
| User accept | Brother says "I'd use this in actual research" | Yes |

---

## 10. The data — what lives where

A map of every persistent artifact.

### 10.1 In the bundle (downloaded to `data/`)

| File | What | Who reads | Who writes |
|---|---|---|---|
| `data/sqlite.db` | `papers`, `chunks`, `failed_papers`, `index_manifest` tables | Phase 2 (read), Phase 3 (read) | Phase 1 |
| `data/faiss.index` | 250k × 768-dim vectors | Phase 2 (read), Phase 3 (read) | Phase 1 |
| `data/bm25.pkl` | Pickled `rank_bm25` index | Phase 2 (read), Phase 3 (read) | Phase 1 |

### 10.2 Created at runtime by Phase 2

| File | What | Who reads | Who writes |
|---|---|---|---|
| `data/sqlite.db::query_traces` table | One row per `/ask` call. Forensics. | Phase 3 (eval optional), dev | Phase 2 |
| stdout (Docker logs) | Structured JSON log lines | Brother (errors), dev (debug) | Phase 2 |

### 10.3 Created at runtime by Phase 3

| File | What | Who reads | Who writes |
|---|---|---|---|
| `results/run_<git_sha>_<timestamp>.parquet` | Eval results | `python -m eval compare`, dev | Phase 3 |

### 10.4 Config

| File | Committed? | What |
|---|---|---|
| `.env` | gitignored (copy from `.env.example`) | Secrets only: `ANTHROPIC_API_KEY` |
| `config.yaml` | committed | Tunables: rerank_floor, NLI thresholds, model names, monthly cap, paths, bundle URL |

Pydantic Settings (`pydantic-settings`) reads both files and exposes a typed `settings` singleton on boot.

### 10.5 Models (cached locally)

| Model | Where loaded | Size |
|---|---|---|
| `MedCPT-Query-Encoder` | RAM at boot | ~400 MB |
| `MedCPT-Cross-Encoder` | RAM at boot | ~400 MB |
| `MedCPT-Article-Encoder` | Phase 1 only, dev machine | ~400 MB |
| `microsoft/deberta-v3-large-mnli` | RAM at boot | ~1.5 GB |
| `claude-sonnet-4-6` | Anthropic API (network) | n/a |
| `claude-haiku-4-5` | Anthropic API (network) | n/a |

The local models' weights live in HuggingFace's cache directory. Q19 (deferred): exact path; whether to bake into Docker image vs. mount as volume vs. auto-download on first boot.

---

## 11. What runs locally vs. what leaves the machine

This matters for cost, privacy, and brother's understanding.

### 11.1 Always local

- Question tokenization
- Question embedding (MedCPT-Query)
- FAISS search
- BM25 search
- RRF fusion
- Cross-encoder rerank (MedCPT-Cross)
- Sentence splitting
- NLI verification (DeBERTa)
- All SQLite reads/writes
- All log writes
- Frontend rendering

### 11.2 Leaves the machine (sent to Anthropic)

- The system prompt + the top-10 chunks + the question (sent to Claude Sonnet 4.6 generator).
- The judge prompt + one chunk + one sentence (sent to Claude Haiku 4.5 judge), once per borderline pair (~30% of pairs).

Nothing else. The corpus, the index, the embeddings, the question vectors, the NLI scores, the user's history — all stay on brother's machine.

The corpus is public-domain (PubMed/PMC OA), so even what leaves isn't sensitive in any meaningful sense.

---

## 12. Latency and cost budget per query

### 12.1 Latency (target ≤20 s p95 on i7-155H)

| Stage | Time |
|---|---|
| FAISS + BM25 (parallel) | ~0.3 s |
| Cross-encoder rerank top-50 | ~1.0 s |
| Prompt build | <0.05 s |
| Claude Sonnet stream (network + generation) | ~4–8 s |
| Sentence split | <0.05 s |
| NLI batch over ~10 sentences × ~1.5 cited chunks each | ~3–5 s |
| Haiku judge calls (parallel, ~30% of pairs) | ~2–3 s |
| Final emit | <0.05 s |
| **Total typical** | **~10–15 s** |
| **Total p95 budget** | **≤ 20 s** |

User-perceived split: brother sees text streaming after ~0.5 s, finishes reading the answer around ~6–8 s, then waits ~5–8 s for dots. v1 acceptable; post-v1 path is incremental verify (verify each sentence as it streams).

### 12.2 Cost (target ≤$15/month default cap)

| Component | Cost |
|---|---|
| Sonnet generator | ~$0.015–0.020/query |
| Haiku judge (~30% of sentences) | ~$0.003–0.005/query |
| **Total per query** | **~$0.02** |
| Brother monthly (30 q/day × 30 days) | **~$18/month** |
| App-level cap default | $15/month — brother edits `config.yaml` to raise |
| `--full` eval (batch API enforced, 290 q) | ~$28/run |
| Dev budget M1–M6 | **~$140 total** (5 `--full` runs × $28 = $140 floor + ad-hoc) |

Per-query cost split is recorded into `query_traces.generator_cost_usd` and `judge_cost_usd` for forensics.

**Five-layer cost defense (Q21).** App-level cap alone defends against runaway production but not against dev burns or single-query blowups. Five independent layers, ~50 LOC total:

1. **App-level monthly cap** — `monthly_cap_usd: 15`, summed from `query_traces`. Blocks `/ask` with clean error event when exceeded.
2. **Anthropic console hard limit** — set per-key on the dashboard (NOT `config.yaml`). Dev key $50/mo, brother's key $30/mo. Backstop against any code bug that bypasses layer 1.
3. **Per-query ceiling** — `per_query_ceiling_usd: 0.10`. Aborts a single query mid-pipeline if generator + judge cost exceeds (5× typical, catches runaway loops without false-tripping legit queries).
4. **`max_tokens: 1024`** on generator — bounds single-completion length. Without it, Sonnet can ramble to 64k tokens = $0.96 per call worst case.
5. **80% MTD warning** — `warn_threshold_pct: 0.80`. Log loud + `/health.cost_warning=true` + frontend yellow banner. Soft signal before hard fail.

**`--full` eval safety guards (Q21):**
- Confirmation prompt — `python -m eval --full` prints expected cost + question count, requires `YES` typed back.
- Batch API not optional — `--full` enforces batch endpoint; fails loud if unavailable rather than silently using non-batch ($60 footgun closed).
- Run log — `eval/runs.jsonl` records each run (timestamp, git_sha, expected/actual cost, gold-set size). `tail eval/runs.jsonl` answers "did I already run this?"

**`python -m rag_med cost` CLI** — read-only SQL over `query_traces`. Prints MTD spend, generator vs judge split, days remaining in month, projected end-of-month spend at current rate.

---

## 13. A worked example end-to-end

Brother types:

> What is the long-term FEV1 decline rate in COPD patients on triple therapy?

### t = 0.0 s — Browser

`fetch('/ask', {body: {...}})` opens the stream.

### t = 0.05 s — FastAPI accepts

Pydantic validates. Pipeline starts.

### t = 0.05–0.4 s — Retrieval

In parallel:
- MedCPT-Query embeds the question → 768-dim vector.
- Biomedical tokenizer produces `["long-term", "fev1", "decline", "rate", "copd", "triple", "therapy"]`.

Then in parallel again:
- FAISS returns 50 chunks (Vestbo 2013 results, Calverley 2007 methods, …).
- BM25 returns 50 chunks (overlap with FAISS plus a few rare-token finds).

RRF fuses → 50 unique candidates.

### t = 0.4–1.4 s — Rerank + emit retrieved

Cross-encoder scores all 50. Top-10 returned. Top-1 score 0.84 → above `rerank_floor` 0.0. No hard refusal.

`event: retrieved` fires with chunk previews. Frontend renders the panel.

### t = 1.4–7.0 s — Generator stream

Prompt built. Sonnet streams. Brother sees:

```
The TORCH trial reported an annual FEV1 decline of "39 mL/year" in
patients on triple therapy [1]. The IMPACT trial confirmed a similar
"33 mL/year" rate over 52 weeks [2]. Vestbo et al. found that decline
correlated with exacerbation frequency [3]. These findings are
encouraging for clinical practice [3].
```

Four sentences. Streamed live, brother already reading by t = 4 s.

### t = 7.0–7.05 s — Sentence split

Four sentences detected. Citations parsed: `[1]`, `[2]`, `[3]`, `[3]`. All resolve to real chunks (no fabricated citation).

### t = 7.05–11.0 s — NLI batch

Four sentences × 1 cited chunk each = 4 pairs. One batched forward pass through DeBERTa.

| Sentence | NLI entailment |
|---|---|
| 1 (TORCH 39 mL) | 0.93 (confident supported) |
| 2 (IMPACT 33 mL) | 0.91 (confident supported) |
| 3 (Vestbo correlation) | 0.72 (borderline → judge) |
| 4 (encouraging) | 0.41 (borderline → judge) |

### t = 11.0–13.5 s — Judge (parallel)

Two Haiku calls dispatched concurrently.

- Sentence 3: judge returns `{"verdict": "supported", "confidence": 0.85}`.
- Sentence 4: judge returns `{"verdict": "unclear", "confidence": 0.5}`. ("These findings are encouraging" is too vague to verify against a Discussion paragraph.)

### t = 13.5 s — Finalize + emit

| Sentence | Status | failure_kind | Dot |
|---|---|---|---|
| 1 | supported | null | green |
| 2 | supported | null | green |
| 3 | supported | null | green |
| 4 | unclear | judge_inconclusive | yellow |

`event: verified` fires with all four. Frontend re-renders the answer block with dots inline. Footer: "3/4 claims verified · 1 unclear · 0 unsupported".

`event: done`. Stream closes.

### t = 13.5 s — Trace written

```json
{
  "query_id": "q_2026_05_10_18_23_44_a3f1",
  "git_sha": "abc1234",
  "bundle_version": "v1.0.0",
  "question": "What is the long-term FEV1 decline rate…",
  "final_chunks": ["12345_results_02", "67890_results_01", "54321_discussion_03"],
  "section_type_histogram": {"results": 2, "discussion": 1},
  "answer_sentences": 4,
  "supported": 3, "unclear": 1, "unsupported": 0, "unknown": 0,
  "refusal": null,
  "generator_cost_usd": 0.018,
  "judge_cost_usd": 0.004,
  "total_cost_usd": 0.022,
  "latency_ms": 13500
}
```

### t = 13.5 s — Brother

Reads sentences 1–3 with green dots, trusts the numbers. Sees yellow on sentence 4, knows that's filler, ignores it. Clicks `[1]` → citation panel slides in showing the TORCH paper. Confident he has a real number to cite in his thesis chapter.

That's what we're building.

---

## 14. Glossary of the moving parts

The full ubiquitous-language doc is `glossary.md`. Here's the minimal subset you need to walk this codebase:

| Term | What it is |
|---|---|
| `paper` | One PMID-keyed source document. Row in `papers` table. |
| `chunk` | One ~350-DeBERTa-token unit of paper text. Row in `chunks` table + vector in FAISS + entry in BM25. |
| `chunk_id` | `{pmid}_{section_type}_{ordinal:02d}`, e.g. `12345678_methods_03`. |
| `question` | What the user typed. Never call this "query". |
| `query_vector` | 768-dim MedCPT embedding of the question. |
| `bm25_tokens` | Biomedical-tokenized question. |
| `final_chunks` | The top-K chunks sent into the prompt. List of chunk_ids. |
| `answer` | The full text Claude produced. Never "response", "completion", "output". |
| `sentence` (code) / `claim` (UI) | One unit of verification. |
| `citation` | The `[n]` marker in the answer. Never "reference", "source". |
| `cited_chunk_ids` | The resolved chunks a sentence's `[n]`s point to. |
| `status` | One of `supported` / `unclear` / `unsupported` / `unknown`. |
| `failure_kind` | Forensic tag when `status != supported`. Enum in `glossary.md`. |
| `generator` / `judge` | Sonnet 4.6 / Haiku 4.5. Two separate clients. Never one shared `claude_client`. |
| `gold_set` | The 290-question eval corpus. |
| `gold_item` | One question + paper-level relevance labels. |
| `bundle` | The downloadable tarball: SQLite + FAISS + BM25 + manifest. |
| `query_traces` | The persistent SQL log, one row per `/ask`. |
| `hard_refusal` / `soft_refusal` | Two refusal types, both tracked. |

**Words you should never write:** `article`, `document`, `query` (for user input), `response`, `completion`, `delta`, `reference` (for `[n]`), `source` (for `[n]`), `semantic_search`, `keyword_search`, `claude_client`, bare `n_tokens`, `chunk-level relevance`. (Full banned list in `glossary.md`.)

---

## How to use this document

- **Before M1:** read this top-to-bottom (especially §0). Then read `architecture.md` and `decisions.md`. Three docs, one mental model.
- **During M1:** keep this open in a tab. When code disagrees with a step here, that's either a bug or a real decision worth recording back to `decisions.md`.
- **After M2:** update §3 if the verifier sub-stage shape changes. Update §13 with real measured numbers from the toy corpus once they exist.
- **Before showing brother:** §0 + §2 + §13 are the brother-facing summary. The rest is dev-facing.

If something in the implementation contradicts something here and is not a bug, update this document in the same PR. Walkthrough drift is the worst kind of drift — it hides the system from the people building it.
