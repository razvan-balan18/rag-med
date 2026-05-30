# rag-med
A question answering tool for pneumology research papers (COPD, asthma, ILD, PH). You ask a question, it finds the most relevant passages from open sourced papers and has Claude write an answer grounded from them. Before showing the answer, every sentence is checked against its sources, so you can see what's actually supported by the evidence and what isn't. Hybrid retrieval (FAISS and BM25) and rerank mechanism, Sonnet generator + NLI and Haiku judge that verifies each sentence against evidence.

# Note from creator
Agentic AI has played a big role in the making of this project. For transparency, I decided to include the .claude folder, with all the settings and decisions. The files in the .claude/research have been created with the /grill-me skill. For more details check Mat Pocock's set of skills, which helped a lot during the development of this project. Decided to do TDD for the most part.
I tried to review the ai generated code to the best of my abilities, being a learning project. So, if necessary, kindly review. Cheers.

# Main DB structure for the RAG

<table>
<tr valign="top">
<td width="50%">

<b><code>papers</code></b>

<table>
<tr><th>Column</th><th>Notes</th></tr>
<tr><td><code>pmid</code> (PK)</td><td>all papers have a pubmed ID</td></tr>
<tr><td><code>pmcid</code></td><td>for full papers, pmid can be just the abstract and title</td></tr>
<tr><td><code>doi</code></td><td>digital object identifier</td></tr>
<tr><td><code>title</code></td><td></td></tr>
<tr><td><code>journal</code></td><td></td></tr>
<tr><td><code>year</code></td><td></td></tr>
<tr><td><code>source_type</code></td><td>abstract or full_text</td></tr>
<tr><td><code>mesh_terms_json</code></td><td>medical subject headings</td></tr>
<tr><td><code>fetched_at</code></td><td></td></tr>
</table>

</td>
<td width="50%">

<b><code>chunks</code></b> — all papers get transformed into 300-400 sized deberta tokens

<table>
<tr><th>Column</th><th>Notes</th></tr>
<tr><td><code>chunk_id</code> (PK)</td><td></td></tr>
<tr><td><code>pmid</code></td><td>references <code>papers(pmid)</code></td></tr>
<tr><td><code>section_type</code></td><td></td></tr>
<tr><td><code>ordinal</code></td><td></td></tr>
<tr><td><code>text</code></td><td></td></tr>
<tr><td><code>n_deberta_tokens</code></td><td></td></tr>
<tr><td><code>n_medcpt_tokens</code></td><td></td></tr>
</table>

</td>
</tr>
<tr valign="top">
<td width="50%">

<b><code>paper_xml</code></b>

<table>
<tr><th>Column</th><th>Notes</th></tr>
<tr><td><code>pmid</code></td><td>references <code>papers(pmid)</code></td></tr>
<tr><td><code>raw_xml</code></td><td></td></tr>
<tr><td><code>parsed_at</code></td><td></td></tr>
</table>

</td>
<td width="50%">

<b><code>failed_papers</code></b> — for papers that failed to process

<table>
<tr><th>Column</th><th>Notes</th></tr>
<tr><td><code>pmid</code></td><td></td></tr>
<tr><td><code>failure_reason</code></td><td></td></tr>
<tr><td><code>attempted_at</code></td><td></td></tr>
</table>

</td>
</tr>
</table>


## Install
Requires Python 3.11+ and uv.

```bash
uv sync                  # install deps into .venv
cp .env.example .env     # fill in NCBI + Anthropic keys
```
