# GraphRAG-QA — Methodology

This document describes every algorithm used by the system, with the exact
formulas and default parameters implemented in the code. File references point
to `backend/app/`.

## 1. Document ingestion (`ingestion/`)

| Step | Implementation |
|---|---|
| Validation | extension ∈ {.pdf, .txt, .docx}, size ≤ 25 MB, non-empty; filename sanitised (directory parts removed, ASCII only) |
| Identity | `doc_` + first 12 hex chars of SHA-256(file bytes) — identical uploads are detected as duplicates |
| PDF | PyMuPDF `page.get_text("text", sort=True)`, one record per physical page |
| TXT | UTF-8 / UTF-16 / Windows-1252 decoding; form-feed (`\f`) = page break |
| DOCX | python-docx body in document order (paragraphs and tables); `Heading*`/`Title` styles start sections; explicit and last-rendered page breaks give logical pages |
| Normalisation | NFKC, control/zero-width characters removed, `infor-\nmation` → `information`, lines joined within paragraphs, paragraph breaks kept, whitespace collapsed |

Every page keeps its character offset inside the full document text, so any
chunk offset can be mapped back to a page.

## 2. Chunking (`preprocessing/chunker.py`)

Tokens are word-level (`\w+(?:[-'’]\w+)*|[^\w\s]`). Default chunk size 400
tokens, overlap 80.

1. Split into sentences (rule-based, protects abbreviations and initials).
2. A sentence longer than the budget is split into token windows (the only case where a sentence is cut).
3. Pack consecutive sentences while the total stays ≤ `chunk_size`.
4. The next chunk restarts with the trailing sentences of the previous chunk whose total is ≤ `chunk_overlap`.

Chunks never cross a section boundary; sections with fewer than 25 tokens
(e.g. a title) are merged into a neighbour. Chunk text is an exact slice of the
document text (`text == doc.text[char_start:char_end]`).

## 3. Entity and relation extraction (`entities/`)

* spaCy `en_core_web_sm` NER. Kept labels: PERSON, ORG, GPE, LOC, FAC, NORP, EVENT,
  WORK_OF_ART, LAW, PRODUCT, LANGUAGE, DATE (dates only if they contain a year or month).
* Normalisation: NFKC, leading article and possessive removed, punctuation removed,
  dotted acronyms collapsed (`U.S.` → `us`), case-folded.
* **Duplicate resolution** (conservative): identical normalised form; dropping middle
  initials (`Geoffrey E. Hinton` → `Geoffrey Hinton`); a one-word PERSON equal to the
  surname of exactly one multi-word PERSON (same document first, then corpus); an
  all-caps acronym equal to the initials of exactly one multi-word entity.
* **Relations**: for two entities in one sentence whose dependency paths meet at a verb
  (each ≤ 3 arcs away), the verb lemma (surface form for passives) plus the preposition
  introducing the second entity becomes the label (`work at`, `born in`, `acquire`).
* Entity specificity (IDF-like): `spec(e) = log(1 + N / df(e)) / log(1 + N)`, where `N`
  is the number of passages and `df(e)` the number of passages mentioning `e`.

## 4. Embeddings (`embeddings/embedder.py`)

`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, L2-normalised (inner
product = cosine). The model reads at most 256 word pieces, so passages longer
than 180 words are split into sentence windows and the passage vector is the
normalised mean of the window vectors. Offline fallback: signed feature hashing
of stemmed unigrams and bigrams (reported as *degraded* in `/api/health`).

## 5. Hybrid retrieval (`retrieval/`)

* FAISS `IndexIDMap2(IndexFlatIP)` — exact cosine search with stable ids (incremental add/remove, persisted to `data/indexes/faiss.index`).
* BM25 Okapi (k1 = 1.5, b = 0.75) over stemmed, stop-word-filtered terms.
* Candidates = FAISS top-max(50, 5k) ∪ BM25 top-max(50, 5k).

$$\text{sem}(q,p)=\max(0,\cos(\mathbf e_q,\mathbf e_p)),\qquad
\text{bm25}_{norm}(q,p)=\frac{\text{BM25}(q,p)}{\max_{p'}\text{BM25}(q,p')}$$

$$\boxed{\text{Hybrid}(q,p)=\alpha\,\text{sem}(q,p)+(1-\alpha)\,\text{bm25}_{norm}(q,p)},\quad \alpha=0.7$$

## 6. Evidence graph (`graph/builder.py`)

Heterogeneous `networkx.MultiDiGraph`.

| Node type | Id | Edge type | Meaning (always backed by text or metadata) |
|---|---|---|---|
| document | `doc:<id>` | CONTAINS | document → passage |
| passage | `psg:<chunk id>` | MENTIONS | passage → entity (weight = mention count); question → entity |
| entity | `ent:<normalised>` | CO_OCCURS_WITH | two entities in one sentence (weight = count) |
| question | `q:question` | RELATED_TO | verb relation, keeps the source sentence |
| | | SUPPORTS | retrieved passage → question (weight = hybrid score) |
| | | CONNECTS_TO | passages sharing a *bridge entity*, or adjacent chunks of one document |

A bridge entity is any non-DATE entity mentioned in 2 … `max_df` passages,
`max_df = max(4, min(40, 0.25 N))` — hub entities (years, very frequent names)
would connect everything and are excluded. The corpus graph is serialised to
`data/graphs/corpus_graph.json` (with a fingerprint used as a cache key) and
`corpus_graph.graphml`.

**Question subgraph.** Hop 1 = hybrid top-k seeds plus passages that mention a
question entity; hop h = passages that share a bridge entity with (or are
adjacent to) hop h−1 passages, keeping the best `per_hop = 12` by
`hybrid + 0.2·spec(bridge)`, up to `max_hops = 3`. Kept entities: question
entities, bridges (≥ 2 included passages) and the 4 most specific entities per
passage.

## 7. Multi-hop evidence chains (`graph/traversal.py`)

The passage graph contains the question node, edges question→hop-1 passages and
CONNECTS_TO edges (≤ 6 best neighbours per passage). Chains are the simple paths
returned by `networkx.all_simple_paths(PG, q, targets, cutoff=max_hops)` —
simple paths never revisit a node (cycle prevention).

$$\text{score}(C)=0.45\,\overline{\text{rel}}(C)+0.40\,\text{cov}(C)+0.15\,\overline{\text{bridge}}(C)\,[|C|>1]-0.03\,(|C|-1)$$

* $\overline{\text{rel}}(C)=0.7\,\text{rel}(p_1)+0.3\,\text{mean}_{i\ge2}\,\text{rel}(p_i)$ (anchored: a bridge passage is often only weakly similar to the question)
* $\text{cov}(C)$ = IDF-weighted share of question content terms present in the chain's passages
* $\overline{\text{bridge}}$ = mean of $0.5\,\text{spec}(e)+0.5\,\text{anchor}(e)$ over bridge entities; `anchor(e)` = match score of a question entity that `e` is, or co-occurs with
* Pruning: every hop must add at least one new question term; passages need hybrid ≥ 0.05; chains over the same passage set are de-duplicated.

Each step carries a human-readable reason ("Linked to step 1 through the shared
entity 'Geoffrey Hinton'; adds question terms: phd").

## 8. Graph-heuristic reranking (`graph/heuristic_ranker.py`)

Personalized PageRank restarting at the question node (damping 0.85) on the
undirected subgraph; edge weights by type (SUPPORTS 2, MENTIONS 1, RELATED_TO 1,
CO_OCCURS_WITH 0.5, CONNECTS_TO 0.5, CONTAINS 0.2), edges touching an entity
scaled by its specificity.

$$\text{graph}(p)=0.5\,\text{Hybrid}(p)+0.25\,\frac{\text{PPR}(p)}{\max\text{PPR}}+0.25\,\frac{\max_{C\ni p}\text{score}(C)}{\max_C\text{score}(C)}$$

## 9. GNN reranking (`gnn/`)

**Features** (dimension 384 + 8): $\mathbf e(v)\odot\mathbf e_q$ plus
[cosine to question, BM25, hybrid, is-seed, hop distance/6, is-question-entity,
log-degree, specificity]. Node text: question/passage embeddings; entity name
embedding; document = mean of its passages. Edge types are one-hot `edge_attr`.

**Architecture** (`GATReranker`, 43 210 trainable parameters with the defaults):

$$h^{(0)}=\text{ReLU}(W_0x+\text{type}(v)),\quad
h^{(l)}=\text{ReLU}\big(\text{GAT}^{(l)}(h^{(l-1)},A,E)\big)+h^{(l-1)},\ l=1,2$$

$$\text{GAT attention: }\alpha_{ij}=\text{softmax}_j\big(\text{LeakyReLU}(a^\top[Wh_i\,\|\,Wh_j\,\|\,W_e e_{ij}])\big),\ 4\text{ heads}$$

$$s(v)=\text{MLP}([h^{(0)}\|h^{(2)}]),\qquad P(\text{evidence}\mid v)=\sigma(s(v))$$

Dropout 0.2 after the projection and each GAT layer.

**Training data.** HotpotQA distractor setting (Yang et al., 2018): every
question has 10 Wikipedia paragraphs, 2 of them gold. `scripts/download_datasets.py`
samples 1 000 train + 200 validation questions from the official *train* split (plus, with `--extra-train 2000`, 2 000 further disjoint training questions — the shipped model uses all 3 000)
and 300 test questions from the official *dev* split (seed 42). Each question is
turned into a question subgraph exactly as at inference time (top-k = 4,
max hops = 2); gold paragraphs are positives.

**Objective.** Binary cross-entropy on passage nodes with negative sampling:
per graph, up to 3 negatives per positive, half *hard* (highest hybrid score),
half random. Adam (lr 1e-3, weight decay 1e-4), batch 16 graphs, ≤ 40 epochs,
early stopping (patience 8) on validation MRR; the best checkpoint is saved with
its metadata (training source, embedding model, feature-spec version). A
checkpoint is only used if the feature specification and embedding model match;
otherwise the system falls back to the graph heuristic and says so.

## 10. Answer generation (`generation/`)

Evidence = top evidence chain (for graph/GNN strategies) followed by the
strategy's top passages (max 5). Backends (`auto` order):

1. **Flan-T5-base** on a compact context (chain passages only when the chain has ≥ 2 steps; the 3–4 most relevant sentences per passage; a pronoun-initial sentence brings its predecessor; ≤ 330 words). A cautious prompt (“…If it cannot be answered, say "unanswerable"”) runs first; if it abstains, a plain reading-comprehension prompt runs and its answer is accepted only after grounding, with half the support.
2. **TinyRoBERTa-SQuAD2** extractive reader over each passage and the concatenated chain; span probability = σ(span score − no-answer score).
3. **Extractive** (model-free) sentence scoring: 0.5·cosine + 0.3·term overlap + 0.2·answer-type match — used only when no model is available.
4. Optional **Ollama** (local LLM) or **Claude API** backends with a strict JSON schema; their citations are validated the same way.

**Grounding.** An answer is accepted only if it occurs in an evidence passage
(case-insensitive, or SQuAD-normalised) or has token-F1 ≥ 0.6 with an evidence
sentence. If a model reads the evidence and abstains, the extractive scorer is
*not* allowed to overrule it.

**Reasoning summary.** Deterministic steps: one **fact** per chain passage (the
sentence that mentions the bridge entity / answer, quoted with its citation),
then an **inference** step (“Combining [1] and [2], the answer is …”).

## 11. Citations and confidence

For every claim (the answer and each fact step) and each cited passage, the
supporting sentence is located; support = 1 if the claim occurs verbatim, else
the share of the claim's content terms found in the sentence. Citations with
support < 0.3 are removed.

$$\text{confidence}=0.30\,S_{ret}+0.25\,S_{cov}+0.15\,S_{chain}+0.30\,S_{ans}$$

$S_{ret}$ = best evidence cosine rescaled from [0.10, 0.70]; $S_{cov}$ = question-term
coverage; $S_{chain}$ = best chain score; $S_{ans}$ = generator support (halved
after an abstention, 0 if ungrounded). The answer becomes **“Insufficient
evidence found to answer this question reliably.”** if any hard check fails:
best similarity < 0.22, coverage < 0.30, none of the named question entities
occurs in the evidence, the generator abstained or was not grounded, no
citation survived, or confidence < 0.30.

## 12. Evaluation (`evaluation/`)

For each question the analysis is computed once; every strategy selects its own
evidence and answer from it (identical candidates).

| Metric | Definition |
|---|---|
| EM | normalised prediction equals a normalised gold answer (SQuAD normalisation) |
| F1 | token-level F1, max over gold aliases |
| P@K / R@K | share of the top-K passages that are gold / share of gold passages in the top-K (K = 2) |
| MRR | 1 / rank of the first gold passage |
| Evidence recall | share of gold passages inside the evidence given to the generator |
| Citation coverage / precision | share of gold passages cited / share of cited passages that are gold |
| Abstention accuracy | share of unanswerable questions answered with the insufficient-evidence message |
| Latencies | wall clock of analysis + generation; graph construction + traversal |

Unanswerable questions score EM = F1 = 1 when the system abstains.
