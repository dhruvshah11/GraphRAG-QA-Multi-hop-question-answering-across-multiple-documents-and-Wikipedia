# GraphRAG-QA: Viva and Live-Demo Script

This is a practical script for the viva. It covers what to set up, what to click, what to say, which questions to
ask live (all checked on the running app on 22 Nov 2026), how to read each graph view, and short, honest answers to
likely examiner questions. Every number here comes from `results/`. Re-check them if you re-run anything.

---

## 0. Setup, 15 minutes before

Run each step in its own PowerShell window, from the `graphrag-qa` folder.

1. Backend:
   ```powershell
   cd backend
   ..\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. Frontend:
   ```powershell
   cd frontend
   npm run dev
   ```
3. Check health: open <http://127.0.0.1:8000/api/health>. Every component should say `ok`. Then open
   <http://127.0.0.1:8000/api/gnn/status>, which should show `"available": true`.
4. If the Documents page is empty: `.\.venv\Scripts\python scripts\seed_demo.py` loads the 5 sample documents.
5. **Warm-up.** Ask Q1 below once before the examiners arrive. The first answer after start-up loads Flan-T5
   (about 2–3 s extra), and later answers take about 0.5–2 s.
6. Have these tabs open: the app at <http://127.0.0.1:5173>, the API docs at <http://127.0.0.1:8000/docs>, and the
   slide deck.
7. Fallback if anything breaks: the real screenshots in `docs/screenshots/` (01–09) show every page. Everything
   except Wikipedia mode runs offline, so you don't need Wi-Fi.

> **Do not demo Wikipedia mode live** unless you have set `GRQA_WIKIPEDIA_USER_AGENT` to your own contact
> details and tried it beforehand. Without that, Wikipedia returns HTTP 403 (the app reports this clearly, but it
> wastes demo time). Say instead: "Open-domain mode is implemented and tested with a stubbed HTTP layer; live
> calls need a policy-compliant User-Agent."

---

## 1. The demo flow (about 10 minutes)

| Time | Page | Do | Say (key points) |
|---|---|---|---|
| 0:00 | **Dashboard** | Point at the counts: documents, passages, entities, and graph node/edge types. | "Five documents (2 PDF, 2 TXT, 1 DOCX) ingested into one evidence graph. Every number on this page comes from the backend." |
| 0:45 | **Documents** | Open `nobel_prizes_2024.pdf` and show pages, chunks and entities. | "Text is extracted with page numbers, split into 400-token sentence-aware chunks with 80-token overlap, and spaCy finds the entities. Entity names are normalised, so *Geoffrey E. Hinton* and *Hinton* become one node." |
| 2:00 | **Question Answering**, Q1 | Ask Q1 with strategy **GNN (GAT) rerank**. | See §2, Q1. |
| 4:00 | **Question Answering**, Q4 contrast | Ask Q4 with **Hybrid**, then again with **GNN**. | See §2, Q4. This is the strongest single argument for the project. |
| 5:30 | **Question Answering**, Q11 | Ask the FIFA question. | See §2, Q11. "The system refuses to make things up." |
| 6:30 | **Graph Explorer** | *Open in Graph Explorer* from Q1, then show Evidence chain, Question subgraph and Full graph. | See §3. |
| 8:30 | **Analytics** | Per-passage score bars for the last question, rank changes and the GNN training curves. | "This shows *why* the ranking changed: each passage gets a FAISS, BM25, hybrid, graph and GNN score." |
| 9:15 | **Evaluation Lab** | Show the stored HotpotQA run. Optionally start a `demo` run (12 questions × 5 strategies, about 1 minute). | "Held-out results on 300 HotpotQA questions. Model selection used only the validation set; the test set was evaluated once." |
| 10:00 | **Settings** (optional) | Show chunk size, overlap, top-k, hops and α. | "All parameters are validated and live; changing chunking triggers reprocessing." |

---

## 2. Questions to ask live (results checked on the running app)

All of these use the 5 bundled documents. Answers, confidences and times were recorded with the GNN strategy on
22 Nov 2026. Times are wall-clock on the development laptop (CPU), after warm-up.

| # | Question | GNN answer | Confidence | Use it for |
|---|---|---|---|---|
| **Q1** | From which university did the 2024 Nobel Prize in Physics laureate from the University of Toronto receive his PhD? | **University of Edinburgh** ✓ | 0.67 | The main two-hop demo |
| Q2 | In which city was the AI company co-founded by a 2024 Nobel Prize in Chemistry laureate founded? | London ✓ | 0.80 | Second bridge question (Nobel PDF → DeepMind DOCX) |
| Q3 | Which co-founder of DNNresearch later became chief scientist of OpenAI? | Ilya Sutskever ✓ | 0.71 | TXT → TXT bridge |
| **Q4** | Which award did the scientist who supervised Ilya Sutskever's PhD share with Yann LeCun? | **ACM A.M. Turing Award** ✓ (Hybrid: *Nobel Prize in Physics 2024* ✗) | 0.73 | The strategy contrast |
| Q5 | Which company acquired the London AI lab whose chief executive shared the 2024 Nobel Prize in Chemistry? | Google ✓ | 0.87 | Bridge question |
| Q6 | In which city was the AlphaGo match against Lee Sedol held? | Seoul ✓ | 0.88 | Simple single-hop question |
| Q7 | When was the 2024 Nobel Prize in Physics announced? | 8 October 2024 ✓ | 0.93 | Single-hop question with a date answer |
| Q9 | Which deep learning team merged with the company that developed AlphaGo to form Google DeepMind? | Google Brain ✓ | 0.95 | Bridge question |
| **Q11** | Who won the 2030 FIFA World Cup final? | *Insufficient evidence found to answer this question reliably.* ✓ | 0.36 | Safe abstention |
| Q12 | What is the boiling point of liquid nitrogen in Kelvin? | *Insufficient evidence…* ✓ | 0.03 | Abstention with four failed checks |
| ⚠ Q10 | Which organisation was founded first, DeepMind or OpenAI? | **OpenAI ✗** (Graph heuristic: DeepMind ✓) | 0.95 | **Avoid with GNN.** Use only if asked about failure cases |

### Q1: the main demo (what to point at, top to bottom)
- **Final answer:** "University of Edinburgh", with confidence 0.67.
- **Confidence components:** retrieval 0.79, coverage 0.78, chain 0.71, answer support 0.44.
  - "Confidence is computed from measured signals, never by the language model:
    0.30·retrieval + 0.25·coverage + 0.15·chain + 0.30·answer support."
- **Reasoning summary** (facts vs inference):
  - [1] Nobel PDF, p.1: Hinton of the University of Toronto won the Physics prize.
  - [2] Hinton profile: "He then completed a PhD in artificial intelligence at the University of Edinburgh in 1978…"
  - Inference: "Combining [1] and [2]…"
  - "Facts are quoted and cited; the inference step is labelled as inference."
- **Evidence chain:** Nobel PDF passage → bridge entity **Geoffrey Hinton** → profile passage.
  - "Neither document alone answers the question. The graph finds the second document through the shared entity."
- **Citations:** click one and the exact supporting sentence is highlighted, along with the document, page, chunk,
  and the retrieval, graph, GNN and support scores.
- **Timings:** about 1.8 s in total. Answer generation takes about 1.7 s; the whole graph part (subgraph,
  traversal, ranking, GNN) is about 33 ms.
- **Generator note:** "The cautious prompt abstained; a second reading-comprehension pass produced this answer,
  which was accepted only after grounding." This is the two-pass Flan-T5 design. Mention it if asked.
- If asked why step [1] shows the heading "The Nobel Prize in Physics 2024": the supporting-sentence picker chose
  the sentence with the highest term overlap, which here is the heading. The cited passage itself contains the
  full fact. This is a known cosmetic limitation.

### Q4: the contrast (the most convincing two minutes)
1. Select **Hybrid**. The answer is **"Nobel Prize in Physics 2024"**, which is wrong. The question mentions a
   scientist, a supervisor and an award, and plain retrieval ranks the Nobel passage (an award, plus Hinton) first.
2. Select **GNN (GAT) rerank**. The answer is **"ACM A.M. Turing Award"**, which is right. The evidence chain is
   the Turing PDF passage ("Bengio, Geoffrey Hinton and Yann LeCun were the recipients of the 2018 ACM A.M. Turing
   Award"), then bridge **Geoffrey Hinton**, then the Hinton profile ("Sutskever completed his PhD at the
   University of Toronto under Hinton's supervision"). The graph strategies rank the Turing passage first.
3. Say: "The same retriever and the same generator; only the reranking differs. This is the multi-hop problem
   the project addresses."

### Q11: abstention
- Answer: the exact message *"Insufficient evidence found to answer this question reliably."*
- Reason shown: "None of the entities named in the question ('World Cup') appears in the evidence."
- Explain: the generator actually proposed a candidate (*South Korea*, from the AlphaGo passage). The
  **entity-grounding check** rejected it because the evidence is about something else, however similar it
  looks. This check was added after Flan-T5 hallucinated a FIFA answer during development.
- Q12 shows the other checks failing too: similarity 0.13 is below the 0.22 threshold, term coverage is 0%,
  no question entity appears in the evidence, and the generator found no answer.

### Q10: the honest failure case (only if asked, "Where does it fail?")
- With the GNN, the answer is "OpenAI" with confidence 0.95, which is wrong. With the graph heuristic, it is
  "DeepMind" (0.89), which is right.
- Why it fails:
  - It is a **comparison** question. The model must compare 2010 (DeepMind) with 2015 (OpenAI).
  - Both passages are retrieved, but the GNN ranks the OpenAI passage first, and the small Flan-T5 tends to pick
    the entity in the top passage.
  - High confidence here shows a real limitation: confidence measures *evidence support*, not whether the
    comparison logic was right.
- Consistent with the benchmark: on the 57 comparison questions of the test set, every strategy is weak (EM
  0.21–0.25), and the GNN is slightly below the heuristic (0.228 vs 0.246).
- Fix (future work): decompose comparison questions into sub-questions and compare the extracted values
  explicitly.

---

## 3. What each graph view shows (Graph Explorer)

**Colours (legend at bottom-left):**
- Question: red/rose.
- Document: sky blue.
- Passage: indigo.
- Entity: green, tinted by category (person, organisation, location…). Question entities have a dashed border.
- Edges:
  - CONTAINS: blue.
  - MENTIONS: indigo.
  - RELATED_TO: green.
  - CO_OCCURS_WITH: grey, dotted.
  - SUPPORTS: red.
  - CONNECTS_TO: amber, dashed.

**Edge types and what creates them:**

| Edge | From → To | Created when |
|---|---|---|
| CONTAINS | document → passage | the chunk belongs to the document |
| MENTIONS | passage → entity | spaCy found the entity in the passage |
| RELATED_TO | entity → entity | a verb relation in one sentence (dependency parse), e.g. *Hinton —supervised→ Sutskever* |
| CO_OCCURS_WITH | entity ↔ entity | the two entities appear in the same sentence (weight = count) |
| SUPPORTS | passage → question | the passage was retrieved for this question (weight = hybrid score) |
| CONNECTS_TO | passage ↔ passage | the passages share a **bridge entity**, or are consecutive chunks of one document |

A bridge entity must not be a DATE (years link passages only by coincidence). It must also appear in at least 2
passages and in no more than max(4, min(40, 25% of passages)), so that hub entities do not connect everything.

### View 1: Evidence chain (`view=chain`)
- **Shows:** only the winning reasoning path for one answered question. For Q1: the question, then the Nobel PDF
  passage, then the Geoffrey Hinton entity, then the Hinton profile passage. All other nodes are hidden.
- **How it was found:**
  - `networkx.all_simple_paths` from the question over the passage graph, up to `max_hops = 3`. Simple paths
    never repeat a node, which rules out cycles.
  - A hop is kept only if it adds at least one new question term.
  - Each chain is scored as
    `0.45·anchored relevance + 0.40·question-term coverage + 0.15·bridge specificity − 0.03·(length − 1)`.
    Anchored relevance is 0.7·rel(first passage) + 0.3·mean rel(the others).
- **Say:** "This is the explanation: *why* two documents were combined, and through which entity."

### View 2: Question subgraph (`view=subgraph`, the default after asking)
- **Shows:** the per-question graph that the rankers actually run on. For Q1 it has 58 nodes and 201 edges: 1
  question, 5 documents, 8 passages and 44 entities. By edge type: CONTAINS 8, MENTIONS 68, RELATED_TO 32,
  CO_OCCURS_WITH 69, SUPPORTS 8, CONNECTS_TO 16.
- **How it was built:**
  - Start from the top-k retrieved passages (SUPPORTS edges).
  - Add the passages' documents and entities.
  - Expand through non-DATE bridge entities to passages that were *not* retrieved, up to the hop limit.
  - The expansion log in the response lists what was added and why.
- **Highlighting:** the chain's nodes are emphasised and the rest are dimmed. Click any node to see its details
  and neighbours (loaded from `/api/graph/node/{id}/neighbors`).
- **Say:** "Both rankers run on this graph:
  - The graph heuristic: 0.5·hybrid + 0.25·Personalized PageRank (restart at the question node, damping 0.85,
    with hub entities down-weighted) + 0.25·chain support.
  - The GNN: a GAT over the same nodes and typed edges."

### View 3: Full graph (`view=full`)
- **Shows:** the corpus-level evidence graph over all 5 documents (118 nodes and 527 edges in the benchmark
  run), with no question node.
- **Point at:** the amber dashed CONNECTS_TO edges *between documents*, for example via Geoffrey Hinton, Demis
  Hassabis or Ilya Sutskever. "These cross-document links are what make multi-hop answering possible."
- Search or filter by node type to find an entity; the legend shows counts per node and edge type.

---

## 4. Numbers to remember

| What | Value | Source |
|---|---|---|
| Test set | 300 HotpotQA questions (243 bridge, 57 comparison), distractor setting, official dev split | `results/retrieval_metrics.json` |
| Hybrid → graph heuristic → GNN, **EM** | 0.313 → **0.377** → 0.373 | same |
| **P@2** | 0.678 → 0.702 → **0.720** | same |
| **Evidence recall** (gold paragraphs passed to the answerer) | 0.818 → 0.898 → **0.910** | same |
| Citation precision | 0.697 → 0.699 → **0.714** | same |
| **Significance** (paired bootstrap, 10,000 resamples) | Both graph strategies beat hybrid **significantly** (e.g. GNN − hybrid P@2 +0.042, CI [+0.017, +0.067]). GNN vs heuristic is **not significant** (P@2 +0.018, CI [−0.003, +0.042]). | `results/significance.json` |
| GAT | 43,210 parameters; 2× GATConv, 4 heads; input = 384-d (passage ⊙ question embedding) + 8 scalar features = 392 | `models/gnn/last_training_summary.json` |
| Training | 3,000 questions, 200 validation; best epoch 13 of 21 (early stopping); val MRR 0.947 vs 0.937 (heuristic) and 0.926 (hybrid); 179 s of training plus about 25 min of graph construction | same |
| Data scaling | Trained on 1,000 questions: test P@2 0.677 (no better than hybrid). On 3,000: 0.720. | `results/previous_runs/` |
| Demo set (12 questions, used in development, so optimistic) | Heuristic 10/10 answerable correct; GNN 9/10; all strategies abstain correctly on both unanswerable questions | `results/retrieval_metrics.json` |
| Latency | About 1–2 s per answer on CPU; generation dominates; graph work about 25–35 ms; GNN inference about 13 ms | `results/latency_results.json` |
| Tests | 57 pytest tests, all passing | `results/test_results.txt` |

---

## 5. Likely examiner questions, with short answers

### Concept and motivation
1. **What problem does this solve?**
   Multi-hop questions whose answer needs facts from two or more documents, for example "the Toronto Nobel
   laureate's PhD university". Single-shot RAG retrieves passages that look like the question, but the second-hop
   passage often looks nothing like it (it is about Hinton's education, not the Nobel Prize). The evidence graph
   reaches it through the shared entity.
2. **What is the contribution / novelty?**
   An end-to-end, CPU-only system that combines:
   - hybrid retrieval;
   - an explicit, typed evidence graph built without an LLM;
   - multi-hop chain search;
   - a trained, edge-type-aware GAT reranker;
   - grounded generation with checked citations and principled abstention;
   - a UI that exposes every intermediate score.

   It is evaluated honestly on held-out HotpotQA with a significance test. We don't claim a new state of the art.
3. **How is this different from Microsoft GraphRAG?**
   GraphRAG (Edge et al., 2024) uses an LLM to extract a corpus-wide entity graph and community summaries, mainly
   for global "sense-making" questions. Ours builds the graph with spaCy (cheap, deterministic), builds a
   *per-question* subgraph, and targets precise multi-hop factoid QA with citations, on a laptop CPU. It is closer
   in spirit to HotpotQA graph models such as HGN and DFGN, and to HippoRAG's use of PageRank.
4. **Why a graph at all? Why not just retrieve more passages?**
   More passages means more noise for a small generator, and you still need to know *which* two belong together.
   The graph gives an explicit path, which is an explanation as well as a ranking signal. Measured: evidence recall
   rises from 0.818 to 0.898–0.910, and the gain is significant.

### Retrieval
5. **Why hybrid, and why α = 0.7?**
   - Dense (MiniLM) catches paraphrases; BM25 catches exact names and rare terms.
   - Hybrid = 0.7·max(0, cos) + 0.3·BM25/max(BM25). Measured on the test set, hybrid P@2 beats FAISS by +0.063 and
     BM25 by +0.042, both significant.
   - α = 0.7 is the specified default (semantic-dominant). It is adjustable in Settings. We did not run a full α
     sweep; that is an honest gap.
6. **Why MiniLM?**
   all-MiniLM-L6-v2: 384-d, about 22M parameters, fast on CPU, and strong on sentence-similarity benchmarks. Its
   input limit is 256 word pieces, so a 400-token passage is embedded as overlapping windows that are averaged
   and re-normalised.
7. **Why FAISS IndexFlatIP?**
   The vectors are L2-normalised, so the inner product equals cosine similarity. A flat index is exact, and with
   thousands of passages exact search takes milliseconds. `IndexIDMap2` lets us remove a document's vectors when
   it is deleted. For millions of passages we would switch to IVF or HNSW.
8. **Why 400-token chunks with 80 overlap?**
   400 tokens holds a few related sentences, which is enough context for a fact. The 80-token overlap avoids
   cutting a fact at a boundary. Chunks are sentence-aware (no mid-sentence cuts), and tiny sections are merged
   into neighbours. Both values are configurable; changing them triggers reprocessing.

### Graph and multi-hop
9. **How are entities matched across documents?**
   Normalisation (case, punctuation, dotted acronyms such as "U.S."), then conservative resolution: initials
   ("Geoffrey E. Hinton" = "Geoffrey Hinton"), surname to full name when unambiguous, and acronym to expansion.
   It is string-based; there is no linking to a knowledge base such as Wikidata (a stated limitation).
10. **Why exclude dates as bridges?**
    "2024" appears in unrelated passages. Linking through it creates spurious chains. Excluding DATE bridges fixed
    wrong chains during development.
11. **How do you prevent cycles and path explosion?**
    Simple paths only (no repeated nodes), a hop limit of 3 in the app (the HotpotQA benchmark used top-k 4 and 2
    hops, because each question has only 10 paragraphs), at most 6 neighbours per node and 5,000 paths, a
    minimum relevance per passage, and a hop is kept only if it adds a new question term.
12. **What is Personalized PageRank doing?**
    A random walk that restarts at the question node with probability 0.15. Passages that are well connected to
    the question through specific entities collect more probability. Edges are weighted by type (SUPPORTS 2.0,
    MENTIONS 1.0, RELATED_TO 1.0, CO_OCCURS_WITH 0.5, and so on). Edges touching generic hub entities are scaled
    down by entity specificity.

### GNN
13. **Describe the model.**
    - Input per node: (node embedding ⊙ question embedding), 384-d, plus 8 scalar features (cosine to the
      question, BM25, hybrid, is-seed, hop distance, is-question-entity, log degree, specificity), giving 392
      features.
    - A linear projection plus a node-type embedding.
    - Two GATConv layers (4 heads, using the one-hot edge type as `edge_attr`), with residual connections.
    - An MLP scorer on [h0 ‖ h2], plus a "wide" linear path from the 8 scalars straight to the logit.
    - One logit per passage. 43,210 parameters in total.
14. **Why GAT rather than GCN?**
    GCN averages neighbours with fixed weights. GAT learns *attention*: how much each neighbour matters, and,
    with edge features, how much each edge type matters. A passage connected through a specific bridge entity
    should count more than one connected through a generic co-occurrence.
15. **Why the wide path?**
    Without it, the strongest single signal (retrieval relevance) is diluted inside the 392-d projection. The
    first version (without it) ranked below hybrid retrieval on validation. With the wide path, the 1,000-question
    model reached parity (val MRR 0.928 vs 0.926 for hybrid; `results/previous_runs/gnn_1k_questions`).
16. **How was it trained?**
    - Each HotpotQA question becomes one graph (its 10 paragraphs). Labels: gold supporting paragraphs = 1.
    - Binary cross-entropy on positives plus 3 negatives per positive, half of them *hard* (the highest-scoring
      non-gold passages by hybrid score).
    - Adam, early stopping on validation MRR. Best epoch 13 of 21.
17. **How do you know it's really trained, and not memorising?**
    - The checkpoint stores the training history, the feature-spec version and the embedding model, and is only
      loaded if they match. A randomly initialised model is never used for ranking.
    - Results are on 300 *test* questions never seen in training or model selection. The 2,000 extra training
      questions were filtered to exclude validation and test IDs.
18. **The GNN beats the heuristic by only 1.8 points of P@2. Is that real?**
    No, not on this sample. The paired bootstrap 95% CI is [−0.003, +0.042], which includes 0. What *is*
    significant is that both graph rerankers beat retrieval alone. The honest claim: the learned reranker matches
    a carefully hand-designed one *without hand tuning*, and data scaling shows it is still improving (1,000 → 3,000
    questions: test P@2 0.677 → 0.720). A larger test set or more training data could separate them.
19. **Then why is EM not higher with a better ranker?**
    The answer step is now the bottleneck. Flan-T5-base (about 250M parameters) and a small SQuAD reader on CPU
    sometimes misread correct evidence. Evidence recall is 0.91, but EM is only 0.37.
20. **What did not work?**
    - A "v2" feature set that also fed the heuristic score into the GNN made validation worse (MRR 0.922) and was
      rejected *on validation*, never on test.
    - The first 1,000-question model did not beat hybrid ranking.

    Both are reported in README §26.3, and their files are kept in `results/previous_runs/`.

### Generation, hallucination and confidence
21. **How do you stop hallucination?**
    Several layers:
    1. The generator sees only the cited evidence passages.
    2. The answer must be found in (grounded in) the evidence text.
    3. Entity grounding: at least one named question entity must appear in the evidence.
    4. Citations are kept only if their support score is ≥ 0.3.
    5. Hard thresholds: semantic similarity ≥ 0.22, minimum question-term coverage, confidence ≥ 0.30.

    If any check fails, the system returns the fixed message "Insufficient evidence found to answer this question
    reliably." together with the reasons. Abstention accuracy on the demo set is 1.0.
22. **Is the confidence a probability?**
    No. It is a transparent weighted score of measured signals:
    0.30·retrieval + 0.25·coverage + 0.15·chain + 0.30·answer support. It is not calibrated. Q10 shows a wrong
    answer with 0.95 confidence, because the evidence was relevant even though the comparison was wrong.
    Calibration (for example, Platt scaling on validation data) is future work.
23. **Why Flan-T5 and not GPT/Claude?**
    The requirement was a system that runs locally and free on a laptop. Optional Ollama and Claude backends are
    implemented behind the same interface, but they were **not exercised** (no model or key configured). The
    system falls back automatically: Flan-T5, then the reader, then extractive.
24. **What is the two-pass prompting?**
    Pass 1 is a cautious prompt that may answer "unanswerable". If it abstains, pass 2 is a direct
    reading-comprehension prompt on chain-focused snippets. A pass-2 answer is accepted only if it passes
    grounding, and its answer-support score is halved.

### Evaluation
25. **Why HotpotQA?**
    It is the standard multi-hop QA benchmark, with gold supporting paragraphs, so retrieval (P@2, evidence
    recall) can be measured as well as answers. We use the distractor setting: each question has 10 paragraphs, 2
    gold and 8 distractors.
26. **Explain your metrics.**
    - EM and F1: the normalised answer string against gold.
    - P@2 and R@2: gold paragraphs in the top 2 (HotpotQA has 2 gold, so they are equal here).
    - MRR: 1 / rank of the first gold paragraph.
    - Evidence recall: the share of gold paragraphs that reach the generator.
    - Citation precision: the share of cited passages that are gold.
    - Abstention accuracy: correct refusals on unanswerable questions.
27. **Why is MRR the same (about 0.93) for all strategies?**
    The *first* gold paragraph is almost always ranked first already. The gains come from the *second* (bridge)
    paragraph, which P@2 and evidence recall capture.
28. **Why only 300 test questions?**
    CPU budget: 5 strategies × 300 questions × about 0.9 s, plus graph construction. It is enough to show
    significant gains over retrieval, but not enough to separate the two graph rerankers (we report that).
29. **The demo set shows 100%. Isn't that suspicious?**
    Yes, and we say so. The 12 demo questions were used during development, so those scores are optimistic. All
    conclusions rest on the held-out HotpotQA test set.
30. **Is it reproducible?**
    - Seeds are fixed. Every script regenerates the files in `results/`.
    - CPU inference is not bit-exact across runs: the heuristic row moved by about one question between runs, and
      we report that.
    - Commands: `scripts/run_benchmark.py`, `scripts/train_gnn.py --extra-train`, `scripts/significance_test.py`.

### Engineering
31. **Architecture?**
    - A layered FastAPI backend with a composition-root `Container` (dependency injection).
    - Services: ingestion, preprocessing, entities, embeddings, retrieval, graph, GNN, generation, citations,
      pipeline, evaluation.
    - A SQLite repository (WAL mode). FAISS and BM25 indexes persisted on disk.
    - A React/Vite/TypeScript frontend with 7 pages using Recharts and React Flow.
    - 15 PlantUML OOAD diagrams: use case, class, 6 sequence diagrams, activity, component, deployment, DFD levels
      0 and 1, architecture, and ER.
32. **How is it tested?**
    57 pytest tests: unit (extraction, chunking, NER, embeddings, FAISS, BM25, hybrid, graph nodes and edges,
    traversal, GAT forward pass, checkpoint loading, generation, citations, abstention), integration (an end-to-end
    two-document upload → ask), API endpoints, a frontend–backend contract test, and open-domain mode with a
    stubbed HTTP layer.
33. **What happens with a scanned PDF / a huge file / a malicious filename?**
    - Scanned PDFs have no text layer, and there is no OCR (a stated limitation). Ingestion fails with a clear
      message: "PDF contains no extractable text (scanned images are not supported; OCR is out of scope)".
    - File size is validated (HTTP 413).
    - Filenames are sanitised (`safe_filename`), and documents get checksum-based IDs, so duplicates are detected.
34. **How would you scale it?**
    - Use an approximate FAISS index (IVF/HNSW).
    - Update the graph incrementally instead of rebuilding it.
    - Batch GNN inference and run it on a GPU.
    - Use a cross-encoder reranker before the graph.
    - Use a larger generator or an LLM backend.
    - Move from SQLite to Postgres for multi-user use.

### Limitations and future work (have these ready)
- Small CPU models; comparison and yes/no questions are weak; no OCR; string-based entity linking; confidence is
  not calibrated; Wikipedia mode was not tested live; the GNN is statistically tied with the heuristic on 300
  questions.
- Next steps: question decomposition for comparison questions, a cross-encoder reranker, Wikidata entity linking,
  more GNN training data and joint training with the retriever, confidence calibration, OCR.

---

## 6. If something goes wrong

| Symptom | Fix |
|---|---|
| The health indicator in the top bar shows the backend as down | The backend is not running, or is still warming up (about 5 s). Check `/api/health`. |
| HTTP 409 "no documents" on ask | Run `python scripts\seed_demo.py` (or upload the sample files). |
| The GNN option shows a warning | No compatible checkpoint was found: the app falls back to the graph heuristic and says so. Check `/api/gnn/status`. |
| The first answer is slow | Flan-T5 loads on first use; ask the warm-up question first. |
| Wikipedia returns 503/403 | The User-Agent is not configured, or there is no internet (expected; see §0). |
| Port 8000 or 5173 is in use | Close the old server window, or run `netstat -ano \| findstr :8000` and end that process. |
| Anything else | Show the screenshots in `docs/screenshots/` and the API docs at `/docs`. |
