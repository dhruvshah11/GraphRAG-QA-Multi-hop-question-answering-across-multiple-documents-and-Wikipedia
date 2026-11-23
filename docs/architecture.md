# GraphRAG-QA — Architecture

Rendered PlantUML diagrams live in [`diagrams/rendered/`](diagrams/rendered) (PNG + SVG),
sources in [`diagrams/src/`](diagrams/src). Re-render with `python scripts/render_diagrams.py`.
The Mermaid versions below render directly on GitHub.

| Diagram | File |
|---|---|
| System architecture | `architecture.png` |
| Use cases | `use_case.png` |
| Classes | `class_diagram.png` |
| Sequences A–F | `seq_document_upload.png`, `seq_multidoc_qa.png`, `seq_open_domain.png`, `seq_graph_construction.png`, `seq_gnn_reranking.png`, `seq_answer_generation.png` |
| Activity (QA workflow) | `activity_qa.png` |
| Components | `component.png` |
| Deployment | `deployment.png` |
| Data flow (level 0 / 1) | `dfd_level0.png`, `dfd_level1.png` |
| Entity–relationship | `er_diagram.png` |

## Layers

```mermaid
flowchart TB
  subgraph UI["React SPA (Vite, TypeScript, Tailwind, React Flow, Recharts)"]
    D[Dashboard] --- W[Documents] --- Q[Question Answering] --- G[Graph Explorer] --- A[Analytics] --- E[Evaluation Lab] --- S[Settings]
  end
  UI -->|HTTP JSON /api| API["FastAPI + APIController<br/>Pydantic schemas, CORS, background jobs"]
  API --> ING["Ingestion: PyMuPDF / python-docx<br/>normalisation"]
  ING --> CH["Chunker 400/80 tokens"]
  CH --> NER["spaCy NER + verb relations"]
  CH --> EMB["MiniLM-L6-v2 embeddings"]
  EMB --> FAISS[(FAISS IndexIDMap2)]
  CH --> BM25[(BM25)]
  NER --> REG["EntityRegistry<br/>duplicate resolution"]
  REG --> KG[(Evidence graph<br/>NetworkX)]
  API --> PIPE["QAPipeline / GraphReasoner"]
  PIPE --> FAISS & BM25 & KG
  PIPE --> WIKI["OpenDomainRetriever"] --> MW((Wikipedia API))
  PIPE --> TRAV["GraphTraversal<br/>evidence chains"]
  PIPE --> RR["Rerankers: PPR heuristic | GAT (PyG)"]
  PIPE --> GEN["AnswerGenerator<br/>Flan-T5 / reader / extractive"]
  GEN --> CIT["CitationBuilder + confidence"]
  API --> DB[(SQLite)]
```

## Question answering sequence

```mermaid
sequenceDiagram
  actor U as User
  participant UI as React UI
  participant API as FastAPI
  participant P as QAPipeline
  participant R as GraphReasoner
  participant G as AnswerGenerator
  U->>UI: question + strategy
  UI->>API: POST /api/qa/ask
  API->>P: ask()
  P->>R: analyze(corpus, question)
  R->>R: FAISS + BM25 + hybrid scores
  R->>R: question subgraph, chains, PPR, GNN
  R-->>P: rankings for 5 strategies + chains
  P->>G: evidence (chain first) → grounded answer
  P->>P: citations, confidence, insufficient-evidence check
  P-->>API: answer, reasoning, chain, citations, scores, timings
  API-->>UI: JSON
```

## Storage

```mermaid
erDiagram
  documents ||--o{ pages : has
  documents ||--o{ chunks : split_into
  chunks ||--o{ entity_mentions : contains
  entities ||..o{ entity_mentions : resolves
  chunks ||--o{ relations : states
  queries ||--|| answers : produces
  answers ||--o{ citations : cites
  evaluation_runs ||--o{ evaluation_results : records
```

Files on disk: `data/graphrag_qa.sqlite3`, `data/indexes/faiss.index` (+ `.meta.json`
with model id and dimension), `data/graphs/corpus_graph.json|graphml`,
`data/graphs/queries/<query_id>.json` (subgraph + evidence-chain views with node
positions), `models/gnn/gat_reranker.pt` + `training_history.json`.

## Key design decisions

* **One analysis, five strategies.** `GraphReasoner.analyze` computes every score once; FAISS, BM25, hybrid, graph-heuristic and GNN rankings are therefore compared on identical candidates (UI analytics and evaluation).
* **Snapshots.** `CorpusManager` rebuilds an immutable `Corpus` snapshot (registry, BM25, graph) after every document change; queries never see half-updated indexes. Index mutations run in a single-worker job queue.
* **Honest model status.** The GNN is only used when a compatible trained checkpoint exists; otherwise the response says that the graph heuristic was used. Model fallbacks (hashing embedder, regex NER) are reported as *degraded* in `/api/health`.
* **Grounding first.** Generated text never reaches the user unless it is found in the evidence and backed by validated citations; otherwise the insufficient-evidence message is returned with the reasons.
* **Composition root.** `core/container.py` creates each service once (models load once per process); the API, scripts and tests share it.
