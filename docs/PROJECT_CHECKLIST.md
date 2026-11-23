# GraphRAG-QA — Final Feature Verification Checklist

Status legend: ✅ verified in this build · ⚠️ works with a stated condition · ❌ not done.
"How verified" names the actual evidence (test, command or screenshot).

| # | Item | Status | How verified |
|---|---|---|---|
| 1 | Frontend starts | ✅ | `npm run dev` → http://127.0.0.1:5173 returns 200; `npm run build` + `npm run typecheck` pass with 0 errors |
| 2 | Backend starts | ✅ | `uvicorn app.main:app` → `/api/health` status ok; warm-up ≈ 5 s |
| 3 | PDF upload | ✅ | `test_api.py::test_end_to_end_two_document_workflow` (nobel_prizes_2024.pdf, 2 pages) |
| 4 | TXT upload | ✅ | same test (geoffrey_hinton_profile.txt); `test_ingestion.py` (cp1252, form feeds) |
| 5 | DOCX upload | ✅ | `test_ingestion.py::test_docx_ingestion_headings_become_sections`; seed_demo ingests deepmind_history.docx |
| 6 | Text extraction (page numbers kept) | ✅ | `test_pdf_extraction_preserves_pages` |
| 7 | Chunking (400/80, sentence-aware) | ✅ | `test_chunking.py` (8 tests) |
| 8 | Entity extraction | ✅ | `test_entities.py` (spaCy NER, normalisation, duplicate merging) |
| 9 | Embeddings generated | ✅ | `test_retrieval.py::test_embeddings_are_normalized_and_semantic` |
| 10 | FAISS retrieval (persist, add, remove) | ✅ | `test_faiss_insert_search_remove_persist` |
| 11 | BM25 retrieval | ✅ | `test_bm25_ranks_lexical_match_first` |
| 12 | Hybrid retrieval | ✅ | `test_hybrid_formula_and_retrievers` |
| 13 | Knowledge graph generated | ✅ | `test_graph.py` (nodes, typed edges, cross-document CONNECTS_TO, JSON/GraphML round trip) |
| 14 | Multi-hop traversal | ✅ | `test_question_subgraph_and_multihop_chain` (Nobel PDF → Geoffrey Hinton → profile) |
| 15 | Evidence chains ranked | ✅ | chain scores + explanations in every answer; `test_max_hops_limits_expansion` |
| 16 | GNN architecture implemented | ✅ | `backend/app/gnn/model.py` (2×GATConv, 43 210 params); `test_features_and_forward_pass` |
| 17 | GNN training pipeline works | ✅ | `scripts/train_gnn.py --extra-train` on 3 000/200 HotpotQA graphs → `models/gnn/gat_reranker.pt`, `results/training_history.json`; `test_training_checkpoint_loading_and_inference` |
| 18 | GNN inference with trained checkpoint | ✅ | `/api/gnn/status` available=true; QA responses show `effective_strategy: gnn`; 300-question evaluation includes the GNN |
| 19 | Wikipedia retrieval when internet is available | ⚠️ | Full open-domain pipeline verified by `test_open_domain.py` with a stubbed HTTP layer (two-hop bridge, section citations, cache, offline error). Live calls need `GRQA_WIKIPEDIA_USER_AGENT` with your contact details (Wikimedia policy; without it Wikipedia answers HTTP 403, which the API reports clearly) |
| 20 | Answers contain source citations | ✅ | `test_citations_point_at_exact_supporting_text`; screenshot 03 |
| 21 | Unsupported questions handled safely | ✅ | `test_insufficient_evidence` (2 cases); abstention accuracy 1.0 on the demo set |
| 22 | Interactive graph explorer | ✅ | screenshots 04–06 (chain, subgraph, full graph from real backend data) |
| 23 | Analytics use actual measurements | ✅ | metrics come from `metrics_log`, stored payloads and `results/*.json`; screenshot 07 |
| 24 | Evaluation scripts work | ✅ | `scripts/run_benchmark.py` (demo + 300 HotpotQA questions × 5 strategies); `test_evaluation_run_and_csv` |
| 25 | Automated tests execute | ✅ | `pytest`: 57 passed (`results/test_results.txt`) |
| 26 | README complete | ✅ | `README.md`: title + description followed by 30 numbered sections, covering all 32 required items (incl. measured results and real screenshots) |
| 27 | OOAD diagrams generated | ✅ | 15 PlantUML sources + PNG/SVG in `docs/diagrams/` (+ Mermaid in `docs/architecture.md`) |
| 28 | Academic report generated | ✅ | `docs/report/GraphRAG_QA_Project_Report.docx` + `.pdf` (86 pages; 9 chapters, references, appendices A–C), built by `scripts/build_report.py` from `results/*` |
| 29 | PowerPoint generated | ✅ | `docs/presentation/GraphRAG_QA_Final_Presentation.pptx` (24 slides), built by `scripts/build_presentation.py` from `results/*`; plus a 68-slide technical deck `GraphRAG_QA_Technical_Presentation.pptx` (51 main + 17 backup, incl. significance and examiner Q&A) from `scripts/build_technical_presentation.py`, rendered in PowerPoint and checked for overflow; plus the 15-slide main deck `GraphRAG_QA_15_Slides.pptx` (numbered references, speaker notes) from `scripts/build_final_presentation.py`, rendered in PowerPoint, no text overflow |
| 30 | Speaker notes generated | ✅ | `docs/presentation/Speaker_Notes.md` (24 slide sections); the same notes are embedded in all 24 slides |
| 31 | Demo dataset included | ✅ | 5 documents (2 PDF, 2 TXT, 1 DOCX) + 12 QA pairs + 5 Wikipedia questions + HotpotQA sample |
| 32 | Installation instructions verified | ✅ (Windows) | full install performed on Windows 11 with Python 3.13 / Node 24; `scripts/setup.py` verified on an empty data directory. Linux/macOS commands are written but were not executed on those systems |
| 33 | Final checklist | ✅ | this file |
| 34 | Viva / demo script | ✅ | `docs/presentation/Viva_Demo_Script.md`; every live question re-run on the app on 22 Nov 2026 (11/12 correct with the GNN; demo-10 is a documented failure case) |
| 35 | Clean submission package | ✅ | `scripts/package_submission.py` → `GraphRAG-QA_Submission.zip` (22 MB, no .venv/node_modules/models/DB). Unzipped into an empty folder and installed exactly as in README §15: venv + pip install, `download_models.py --with-generator`, `setup.py` (seeded 5 docs), `pytest` 57 passed, backend answered Q1 correctly with the GNN, `npm install` + typecheck + build passed (≈ 10 min with a warm pip cache; a first-time install downloads ≈ 4 GB) |

## Honest notes
* **GNN vs heuristics.** On the held-out HotpotQA sample the GNN (trained on 3 000 questions) has the best ranking scores (P@2 0.720 vs 0.702 heuristic, 0.678 hybrid) and the best evidence recall; the graph heuristic is marginally best on answer EM (0.377 vs 0.373). A paired bootstrap (`scripts/significance_test.py`) shows both graph strategies are significantly better than hybrid retrieval, but the GNN's lead over the heuristic is **not** significant (P@2 95% CI [−0.003, +0.042]). The earlier 1 000-question GNN did not beat hybrid ranking; both runs are reported in README §26.3.
* **Demo-set scores are optimistic**: the 12 demo questions were used while developing the pipeline (e.g. to tune context construction for Flan-T5).
* **Node.js** was not installed on the development machine; a portable Node 24 LTS was unpacked to `%LOCALAPPDATA%\Programs\node-v24.22.3-win-x64` (not added to PATH). Install Node.js LTS normally (or add that folder to PATH) to run the frontend from your own terminal.
* **Optional LLM backends** (Ollama, Claude API) are implemented but were not exercised (no model configured / no API key); the application falls back to the local models automatically.
* **PlantUML** is rendered with version 1.2023.13 because the machine has Java 8 (newer PlantUML needs Java 11).
