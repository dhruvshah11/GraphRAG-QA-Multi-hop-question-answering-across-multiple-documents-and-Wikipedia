# Previous runs (kept for traceability)

Numbers quoted in README §26.3, the report and the slides for superseded GNN models come from these files.
The current results in `results/` use the adopted GAT trained on 3 000 HotpotQA questions.

| Folder | What it is | Key numbers |
|---|---|---|
| `gnn_1k_questions/` | First GAT, trained on 1 000 questions; full benchmark output of that run (`retrieval_metrics.json`, `evaluation_results.csv`) and its training history | val MRR 0.928 / R@2 0.708; test (300 q) EM 0.367, P@2 0.677, evidence recall 0.875 |
| `gnn_v2_feature_experiment/` | Variant with the graph-heuristic score as a 9th input feature, 1 000 questions; rejected on validation, never evaluated on test | val MRR 0.922 / R@2 0.705; graph heuristic on validation MRR 0.934 / R@2 0.730 |
