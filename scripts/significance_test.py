"""Paired bootstrap significance test on the held-out HotpotQA evaluation.

    python scripts/significance_test.py            # reads results/evaluation_results.csv
                                                   # writes results/significance.json

For every pair of strategies and every metric it resamples the 300 test
questions with replacement (10 000 times, seed 0) and reports the mean paired
difference with a 95% percentile confidence interval. A difference is called
significant when the interval excludes 0. Also reports results per question
type (bridge / comparison).
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "evaluation_results.csv"
OUT = ROOT / "results" / "significance.json"
METRICS = ["exact_match", "f1", "precision_at_k", "recall_at_k", "mrr", "evidence_recall", "citation_precision"]
PAIRS = [("gnn", "graph"), ("gnn", "hybrid"), ("graph", "hybrid"), ("hybrid", "faiss"), ("hybrid", "bm25")]
B = 10_000


def main() -> int:
    rows = [r for r in csv.DictReader(CSV.open(encoding="utf-8")) if r["dataset"] == "hotpotqa"]
    run_id = rows[-1]["run_id"]  # latest HotpotQA run
    by: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        if r["run_id"] == run_id:
            by[r["question_id"]][r["strategy"]] = r
    strategies = sorted({s for q in by.values() for s in q})
    qids = sorted(q for q in by if all(s in by[q] for s in strategies))
    rng = random.Random(0)
    n = len(qids)
    samples = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]

    def val(q: str, s: str, m: str) -> float:
        v = by[q][s].get(m, "")
        return float(v) if v not in ("", None) else 0.0

    pairs = []
    for a, b in PAIRS:
        if a not in strategies or b not in strategies:
            continue
        for m in METRICS:
            d = [val(q, a, m) - val(q, b, m) for q in qids]
            mean = sum(d) / n
            boots = sorted(sum(d[i] for i in s) / n for s in samples)
            lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
            pairs.append({"a": a, "b": b, "metric": m, "mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                          "significant": lo > 0 or hi < 0})
    types = Counter(by[q][strategies[0]]["question_type"] for q in qids)
    per_type = {}
    for t in types:
        qs = [q for q in qids if by[q][strategies[0]]["question_type"] == t]
        per_type[t] = {"questions": len(qs), **{s: {m: round(sum(val(q, s, m) for q in qs) / len(qs), 4)
                                                     for m in ("exact_match", "f1", "precision_at_k", "evidence_recall")}
                                                 for s in strategies}}
    result = {"run_id": run_id, "questions": n, "bootstrap_samples": B, "seed": 0, "pairs": pairs, "per_question_type": per_type}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for p in pairs:
        if p["metric"] in ("exact_match", "precision_at_k", "evidence_recall"):
            print(f"{p['a']:>6} - {p['b']:<6} {p['metric']:<16} {p['mean_diff']:+.3f}  95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
                  f"  {'significant' if p['significant'] else 'n.s.'}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
