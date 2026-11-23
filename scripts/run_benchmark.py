"""Reproducible benchmark: stage latencies + retrieval/answer metrics for all strategies.

    python scripts/run_benchmark.py                     # demo set + HotpotQA test sample (if downloaded)
    python scripts/run_benchmark.py --hotpot-limit 100  # fewer HotpotQA questions (faster)
    python scripts/run_benchmark.py --skip-eval         # latency benchmark only

Writes (all values measured during this run):
    results/latency_results.json     stage-wise latency / memory / graph sizes
    results/retrieval_metrics.json   per-dataset, per-strategy metric summaries
    results/evaluation_results.csv   per-question, per-strategy rows
    results/training_history.json    copy of the GNN training history (if trained)
    results/figures/*.png            charts generated from the files above
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.container import get_container  # noqa: E402
from app.core.seeds import set_seed  # noqa: E402
from app.evaluation.benchmark import BenchmarkRunner  # noqa: E402
from app.evaluation.evaluator import CSV_COLUMNS, Evaluator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hotpot-limit", type=int, default=300)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    c = get_container()
    set_seed(c.settings.seed)
    results = c.settings.results_dir
    results.mkdir(parents=True, exist_ok=True)

    demo = json.loads((c.settings.samples_dir / "datasets" / "demo_qa.json").read_text(encoding="utf-8"))
    questions = [q["question"] for q in demo["questions"]]
    print(f"[1/3] Stage latency benchmark ({len(questions)} questions x {args.repeats} repeats)...")
    t = time.perf_counter()
    latency = BenchmarkRunner(c).run(questions, repeats=args.repeats)
    (results / "latency_results.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
    print(f"      done in {time.perf_counter() - t:.1f}s -> results/latency_results.json")

    if not args.skip_eval:
        evaluator = Evaluator(c)
        metrics, rows_csv = {}, io.StringIO()
        writer = csv.DictWriter(rows_csv, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        datasets = [("demo", None)]
        if (c.settings.samples_dir / "datasets" / "hotpotqa" / "test.json").exists():
            datasets.append(("hotpotqa", args.hotpot_limit))
        else:
            print("      HotpotQA sample not found (scripts/download_datasets.py) - evaluating the demo set only")
        for i, (name, limit) in enumerate(datasets, start=2):
            print(f"[{i}/3] Evaluating dataset '{name}' (limit={limit}) on all strategies...")
            t = time.perf_counter()
            out = evaluator.run(dataset=name, limit=limit, persist=True,
                                progress=lambda f, m: print(f"      {m}", end="\r"))
            print(f"\n      done in {time.perf_counter() - t:.1f}s (run {out['run_id']})")
            metrics[name] = {"run_id": out["run_id"], "config": out["config"], **out["summary"]}
            for r in out["rows"]:
                d = r.get("details") or {}
                writer.writerow({**r, "run_id": out["run_id"], "dataset": name,
                                 "generator_backend": d.get("generator_backend"), "confidence": d.get("confidence")})
            for s, m in out["summary"]["by_strategy"].items():
                print(f"      {s:7s} EM={m['exact_match']} F1={m['f1']} P@2={m['precision_at_k']} "
                      f"R@2={m['recall_at_k']} MRR={m['mrr']} EvR={m['evidence_recall']}")
        (results / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (results / "evaluation_results.csv").write_text(rows_csv.getvalue(), encoding="utf-8")
    history = c.settings.gnn_dir / "training_history.json"
    if history.exists():
        shutil.copyfile(history, results / "training_history.json")
    if not args.no_figures:
        sys.path.insert(0, str(ROOT / "scripts"))
        from make_figures import main as make_figures  # noqa: E402

        make_figures()
    print("Results written to", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
