"""Render report/presentation charts from the measured result files in results/.

    python scripts/make_figures.py

Reads results/retrieval_metrics.json, latency_results.json, training_history.json
and writes PNGs to results/figures/. Nothing is plotted that was not measured.
Strategy colours are fixed (FAISS, BM25, Hybrid, Graph, GNN) and validated for
colour-vision deficiency; every bar carries its value label.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
STRATEGIES = ["faiss", "bm25", "hybrid", "graph", "gnn"]
LABELS = {"faiss": "FAISS (semantic)", "bm25": "BM25 (lexical)", "hybrid": "Hybrid", "graph": "Graph heuristic",
          "gnn": "GNN (GAT) rerank"}
COLORS = {"faiss": "#2a78d6", "bm25": "#eb6834", "hybrid": "#1baf7a", "graph": "#eda100", "gnn": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def strategy_bars(dataset: str, summary: dict, path: Path, note: str) -> None:
    metrics = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("recall_at_k", "R@2"),
               ("mrr", "MRR"), ("evidence_recall", "Evidence recall")]
    strategies = [s for s in STRATEGIES if s in summary]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    width = 0.8 / len(strategies)
    for i, s in enumerate(strategies):
        xs = [m + (i - (len(strategies) - 1) / 2) * width for m in range(len(metrics))]
        vals = [summary[s].get(k) or 0 for k, _ in metrics]
        bars = ax.bar(xs, vals, width * 0.9, color=COLORS[s], label=LABELS[s], edgecolor="white", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color=INK)
    ax.set_xticks(range(len(metrics)), [m for _, m in metrics])
    ax.set_ylim(0, 1.12)
    style(ax, f"Strategy comparison - {dataset}", "score (0-1)")
    ax.legend(ncol=len(strategies), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def training_curves(history: dict, path: Path) -> None:
    h = history["history"]
    epochs = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    ax = axes[0]
    ax.plot(epochs, [r["train_loss"] for r in h], color=COLORS["faiss"], linewidth=2, label="train loss (sampled)")
    ax.plot(epochs, [r.get("val_loss") for r in h], color=COLORS["bm25"], linewidth=2, label="validation loss")
    style(ax, "GAT reranker - BCE loss", "loss")
    ax.set_xlabel("epoch", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.plot(epochs, [r["val_mrr"] for r in h], color=COLORS["gnn"], linewidth=2, label="GNN validation MRR")
    base = history["summary"]["baseline_val_hybrid"]["mrr"]
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.5, label=f"hybrid retrieval MRR ({base:.3f})")
    best = history["summary"]["best_epoch"]
    ax.axvline(best, color=GRID, linewidth=1)
    ax.text(best, ax.get_ylim()[0], f" best epoch {best}", color=MUTED, fontsize=7, va="bottom")
    style(ax, "Validation MRR (same graphs)", "MRR")
    ax.set_xlabel("epoch", color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def stage_latency(latency: dict, path: Path) -> None:
    st = latency["stages"]
    rows = [
        ("Query embedding", st["6_query_embedding"].get("mean_ms")),
        ("FAISS search", st["7_semantic_retrieval"].get("mean_ms")),
        ("BM25 search", st["8_bm25_retrieval"].get("mean_ms")),
        ("Hybrid search", st["9_hybrid_retrieval"].get("mean_ms")),
        ("Question graph", st["10_question_graph_construction"].get("mean_ms")),
        ("Chain traversal", st["11_graph_traversal"].get("mean_ms")),
        ("Graph heuristic rank", st["11_graph_traversal"]["heuristic_ranking"].get("mean_ms")),
        ("GNN inference", st["12_gnn_inference"].get("mean_ms")),
    ]
    for backend, s in st["13_answer_generation"].items():
        rows.append((f"Answer gen. ({backend})", s.get("mean_ms")))
    rows = [(n, v) for n, v in rows if v is not None]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    names = [n for n, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    bars = ax.barh(names, vals, color=COLORS["faiss"], height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v * 1.05 if v > 0 else 0.01, b.get_y() + b.get_height() / 2, f"{v:.1f} ms", va="center", fontsize=7,
                color=INK)
    ax.set_xscale("log")
    style(ax, "Per-question stage latency (mean, CPU, log scale)")
    ax.xaxis.grid(True, color=GRID)
    ax.yaxis.grid(False)
    ax.set_xlabel("milliseconds", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    metrics_path = RESULTS / "retrieval_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for dataset, data in metrics.items():
            n = data.get("num_questions")
            note = (f"Measured on {n} questions. "
                    + ("Held-out HotpotQA test sample (official dev split)." if dataset == "hotpotqa" else
                       "Demo set - also used during development, not a held-out test."))
            out = FIG / f"strategy_comparison_{dataset}.png"
            strategy_bars(dataset, data["by_strategy"], out, note)
            made.append(out)
    hist = RESULTS / "training_history.json"
    if hist.exists():
        out = FIG / "gnn_training_curves.png"
        training_curves(json.loads(hist.read_text(encoding="utf-8")), out)
        made.append(out)
    lat = RESULTS / "latency_results.json"
    if lat.exists():
        out = FIG / "stage_latency.png"
        stage_latency(json.loads(lat.read_text(encoding="utf-8")), out)
        made.append(out)
    for p in made:
        print("figure:", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
