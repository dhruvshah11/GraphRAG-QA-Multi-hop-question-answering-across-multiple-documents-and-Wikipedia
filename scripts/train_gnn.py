"""Train the GAT evidence reranker.

    python scripts/train_gnn.py                       # HotpotQA sample (run download_datasets.py first)
    python scripts/train_gnn.py --source demo         # tiny rule-derived demonstration set (labelled as such)
    python scripts/train_gnn.py --epochs 60 --train-limit 500

Outputs:
    models/gnn/gat_reranker.pt          best checkpoint (by validation MRR) + metadata
    models/gnn/training_history.json    per-epoch losses/metrics
    results/training_history.json       copy for the report / benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.container import get_container  # noqa: E402
from app.gnn.training_service import train_gnn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["hotpotqa", "demo"], default="hotpotqa")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-train", action="store_true",
                        help="also train on data/samples/datasets/hotpotqa/train_extra.json")
    args = parser.parse_args()
    c = get_container()
    if args.source == "hotpotqa" and not (c.settings.samples_dir / "datasets" / "hotpotqa" / "train.json").exists():
        print("HotpotQA sample not found. Run: python scripts/download_datasets.py  (or use --source demo)")
        return 1
    summary = train_gnn(c, source=args.source, epochs=args.epochs, train_limit=args.train_limit,
                        val_limit=args.val_limit, lr=args.lr, seed=args.seed,
                        extra_train=args.extra_train)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
