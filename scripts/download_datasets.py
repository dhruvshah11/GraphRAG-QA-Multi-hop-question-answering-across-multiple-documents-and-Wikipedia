"""Optional: download a small, seeded HotpotQA sample (distractor setting).

HotpotQA (Yang et al., 2018) is distributed under CC BY-SA 4.0. Rows are read
through the Hugging Face datasets-server API (100 rows per request), so only
the sampled questions are downloaded - never the 540 MB training file.

    python scripts/download_datasets.py --train 1000 --val 200 --test 300

Splits written to data/samples/datasets/hotpotqa/:
    train.json, val.json  - disjoint samples of the official *train* split
    test.json             - sample of the official *dev* (validation) split
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "datasets" / "hotpotqa"
API = "https://datasets-server.huggingface.co/rows"
DATASET = "hotpotqa/hotpot_qa"
TOTAL = {"train": 90447, "validation": 7405}
HEADERS = {"User-Agent": "GraphRAG-QA/1.0 (academic NLP course project; dataset sampler)"}


def fetch_page(split: str, offset: int, client: httpx.Client) -> list[dict]:
    for attempt in range(5):
        try:
            r = client.get(API, params={"dataset": DATASET, "config": "distractor", "split": split,
                                        "offset": offset, "length": 100}, timeout=60)
            r.raise_for_status()
            return [row["row"] for row in r.json()["rows"]]
        except (httpx.HTTPError, KeyError) as exc:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} for {split}@{offset} in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {split} rows at offset {offset}")


def convert(row: dict) -> dict:
    ctx = row["context"]
    paragraphs = [{"title": t, "text": " ".join(s.strip() for s in sents)}
                  for t, sents in zip(ctx["title"], ctx["sentences"])]
    sf = row["supporting_facts"]
    return {
        "id": row["id"], "question": row["question"], "answer": row["answer"], "type": row["type"],
        "level": row["level"], "paragraphs": paragraphs,
        "supporting_titles": sorted(set(sf["title"])),
        "supporting_facts": [[t, i] for t, i in zip(sf["title"], sf["sent_id"])],
    }


def sample(split: str, n: int, rng: random.Random, client: httpx.Client, exclude: set[str]) -> list[dict]:
    pages = n // 100 + 2
    offsets = sorted(rng.sample(range(0, TOTAL[split] - 100, 100), pages))
    rows: list[dict] = []
    for off in offsets:
        print(f"  {split}: rows {off}-{off + 99}")
        for row in fetch_page(split, off, client):
            if row["id"] not in exclude and len(row["context"]["title"]) >= 2:
                rows.append(convert(row))
    rng.shuffle(rows)
    return rows[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-train", type=int, default=0,
                        help="only add N more training questions as train_extra.json (existing splits untouched, "
                             "their question ids excluded)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.extra_train:
        used: set[str] = set()
        for name in ("train", "val", "test"):
            path = OUT / f"{name}.json"
            if path.exists():
                used |= {ex["id"] for ex in json.loads(path.read_text(encoding="utf-8"))["examples"]}
        extra_rng = random.Random(args.seed + 1000)  # different offsets from the original sample
        with httpx.Client(headers=HEADERS) as client:
            rows = sample("train", args.extra_train, extra_rng, client, used)
        meta = {"source": "HotpotQA distractor setting via huggingface.co/datasets/hotpotqa/hotpot_qa",
                "license": "CC BY-SA 4.0", "citation": "Yang et al., HotpotQA, EMNLP 2018",
                "official_split": "train", "seed": args.seed + 1000, "excludes": "ids in train/val/test.json",
                "count": len(rows), "examples": rows}
        (OUT / "train_extra.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"wrote train_extra.json ({len(rows)} questions, {len(used)} existing ids excluded)")
        return 0
    with httpx.Client(headers=HEADERS) as client:
        train_val = sample("train", args.train + args.val, rng, client, set())
        test = sample("validation", args.test, rng, client, set())
    splits = {"train": train_val[: args.train], "val": train_val[args.train:], "test": test}
    for name, rows in splits.items():
        meta = {"source": "HotpotQA distractor setting via huggingface.co/datasets/hotpotqa/hotpot_qa",
                "license": "CC BY-SA 4.0", "citation": "Yang et al., HotpotQA, EMNLP 2018",
                "official_split": "validation" if name == "test" else "train", "seed": args.seed,
                "count": len(rows), "examples": rows}
        (OUT / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"wrote {name}.json ({len(rows)} questions)")
    (OUT / "LICENSE.txt").write_text(
        "HotpotQA data (c) Yang et al. 2018, licensed under CC BY-SA 4.0 "
        "(https://creativecommons.org/licenses/by-sa/4.0/). Sampled subset; paragraphs are from English Wikipedia.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
