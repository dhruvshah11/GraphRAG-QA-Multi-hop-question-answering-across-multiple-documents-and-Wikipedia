"""Local Hugging Face models used for answer generation.

* :class:`ExtractiveReader` - extractive QA (default ``deepset/tinyroberta-squad2``):
  predicts an answer *span* inside a passage plus a "no answer" score
  (SQuAD 2.0 style). A span is by construction grounded in the evidence.
* :class:`Seq2SeqGenerator` - generative QA (default ``google/flan-t5-base``):
  produces a short free-form answer (used e.g. for yes/no questions). Its
  output is verified against the evidence before it is accepted.

Both load lazily, only from the local cache unless ``allow_download`` is set,
so the application never blocks on a large download during a request.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.core.logging_setup import get_logger

log = get_logger(__name__)


def hf_model_cached(model_name: str, cache_dir: Path) -> bool:
    """True if a model snapshot exists in our local cache directory."""
    folder = Path(cache_dir) / ("models--" + model_name.replace("/", "--"))
    return folder.exists() and any(folder.glob("snapshots/*/config.json"))


@dataclass
class Span:
    text: str
    context_index: int
    char_start: int
    char_end: int
    score: float  # start + end logit
    null_score: float  # CLS start + end logit
    probability: float  # softmax(span vs no-answer)


class ExtractiveReader:
    def __init__(self, model_name: str, cache_dir: Path, device: str = "cpu") -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._tok: Any = None
        self._model: Any = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None
        self.load_ms: Optional[float] = None

    @property
    def available(self) -> bool:
        return self._model is not None or hf_model_cached(self.model_name, self.cache_dir)

    def load(self, allow_download: bool = False) -> bool:
        if self._model is not None:
            return True
        if not allow_download and not hf_model_cached(self.model_name, self.cache_dir):
            self.error = "model not downloaded (run scripts/download_models.py)"
            return False
        with self._lock:
            if self._model is not None:
                return True
            start = time.perf_counter()
            try:
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer

                local = hf_model_cached(self.model_name, self.cache_dir)
                self._tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_dir),
                                                          local_files_only=local)
                self._model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_name, cache_dir=str(self.cache_dir), local_files_only=local)
                self._model.eval()
                self.load_ms = (time.perf_counter() - start) * 1000
                log.info("Loaded reader %s (%.0f ms)", self.model_name, self.load_ms)
                return True
            except Exception as exc:  # pragma: no cover - depends on environment
                self.error = str(exc)
                log.warning("Could not load reader %s: %s", self.model_name, exc)
                return False

    def answer(self, question: str, contexts: list[str], max_answer_tokens: int = 30, max_length: int = 384,
               stride: int = 128) -> list[Span]:
        """Best span for every context (sliding windows over long contexts)."""
        if not self.load():
            raise RuntimeError(self.error or "reader unavailable")
        import torch

        enc = self._tok(
            [question] * len(contexts), contexts, truncation="only_second", max_length=max_length, stride=stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt",
        )
        mapping = enc.pop("overflow_to_sample_mapping").tolist()
        offsets = enc.pop("offset_mapping").tolist()
        with self._lock, torch.no_grad():
            out = self._model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        starts = out.start_logits.numpy()
        ends = out.end_logits.numpy()
        best: dict[int, Span] = {}
        min_null: dict[int, float] = {}
        for f in range(len(mapping)):
            ctx_idx = mapping[f]
            seq_ids = enc.sequence_ids(f)
            if not any(s == 1 for s in seq_ids):
                continue
            # "no answer" score of this window = start/end logits of the [CLS] token
            null = float(starts[f][0] + ends[f][0])
            min_null[ctx_idx] = min(min_null.get(ctx_idx, float("inf")), null)
            s_logits, e_logits = starts[f], ends[f]
            top_s = [i for i in np.argsort(-s_logits)[:20] if seq_ids[i] == 1]
            top_e = [i for i in np.argsort(-e_logits)[:20] if seq_ids[i] == 1]
            for s in top_s:
                for e in top_e:
                    if e < s or e - s + 1 > max_answer_tokens:
                        continue
                    score = float(s_logits[s] + e_logits[e])
                    current = best.get(ctx_idx)
                    if current is None or score > current.score:
                        c_start, c_end = offsets[f][s][0], offsets[f][e][1]
                        best[ctx_idx] = Span(contexts[ctx_idx][c_start:c_end].strip(), ctx_idx, c_start, c_end,
                                             score, 0.0, 0.0)
        for ctx_idx, sp in best.items():
            # SQuAD 2.0 decision: compare the best span with the most confident "no answer"
            sp.null_score = min_null[ctx_idx]
            sp.probability = float(1.0 / (1.0 + np.exp(sp.null_score - sp.score)))
        return sorted(best.values(), key=lambda sp: -sp.score)


class Seq2SeqGenerator:
    def __init__(self, model_name: str, cache_dir: Path, device: str = "cpu") -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._tok: Any = None
        self._model: Any = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None
        self.load_ms: Optional[float] = None

    @property
    def available(self) -> bool:
        return self._model is not None or hf_model_cached(self.model_name, self.cache_dir)

    def load(self, allow_download: bool = False) -> bool:
        if self._model is not None:
            return True
        if not allow_download and not hf_model_cached(self.model_name, self.cache_dir):
            self.error = "model not downloaded (run scripts/download_models.py --with-generator)"
            return False
        with self._lock:
            if self._model is not None:
                return True
            start = time.perf_counter()
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                local = hf_model_cached(self.model_name, self.cache_dir)
                self._tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_dir),
                                                          local_files_only=local)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, cache_dir=str(self.cache_dir), local_files_only=local)
                self._model.eval()
                self.load_ms = (time.perf_counter() - start) * 1000
                log.info("Loaded generator %s (%.0f ms)", self.model_name, self.load_ms)
                return True
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
                log.warning("Could not load generator %s: %s", self.model_name, exc)
                return False

    def generate(self, prompt: str, max_new_tokens: int = 32, max_input_tokens: int = 512) -> tuple[str, float]:
        """Greedy decoding; returns (text, mean token probability)."""
        if not self.load():
            raise RuntimeError(self.error or "generator unavailable")
        import torch

        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
        with self._lock, torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                       output_scores=True, return_dict_in_generate=True)
        tokens = out.sequences[0]
        text = self._tok.decode(tokens, skip_special_tokens=True).strip()
        probs = []
        for step, logits in enumerate(out.scores):
            token_id = int(tokens[step + 1]) if step + 1 < len(tokens) else None
            if token_id is None:
                break
            p = torch.softmax(logits[0], dim=-1)[token_id].item()
            probs.append(p)
        return text, float(np.mean(probs)) if probs else 0.0
