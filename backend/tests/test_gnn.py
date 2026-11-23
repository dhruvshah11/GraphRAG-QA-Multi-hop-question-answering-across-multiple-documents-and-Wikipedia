"""GAT reranker: forward pass, training step, checkpoint loading and inference."""
from __future__ import annotations

import torch

from app.gnn.features import FEATURE_SPEC_VERSION, FeatureBuilder, feature_dim
from app.gnn.model import GATReranker
from app.gnn.ranker import GNNRanker
from app.gnn.trainer import GNNTrainer, TrainConfig, ranking_metrics, sample_training_mask


def _graph(container, corpus, question, gold=None):
    res = container.reasoner.analyze(corpus, question, top_k=4, max_hops=2, use_gnn=False)
    data, ids = FeatureBuilder(container.embedder).build(res.qgraph, corpus, res.scores.query_vec, gold)
    return res, data, ids


def test_features_and_forward_pass(container, demo_corpus):
    _, data, ids = _graph(container, demo_corpus, "Who founded DeepMind?")
    assert data.x.shape == (len(ids), feature_dim(384))
    assert data.edge_attr.shape[1] == 6 and data.edge_index.shape[0] == 2
    model = GATReranker(data.x.shape[1])
    logits = model(data.x, data.edge_index, data.node_type, data.edge_attr)
    assert logits.shape == (len(ids),) and torch.isfinite(logits).all()
    logits[data.passage_mask].sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_attention_weights_available(container, demo_corpus):
    _, data, _ = _graph(container, demo_corpus, "Who founded DeepMind?")
    model = GATReranker(data.x.shape[1]).eval()
    _, (att1, att2) = model(data.x, data.edge_index, data.node_type, data.edge_attr, return_attention=True)
    assert att1[1].shape[1] == 4  # 4 attention heads


def test_negative_sampling_and_metrics(container, demo_corpus):
    gold = {demo_corpus.chunks[0].id}
    _, data, _ = _graph(container, demo_corpus, "When was DeepMind founded?", gold)
    import random

    mask = sample_training_mask(data, neg_ratio=2, rng=random.Random(0))
    assert mask[data.y > 0.5].all() and mask.sum() <= 1 + 2
    m = ranking_metrics(scores=torch.tensor([0.1, 0.9, 0.3]).numpy(), labels=torch.tensor([0.0, 1.0, 1.0]).numpy(), k=2)
    assert m["recall_at_k"] == 1.0 and m["mrr"] == 1.0


def test_untrained_ranker_is_not_used(container, tmp_path):
    ranker = GNNRanker(tmp_path / "missing.pt", container.embedder)
    assert not ranker.is_available and "No trained checkpoint" in ranker.reason


def test_training_checkpoint_loading_and_inference(container, demo_corpus, demo_questions, tmp_path):
    from app.gnn.dataset import gold_chunks_for_evidence

    graphs = []
    for qid in ("demo-01", "demo-02", "demo-03", "demo-06"):
        q = demo_questions[qid]
        gold = gold_chunks_for_evidence(demo_corpus, q["gold_evidence"], q["gold_documents"])
        graphs.append(_graph(container, demo_corpus, q["question"], gold)[1])
    ckpt = tmp_path / "gat.pt"
    trainer = GNNTrainer(TrainConfig(epochs=3, batch_size=2), ckpt, tmp_path / "history.json")
    summary = trainer.train(graphs[:3], graphs[3:], {"training_source": "unit-test",
                                                     "embedding_model": container.embedder.model_id,
                                                     "feature_spec_version": FEATURE_SPEC_VERSION})
    assert summary["epochs_run"] >= 1 and ckpt.exists() and (tmp_path / "history.json").exists()
    ranker = GNNRanker(ckpt, container.embedder)
    assert ranker.is_available and ranker.status()["meta"]["training_source"] == "unit-test"
    res = container.reasoner.analyze(demo_corpus, "Who founded DeepMind?", top_k=4, max_hops=2, use_gnn=False)
    scores = ranker.score(res.qgraph, demo_corpus, res.scores.query_vec)
    assert set(scores) == set(res.qgraph.passages) and all(0 <= v <= 1 for v in scores.values())
    # a checkpoint built for another feature specification is refused
    payload = torch.load(str(ckpt), weights_only=True)
    payload["meta"]["feature_spec_version"] = "old"
    torch.save(payload, str(tmp_path / "old.pt"))
    assert not GNNRanker(tmp_path / "old.pt", container.embedder).is_available
