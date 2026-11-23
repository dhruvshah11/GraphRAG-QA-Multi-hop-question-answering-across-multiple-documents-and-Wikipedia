"""API endpoint tests (FastAPI TestClient) and a basic end-to-end workflow."""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import DOCS, ROOT


@pytest.fixture(scope="module")
def client(container):
    from app.main import create_app

    app = create_app(skip_warmup=True)
    with TestClient(app) as c:
        yield c


def wait_job(client, job_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.5)
    raise TimeoutError(job_id)


def test_health_and_settings(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] in ("ok", "degraded")
    s = client.get("/api/settings").json()
    assert s["runtime"]["chunk_size"] == 400 and "hf_reader" in s["generator_status"]
    r = client.put("/api/settings", json={"chunk_overlap": 999})
    assert r.status_code == 400
    r = client.put("/api/settings", json={"top_k": 5})
    assert r.status_code == 200 and r.json()["changed"] == ["top_k"]
    client.put("/api/settings", json={"top_k": 8})


def test_ask_without_documents_returns_409(client):
    r = client.post("/api/qa/ask", json={"question": "Who founded DeepMind?"})
    assert r.status_code == 409


def test_end_to_end_two_document_workflow(client):
    """DEMO 1: upload two documents, process, inspect, ask a cross-document question, explore the graph."""
    files = [("files", (p.name, p.read_bytes())) for p in (DOCS / "nobel_prizes_2024.pdf", DOCS / "geoffrey_hinton_profile.txt")]
    files.append(("files", ("bad.exe", b"MZ")))
    r = client.post("/api/documents/upload", files=files)
    assert r.status_code == 200
    results = r.json()["results"]
    assert "error" in results[2]
    for res in results[:2]:
        assert wait_job(client, res["job_id"])["status"] == "completed"
    listing = client.get("/api/documents").json()
    assert listing["stats"]["processed_documents"] == 2 and listing["stats"]["total_chunks"] >= 2
    pdf_id = results[0]["document"]["id"]
    detail = client.get(f"/api/documents/{pdf_id}").json()
    assert len(detail["pages"]) == 2 and detail["chunks"] and detail["entities"]

    q = "From which university did the 2024 Nobel Prize in Physics laureate from the University of Toronto receive his PhD?"
    r = client.post("/api/qa/ask", json={"question": q, "strategy": "gnn", "top_k": 4})
    assert r.status_code == 200
    out = r.json()
    assert "Edinburgh" in out["answer"]
    assert out["effective_strategy"] in ("gnn", "graph")
    assert {c["document_name"] for c in out["citations"]} >= {"geoffrey_hinton_profile.txt"}
    assert len({s["document_name"] for s in out["evidence_chain"]}) == 2  # cross-document chain
    assert client.get(f"/api/qa/{out['query_id']}").json()["answer"] == out["answer"]

    g = client.post("/api/graph/query", json={"query_id": out["query_id"], "view": "chain"}).json()
    assert any(n["highlighted"] for n in g["nodes"]) and all("position" in n for n in g["nodes"])
    full = client.get("/api/graph", params={"limit": 50}).json()
    assert full["nodes"] and full["full_stats"]["nodes"] >= len(full["nodes"])
    entity = next(n["id"] for n in full["nodes"] if n["type"] == "entity")
    nb = client.get(f"/api/graph/node/{entity}/neighbors")
    assert nb.status_code == 200 and len(nb.json()["nodes"]) > 1

    ov = client.get("/api/analytics/overview").json()
    assert ov["counts"]["questions_answered"] >= 1 and len(ov["pipeline"]) == 9
    ret = client.get("/api/analytics/retrieval", params={"query_id": out["query_id"]}).json()
    assert ret["score_table"] and set(ret["strategy_rankings"]) >= {"faiss", "bm25", "hybrid", "graph"}
    perf = client.get("/api/analytics/performance").json()
    assert any(s["stage"] == "graph_construction" for s in perf["query_stages"])

    r = client.delete(f"/api/documents/{pdf_id}")
    assert r.status_code == 200 and client.get(f"/api/documents/{pdf_id}").status_code == 404


def test_evaluation_run_and_csv(client):
    r = client.post("/api/evaluation/run", json={"dataset": "demo", "strategies": ["hybrid", "graph"], "limit": 3})
    body = r.json()
    job = wait_job(client, body["job_id"], timeout=300)
    assert job["status"] == "completed", job
    res = client.get("/api/evaluation/results", params={"run_id": body["run_id"]}).json()
    assert len(res["rows"]) == 6 and set(res["run"]["summary"]["by_strategy"]) == {"hybrid", "graph"}
    csv = client.get(f"/api/evaluation/results/{body['run_id']}/csv")
    assert csv.status_code == 200 and csv.text.startswith("run_id,dataset,question_id")


def test_validation_and_errors(client):
    assert client.post("/api/qa/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/qa/ask", json={"question": "Who?", "strategy": "magic"}).status_code == 422
    assert client.get("/api/documents/doc_missing").status_code == 404
    assert client.get("/api/evaluation/results", params={"run_id": "nope"}).status_code == 404
    assert client.get("/api/gnn/status").json()["feature_spec_version"]


def test_cors_allows_frontend_origin(client):
    r = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_frontend_api_contract(client):
    """Frontend integration: every endpoint the React client calls exists in the OpenAPI schema."""
    api_ts = ROOT / "frontend" / "src" / "services" / "api.ts"
    if not api_ts.exists():
        pytest.skip("frontend not built yet")
    schema_paths = list(client.get("/openapi.json").json()["paths"])
    patterns = [re.compile("^" + re.sub(r"\{[^}]+\}", r"[^/]+", p) + "$") for p in schema_paths]
    source = api_ts.read_text(encoding="utf-8")
    called = set(re.findall(r"`(/api/[^`]*)`", source)) | set(re.findall(r"['\"](/api/[^'\"]*)['\"]", source))
    assert called, "no API paths found in api.ts"
    missing = []
    for path in called:
        concrete = re.sub(r"/\$\{[^}]*\}", "/x", path)  # path parameters
        concrete = concrete.split("${")[0]  # appended query-string helpers such as ${qs({...})}
        concrete = concrete.split("?")[0].rstrip("/")
        if not any(p.match(concrete) for p in patterns):
            missing.append(path)
    assert not missing, f"frontend calls unknown endpoints: {missing}"
