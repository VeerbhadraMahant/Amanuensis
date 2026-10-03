import pytest
from fastapi.testclient import TestClient

from amanuensis.eval.promotion import PromotionDecision
from amanuensis.registry import approvals, models
from amanuensis.store import db
from amanuensis.ui.app import create_app

PASS = PromotionDecision(True, ["all promotion gates passed"], {"en": -0.01})


@pytest.fixture
def conn():
    c = db.connect(":memory:", check_same_thread=False)
    models.register(c, "m1", "whisper-small", None, "cfg.yaml")
    return c


def test_only_a_passing_decision_can_be_queued(conn):
    with pytest.raises(PermissionError):
        approvals.request(conn, "m1", PromotionDecision(False, ["no"], {}))
    with pytest.raises(KeyError):
        approvals.request(conn, "ghost", PASS)


def test_approve_promotes_and_decline_rejects(conn):
    models.register(conn, "m2", "whisper-small", None, "cfg.yaml")
    a, b = approvals.request(conn, "m1", PASS), approvals.request(conn, "m2", PASS)
    approvals.approve(conn, a)
    approvals.decline(conn, b)
    assert models.champion(conn)["version"] == "m1" and models.get(conn, "m2")["status"] == "rejected"
    assert approvals.pending(conn) == []
    with pytest.raises(ValueError):
        approvals.approve(conn, a)  # already decided


def test_ui_endpoints(tmp_path, conn):
    approvals.request(conn, "m1", PASS)
    client = TestClient(create_app(conn, tmp_path, ["en"], {}))
    items = client.get("/api/approvals").json()
    assert len(items) == 1 and items[0]["decision"]["reasons"] == ["all promotion gates passed"]
    assert client.post(f"/api/approvals/{items[0]['id']}/approve").status_code == 200
    assert client.post(f"/api/approvals/{items[0]['id']}/approve").status_code == 422
    assert client.post("/api/approvals/99/decline").status_code == 404
    assert models.champion(conn)["version"] == "m1"
