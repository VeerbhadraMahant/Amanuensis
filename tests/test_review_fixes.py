"""Regression tests for the issues found by the code-reviewer pass."""
import dataclasses
import json
import os

import numpy as np
import psutil
import pytest

from amanuensis import process_lock
from amanuensis.eval.promotion import PromotionDecision, PromotionRules, decide
from amanuensis.loop import pipeline, tools
from amanuensis.loop.run import LoopConfig
from amanuensis.registry import approvals, models
from amanuensis.store import db, review, sessions
from amanuensis.training import dataset as ds
from amanuensis.training import train_lora
from amanuensis.training.guard import TrainingBlocked
from amanuensis.ui.app import create_app
from tests.helpers import make_client
from tests.test_dataset import TAGS, VCFG, tone
from tests.test_loop import BASE, BETTER, RULES, env, rep, status  # noqa: F401  (fixture)

PASS = PromotionDecision(True, ["ok"], {})


# ---------- UI: CSRF and DNS rebinding ----------

@pytest.fixture
def conn():
    return db.connect(":memory:", check_same_thread=False)


def test_state_changing_requests_need_the_ui_header(tmp_path, conn):
    from fastapi.testclient import TestClient

    models.register(conn, "m1", "b", None, None)
    rid = approvals.request(conn, "m1", PASS)
    app = create_app(conn, tmp_path, TAGS, {})
    bare = TestClient(app, base_url="http://127.0.0.1:8765")  # what a cross-site page's form post looks like
    assert bare.post(f"/api/approvals/{rid}/approve").status_code == 403
    assert bare.post("/api/lexicon", json={"canonical": "x", "kind": "name"}).status_code == 403
    assert models.champion(conn) is None  # nothing happened
    assert bare.get("/api/approvals").status_code == 200  # reads are fine
    assert make_client(app).post(f"/api/approvals/{rid}/approve").status_code == 200


def test_foreign_host_header_is_refused(tmp_path, conn):
    from fastapi.testclient import TestClient

    app = create_app(conn, tmp_path, TAGS, {})
    evil = TestClient(app, base_url="http://evil.example:8765", headers={"X-Amanuensis-UI": "1"})
    assert evil.get("/api/approvals").status_code == 403
    for host in ("http://localhost:8765", "http://127.0.0.1:1234", "http://[::1]:8765"):
        assert TestClient(app, base_url=host).get("/api/config").status_code == 200


# ---------- locks ----------

def test_lock_roundtrip_and_format(tmp_path):
    lock = tmp_path / "d.lock"
    process_lock.acquire(lock)
    pid, _, started = lock.read_text().partition(":")
    assert int(pid) == os.getpid() and abs(float(started) - psutil.Process().create_time()) < 1
    assert process_lock.is_active(lock)
    process_lock.release(lock)
    assert not process_lock.is_active(lock)


def test_acquire_refuses_when_another_live_process_holds_it(tmp_path):
    lock = tmp_path / "d.lock"
    parent = psutil.Process(os.getppid())
    lock.write_text(f"{parent.pid}:{parent.create_time()}")
    assert process_lock.is_active(lock)
    with pytest.raises(process_lock.LockHeld):
        process_lock.acquire(lock)


def test_recycled_pid_does_not_count_as_the_owner(tmp_path):
    lock = tmp_path / "d.lock"
    lock.write_text(f"{os.getppid()}:{psutil.Process(os.getppid()).create_time() - 5000}")  # same pid, different process
    assert not process_lock.is_active(lock)
    process_lock.acquire(lock)  # so it can be taken over


def test_training_holds_its_own_lock_and_releases_it(tmp_path, monkeypatch):
    dictation = tmp_path / "dictation.lock"
    seen = {}

    def fake(cfg, manifest, audio_dir, run_name, config_path):
        seen["held"] = process_lock.is_active(process_lock.training_lock_path(dictation))
        raise RuntimeError("boom")

    monkeypatch.setattr(train_lora, "_train", fake)
    monkeypatch.setattr(train_lora, "ensure_can_train", lambda *a, **k: None)
    cfg = train_lora.load_train_config()
    with pytest.raises(RuntimeError):
        train_lora.train(cfg, tmp_path / "m", tmp_path, "r", dictation, tmp_path / "c")
    assert seen["held"] is True and not process_lock.is_active(process_lock.training_lock_path(dictation))


def test_second_trainer_is_refused_while_one_runs(tmp_path, monkeypatch):
    dictation = tmp_path / "dictation.lock"
    parent = psutil.Process(os.getppid())
    process_lock.training_lock_path(dictation).write_text(f"{parent.pid}:{parent.create_time()}")  # "another trainer"
    monkeypatch.setattr(train_lora, "ensure_can_train", lambda *a, **k: None)
    monkeypatch.setattr(train_lora, "_train", lambda *a: pytest.fail("must not start"))
    with pytest.raises(process_lock.LockHeld):
        train_lora.train(train_lora.load_train_config(), tmp_path / "m", tmp_path, "r", dictation, tmp_path / "c")


# ---------- promotion gate ----------

@pytest.mark.parametrize("challenger_lat,champion_lat", [(float("nan"), 1.0), (1.0, float("nan")), (None, 1.0), (1.0, None)])
def test_unmeasured_latency_blocks_promotion(challenger_lat, champion_lat):
    d = decide(rep(0.30, {"en": 0.1}, lat=champion_lat), rep(0.20, {"en": 0.1}, lat=challenger_lat), RULES, True)
    assert not d.passed and any("latency" in r for r in d.reasons)


def test_zero_latency_is_a_real_measurement():
    d = decide(rep(0.30, {"en": 0.1}, lat=0.0), rep(0.20, {"en": 0.1}, lat=0.0), RULES, True)
    assert d.passed


# ---------- approvals ----------

def test_duplicate_requests_for_one_model_collapse(conn):
    models.register(conn, "m1", "b", None, None)
    assert approvals.request(conn, "m1", PASS) == approvals.request(conn, "m1", PASS)
    assert len(approvals.pending(conn)) == 1


def test_stale_approval_is_refused_after_the_champion_changes(conn):
    for v in ("m1", "m2"):
        models.register(conn, v, "b", None, None)
    a, b = approvals.request(conn, "m1", PASS), approvals.request(conn, "m2", PASS)  # both compared against "no champion"
    approvals.approve(conn, a)
    with pytest.raises(ValueError, match="stale"):
        approvals.approve(conn, b)
    assert models.champion(conn)["version"] == "m1" and models.get(conn, "m2")["status"] == "challenger"


def test_failed_promotion_leaves_request_pending_and_champion_unchanged(conn):
    models.register(conn, "m1", "b", None, None)
    rid = approvals.request(conn, "m1", PASS)
    conn.execute("UPDATE model_versions SET status = 'rejected' WHERE version = 'm1'")
    conn.commit()
    with pytest.raises(ValueError):
        approvals.approve(conn, rid)
    assert [p["id"] for p in approvals.pending(conn)] == [rid] and models.champion(conn) is None


# ---------- eval leak and manifests ----------

def build_env(tmp_path):
    c = db.connect(":memory:")
    sid = sessions.start_session(c, "m", "live")
    cfg = ds.DatasetConfig(tmp_path / "datasets", 5.0, 0.0, 1, 0.0)
    return c, sid, cfg


def test_utterance_with_an_eval_sentence_is_excluded_even_with_different_audio(tmp_path):
    c, sid, cfg = build_env(tmp_path)
    leak = sessions.log_utterance(c, tmp_path / "a", sid, tone(3, 1), "The Eval Sentence, again!", "The Eval Sentence, again!", None)
    fine = sessions.log_utterance(c, tmp_path / "a", sid, tone(3, 2), "a perfectly ordinary sentence", "a perfectly ordinary sentence", None)
    for u in (leak, fine):
        review.approve_as_is(c, u, ["en"], TAGS)
    r = ds.build_dataset(c, tmp_path / "a", cfg, VCFG, {}, set(), {"the eval sentence again"})
    assert [e["id"] for e in ds.load_manifest(r.manifest_path)] == [fine]
    assert dict(r.rejected)[leak] == ["text matches an item in the frozen eval set (excluded)"]


def test_only_registered_unchanged_manifests_may_be_trained_on(tmp_path):
    c, sid, cfg = build_env(tmp_path)
    u = sessions.log_utterance(c, tmp_path / "a", sid, tone(3, 1), "some reviewed sentence here", "some reviewed sentence here", None)
    review.approve_as_is(c, u, ["en"], TAGS)
    r = ds.build_dataset(c, tmp_path / "a", cfg, VCFG, {}, set())
    ds.verify_registered(c, r.manifest_path)  # fine
    handmade = tmp_path / "evil.jsonl"
    handmade.write_text(r.manifest_path.read_text())
    with pytest.raises(ValueError, match="not a registered"):
        ds.verify_registered(c, handmade)
    r.manifest_path.write_text(r.manifest_path.read_text() + json.dumps({"id": 99}) + "\n")
    with pytest.raises(ValueError, match="changed after"):
        ds.verify_registered(c, r.manifest_path)


def test_loop_refuses_to_train_on_an_edited_manifest(env):  # noqa: F811
    ctx, fakes, _, conn = env
    tools.build_dataset(ctx)
    manifest = ctx.dataset_cfg.datasets_dir / ctx.state["dataset_version"] / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + "\n")  # any edit
    with pytest.raises(ValueError, match="changed after"):
        tools.train_challenger(ctx)
    assert not any(c[0] == "train" for c in fakes.calls)


# ---------- loop ----------

def test_decide_promotion_is_idempotent(env):  # noqa: F811
    ctx, fakes, _, conn = env
    pipeline.run_cycle(ctx, "manual")  # ends awaiting approval; state is kept after the run
    first = ctx.state["decision_result"]
    assert tools.decide_promotion(ctx) == first and len(approvals.pending(conn)) == 1


def test_budget_stop_after_training_does_not_leave_a_candidate(env):  # noqa: F811
    ctx, fakes, _, conn = env
    ctx.loop_cfg = dataclasses.replace(ctx.loop_cfg, step_budget=4)  # trigger, hash, dataset, train, then out of steps
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome.startswith("stopped: step budget") and list(status(conn).values()) == ["rejected"]


def test_budget_stop_keeps_a_challenger_that_already_waits_for_the_owner(env):  # noqa: F811
    ctx, fakes, _, conn = env
    ctx.loop_cfg = dataclasses.replace(ctx.loop_cfg, step_budget=8)  # exactly enough: decision is queued, then stop
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "awaiting_owner_approval" and list(status(conn).values()) == ["challenger"]


def test_database_uses_wal_with_a_busy_timeout(tmp_path):
    c = db.connect(tmp_path / "x.db")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_review_timestamps_have_millisecond_resolution(tmp_path):
    c = db.connect(":memory:")
    sid = sessions.start_session(c, "m", "live")
    u = sessions.log_utterance(c, tmp_path, sid, np.zeros(16000, np.float32), "a", "a", None)
    review.reject(c, u)
    assert "." in review.get_utterance(c, u)["reviewed_at"]
