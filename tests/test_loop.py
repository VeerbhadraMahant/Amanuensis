"""Loop tests with fake trainer/converter/evaluator: the control flow, gates and failure handling are real."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from amanuensis import process_lock
from amanuensis.config import PathsConfig
from amanuensis.eval import manifest as eval_manifest
from amanuensis.eval.promotion import PromotionConfigError, PromotionRules
from amanuensis.loop import pipeline, tools
from amanuensis.loop.run import BudgetExceeded, LoopConfig, LoopRun
from amanuensis.registry import approvals, models
from amanuensis.store import db, lexicon, review, sessions
from amanuensis.store.sessions import write_wav
from amanuensis.training.dataset import DatasetConfig
from amanuensis.training.guard import TrainingBlocked
from amanuensis.training.parity import ParityResult
from amanuensis.training.train_lora import load_train_config
from amanuensis.training.validators import ValidationConfig
from tests.test_dataset import TAGS, tone

RULES = PromotionRules(0.05, 0.02, 0.0, 0.5, True)  # SYNTHETIC thresholds for the gate's logic


def rep(overall, per, ta=0.8, lat=1.0):
    return {"overall": {"normalized_wer": overall, "term_accuracy": ta},
            "per_language": {k: {"normalized_wer": v, "term_accuracy": ta} for k, v in per.items()}, "latency_p50_s": lat}


BASE = rep(0.30, {"en": 0.10, "de": 0.12, "hi-en": 0.50})
BETTER = rep(0.25, {"en": 0.10, "de": 0.12, "hi-en": 0.40})
WORSE = rep(0.35, {"en": 0.12, "de": 0.12, "hi-en": 0.60})


class Fakes:
    def __init__(self, challenger_report=BETTER):
        self.calls, self.challenger_report = [], challenger_report
        self.oom_attempts = 0  # how many train attempts fail with OOM first
        self.convert_error = None
        self.train_error = None

    def trainer(self, cfg, manifest, audio_dir, run_name, lock_file, config_path):
        self.calls.append(("train", cfg.batch_size, cfg.grad_accum))
        if self.train_error:
            raise self.train_error
        if self.oom_attempts:
            self.oom_attempts -= 1
            raise RuntimeError("failed: out of memory (retry with a smaller batch_size)")
        run_dir = cfg.runs_dir / run_name
        (run_dir / "adapter").mkdir(parents=True)
        (run_dir / "train_config.yaml").write_text("x: 1")
        (run_dir / "run.json").write_text(json.dumps({"status": "ok", "steps": 3, "peak_vram_gb": 2.9,
                                                      "history": [{"epoch": 0, "step": 1, "loss": 0.8}, {"epoch": 0, "step": 3, "loss": 0.4}]}))
        return run_dir

    def converter(self, cfg, run_dir):
        self.calls.append(("convert",))
        if self.convert_error:
            raise self.convert_error
        return run_dir / "hf", run_dir / "ct2"

    def parity(self, hf, ct2, paths, lang, max_wer):
        return ParityResult(0.0, 1.0, True, len(paths))

    def evaluator(self, model, eval_cfg, sc, variants, terms, latency_clips):
        self.calls.append(("eval", str(model)))
        return copy.deepcopy(self.challenger_report if "ct2" in str(model) else BASE)  # real runs return fresh dicts


@pytest.fixture
def env(tmp_path):
    conn = db.connect(":memory:", check_same_thread=False)
    audio = tmp_path / "audio"
    sid = sessions.start_session(conn, "m", "live")
    for i in range(6):
        u = sessions.log_utterance(conn, audio, sid, tone(3.0, i), f"reviewed sentence number {i}", f"reviewed sentence number {i}", 500.0)
        review.approve_as_is(conn, u, ["en"], TAGS)
    ev = tmp_path / "eval"
    write_wav(ev / "e0.wav", tone(3.0, 999))
    (ev / "manifest.json").write_text(json.dumps([{"audio": "e0.wav", "text": "eval text here", "language": "en"}]))
    (ev / "manifest.sha256").write_text(eval_manifest.content_hash(ev / "manifest.json", ev))
    fakes = Fakes()
    import dataclasses

    from amanuensis.config import load_streaming

    ctx = tools.LoopContext(
        conn=conn, paths=PathsConfig(tmp_path / "db", audio, tmp_path / "d.lock", tmp_path / "v.yaml"),
        sc=load_streaming(), train_cfg=dataclasses.replace(load_train_config(), runs_dir=tmp_path / "runs"),
        dataset_cfg=DatasetConfig(tmp_path / "datasets", 5.0, 0.0, 1, 0.0), vcfg=ValidationConfig(0.5, 30.0, 0.01, 2.0, 30.0),
        loop_cfg=LoopConfig(200, 40, 14400, 2, "small", tmp_path / "reports", 3, 2, "sonnet", 5.0),
        eval_cfg={"manifest": str(ev / "manifest.json"), "eval_dir": str(ev), "manifest_hash_file": str(ev / "manifest.sha256")},
        variants={}, training_config_path=tmp_path / "t.yaml", rules_loader=lambda: RULES,
        trainer=fakes.trainer, converter=fakes.converter, parity=fakes.parity, evaluator=fakes.evaluator,
    )
    return ctx, fakes, ev, conn


def status(conn):
    return {r["version"]: r["status"] for r in conn.execute("SELECT version, status FROM model_versions")}


# ---------- happy path and the gate ----------

def test_good_challenger_waits_for_owner_and_loop_cannot_promote(env):
    ctx, fakes, _, conn = env
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "awaiting_owner_approval"
    assert models.champion(conn) is None  # the loop itself never promotes
    pending = approvals.pending(conn)
    assert len(pending) == 1 and list(status(conn).values()) == ["challenger"]
    text = r.report_path.read_text(encoding="utf-8")
    for section in ("## Dataset", "## Training", "## Metrics vs champion", "## Promotion decision", "## Failures and retries"):
        assert section in text
    assert "Awaiting your approval" in text and "hi-en" in text
    run_row = conn.execute("SELECT * FROM loop_runs WHERE id = ?", (r.run_id,)).fetchone()
    assert run_row["outcome"] == r.outcome and len(json.loads(run_row["steps"])) == 8  # every step is logged
    approvals.approve(conn, pending[0]["id"])  # the owner's decision is what promotes
    assert models.champion(conn)["version"] == pending[0]["model_version"]


def test_worse_model_is_rejected_and_nothing_is_queued(env):
    ctx, fakes, _, conn = env
    fakes.challenger_report = WORSE
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "rejected_by_gate" and approvals.pending(conn) == []
    assert list(status(conn).values()) == ["rejected"] and models.champion(conn) is None


def test_single_language_regression_blocks_despite_overall_win(env):
    ctx, fakes, _, conn = env
    fakes.challenger_report = rep(0.20, {"en": 0.10, "de": 0.20, "hi-en": 0.30})
    assert pipeline.run_cycle(ctx, "manual").outcome == "rejected_by_gate"


def test_unset_owner_thresholds_block_and_do_not_leave_a_candidate(env):
    ctx, fakes, _, conn = env

    def unset():
        raise PromotionConfigError("owner has not set: min_relative_wer_improvement")

    ctx.rules_loader = unset
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome.startswith("blocked: owner has not set") and list(status(conn).values()) == ["rejected"]


# ---------- trigger ----------

def test_threshold_trigger_skips_when_not_enough_new_data_but_manual_runs(env):
    ctx, fakes, _, conn = env
    r = pipeline.run_cycle(ctx, "threshold")  # 6 reviewed < 200
    assert r.outcome.startswith("skipped: only 6") and fakes.calls == []
    assert pipeline.run_cycle(ctx, "manual").outcome == "awaiting_owner_approval"


def test_loop_does_not_run_while_dictating(env):
    ctx, fakes, _, _ = env
    process_lock.acquire(ctx.paths.lock_file)
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "skipped: dictation is running" and fakes.calls == []


def test_no_reviewed_data_stops_cleanly(env):
    ctx, fakes, _, conn = env
    conn.execute("UPDATE utterances SET status = 'raw', final_text = NULL")
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "stopped: no usable reviewed data" and fakes.calls == []


# ---------- injected failures ----------

def test_out_of_memory_is_retried_with_smaller_batch(env):
    ctx, fakes, _, conn = env
    fakes.oom_attempts = 2
    r = pipeline.run_cycle(ctx, "manual")
    batch = ctx.train_cfg.batch_size
    assert [c[1:] for c in fakes.calls if c[0] == "train"] == [
        (batch, ctx.train_cfg.grad_accum), (batch // 2, ctx.train_cfg.grad_accum * 2), (batch // 4, ctx.train_cfg.grad_accum * 4)]
    assert r.outcome == "awaiting_owner_approval"
    assert "retry" in r.report_path.read_text(encoding="utf-8")


def test_persistent_out_of_memory_fails_without_a_candidate(env):
    ctx, fakes, _, conn = env
    fakes.oom_attempts = 99
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome.startswith("failed:") and "out of memory" in r.outcome
    assert sum(1 for c in fakes.calls if c[0] == "train") == ctx.loop_cfg.oom_retries + 1
    assert status(conn) == {} and approvals.pending(conn) == []


def test_conversion_failure_rejects_challenger_and_skips_eval(env):
    ctx, fakes, _, conn = env
    fakes.convert_error = RuntimeError("ct2 converter exploded")
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome.startswith("failed:") and "converter exploded" in r.outcome
    assert list(status(conn).values()) == ["rejected"] and not any(c[0] == "eval" for c in fakes.calls)
    assert "converter exploded" in r.report_path.read_text(encoding="utf-8")


def test_tampered_eval_set_aborts_before_any_data_or_training(env):
    ctx, fakes, ev, conn = env
    (ev / "e0.wav").write_bytes(b"tampered")
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "aborted: frozen eval set failed its hash check"
    assert fakes.calls == [] and conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] == 0


def test_edited_eval_manifest_text_also_aborts(env):
    ctx, fakes, ev, _ = env
    (ev / "manifest.json").write_text(json.dumps([{"audio": "e0.wav", "text": "changed transcript", "language": "en"}]))
    assert pipeline.run_cycle(ctx, "manual").outcome.startswith("aborted: frozen eval set")


def test_eval_audio_in_the_training_pool_aborts(env, tmp_path):
    ctx, fakes, ev, conn = env
    sid = sessions.start_session(conn, "m", "live")
    u = sessions.log_utterance(conn, ctx.paths.audio_dir, sid, tone(3.0, 999), "leaked eval sentence here", "leaked eval sentence here", None)
    review.approve_as_is(conn, u, ["en"], TAGS)
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome.startswith("aborted: eval leak") and not any(c[0] == "train" for c in fakes.calls)
    assert conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] == 0


def test_corrupt_and_missing_audio_are_rejected_not_fatal(env):
    ctx, fakes, _, conn = env
    rows = conn.execute("SELECT audio_path FROM utterances ORDER BY id").fetchall()
    (ctx.paths.audio_dir / rows[0]["audio_path"]).write_bytes(b"not a wav at all")
    (ctx.paths.audio_dir / rows[1]["audio_path"]).unlink()
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "awaiting_owner_approval"
    built = json.loads(next(s for s in json.loads(conn.execute("SELECT steps FROM loop_runs").fetchone()[0]) if s["name"] == "build_dataset")["result"])
    assert built["rejected_count"] == 2 and built["utterance_count"] == 4
    assert "unreadable audio" in built["rejected_reasons"]


def test_dictation_guard_blocks_training_cleanly(env):
    ctx, fakes, _, _ = env
    fakes.train_error = TrainingBlocked("dictation is running")
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "blocked: dictation is running"


# ---------- budgets ----------

def test_step_budget_stops_run_with_a_report(env):
    ctx, fakes, _, conn = env
    ctx.loop_cfg = LoopConfig(200, 3, 14400, 2, "small", ctx.loop_cfg.reports_dir, 3, 2, "sonnet", 5.0)
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "stopped: step budget of 3 reached" and r.report_path.exists()
    assert conn.execute("SELECT finished_at FROM loop_runs").fetchone()[0] is not None


def test_wall_clock_budget_stops_run(env):
    _, _, _, conn = env
    t = [0.0]
    cfg = LoopConfig(200, 40, 100, 2, "small", Path("x"), 3, 2, "sonnet", 5.0)
    run = LoopRun(conn, cfg, "manual", clock=lambda: t[0])
    run.step("quick", lambda: 1)
    t[0] = 101.0
    with pytest.raises(BudgetExceeded, match="wall-clock"):
        run.step("late", lambda: 2)


def test_failed_step_is_logged_with_its_error(env):
    _, _, _, conn = env
    cfg = LoopConfig(200, 40, 100, 2, "small", Path("x"), 3, 2, "sonnet", 5.0)
    run = LoopRun(conn, cfg, "manual")
    with pytest.raises(ValueError):
        run.step("boom", lambda: (_ for _ in ()).throw(ValueError("bad")))
    logged = json.loads(conn.execute("SELECT steps FROM loop_runs WHERE id = ?", (run.id,)).fetchone()[0])
    assert logged[0]["ok"] is False and "bad" in logged[0]["error"]


# ---------- tool least-privilege ----------

def test_tool_outputs_never_contain_audio_paths(env):
    ctx, _, _, conn = env
    ctx.state["dataset_version"] = None
    first = conn.execute("SELECT id FROM utterances ORDER BY id").fetchone()[0]
    review.save_correction(conn, first, "a corrected sentence", ["en"], TAGS)  # so recent_corrections returns something
    assert tools.recent_corrections(ctx)
    built = tools.build_dataset(ctx)
    ctx2_text = json.dumps([built, tools.recent_corrections(ctx), tools.check_trigger(ctx, True)])
    assert ".wav" not in ctx2_text and "audio" not in ctx2_text.lower().replace("unreadable audio", "")


def test_recent_corrections_are_text_only(env):
    ctx, _, _, conn = env
    u = conn.execute("SELECT id FROM utterances ORDER BY id").fetchone()[0]
    review.save_correction(conn, u, "something different", ["en"], TAGS)
    rows = tools.recent_corrections(ctx)
    assert rows and set(rows[0]) == {"id", "heard", "corrected"}


def test_lexicon_proposals_are_unapproved_and_unused(env):
    ctx, _, _, conn = env
    out = tools.propose_lexicon(ctx, "Kubernetes", ["kubernetes"], "term")
    assert out["approved"] is False and lexicon.approved_terms(conn) == []
    assert "Kubernetes" in pipeline.write_report(ctx, LoopRun(conn, ctx.loop_cfg, "x"), "t").read_text(encoding="utf-8")


def test_gate_cannot_be_bypassed_through_the_registry(env):
    ctx, fakes, _, conn = env
    fakes.challenger_report = WORSE
    pipeline.run_cycle(ctx, "manual")
    version = next(iter(status(conn)))
    from amanuensis.eval.promotion import PromotionDecision

    with pytest.raises(PermissionError):
        approvals.request(conn, version, PromotionDecision(False, ["no"], {}))
    with pytest.raises(ValueError):
        models.promote(conn, version, PromotionDecision(True, ["ok"], {}))  # already rejected, not a challenger
