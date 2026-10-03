import os

import pytest

from amanuensis import process_lock
from amanuensis.eval.promotion import (
    PromotionConfigError, PromotionDecision, PromotionRules, decide, load_rules,
)
from amanuensis.registry import models
from amanuensis.store import db
from amanuensis.training.guard import TrainingBlocked, ensure_can_train

# SYNTHETIC thresholds for testing the gate's logic. The real values are the owner's to set.
RULES = PromotionRules(0.05, 0.02, 0.0, 0.2, True, {"de": 0.01})


def report(overall, per, ta=0.8, lat=1.0):
    return {"overall": {"normalized_wer": overall, "term_accuracy": ta},
            "per_language": {k: {"normalized_wer": v, "term_accuracy": ta} for k, v in per.items()}, "latency_p50_s": lat}


CHAMP = report(0.30, {"en": 0.10, "de": 0.12, "hi-en": 0.50})


def test_good_challenger_passes():
    d = decide(CHAMP, report(0.25, {"en": 0.10, "de": 0.12, "hi-en": 0.40}), RULES, True)
    assert d.passed and d.reasons == ["all promotion gates passed"]
    assert d.deltas["hi-en"] == pytest.approx(-0.10)


def test_insufficient_overall_improvement_fails():
    d = decide(CHAMP, report(0.29, {"en": 0.10, "de": 0.12, "hi-en": 0.48}), RULES, True)
    assert not d.passed and "does not beat champion" in d.reasons[0]


def test_one_language_regression_blocks_even_with_big_overall_win():
    d = decide(CHAMP, report(0.20, {"en": 0.10, "de": 0.14, "hi-en": 0.30}), RULES, True)  # de +0.02 > override 0.01
    assert not d.passed and any("'de' regressed" in r for r in d.reasons)


def test_language_within_tolerance_is_allowed():
    d = decide(CHAMP, report(0.20, {"en": 0.11, "de": 0.12, "hi-en": 0.30}), RULES, True)  # en +0.01 <= 0.02
    assert d.passed


def test_missing_language_slice_fails():
    d = decide(CHAMP, report(0.20, {"en": 0.10, "hi-en": 0.30}), RULES, True)
    assert not d.passed and any("no 'de' slice" in r for r in d.reasons)


def test_term_accuracy_drop_latency_and_parity_each_block():
    base = {"en": 0.10, "de": 0.12, "hi-en": 0.40}
    assert not decide(CHAMP, report(0.25, base, ta=0.7), RULES, True).passed
    assert not decide(CHAMP, report(0.25, base, lat=1.5), RULES, True).passed
    assert not decide(CHAMP, report(0.25, base), RULES, False).passed


def test_all_failures_are_reported_together():
    d = decide(CHAMP, report(0.31, {"en": 0.2, "de": 0.2, "hi-en": 0.6}, ta=0.1, lat=9), RULES, False)
    assert len(d.reasons) >= 5


def test_rules_refuse_to_load_until_owner_fills_them(tmp_path):
    with pytest.raises(PromotionConfigError):
        load_rules(tmp_path / "missing.yaml")
    f = tmp_path / "p.yaml"
    f.write_text("min_relative_wer_improvement: null\nmax_language_regression_abs: 0.01\n")
    with pytest.raises(PromotionConfigError, match="min_relative_wer_improvement"):
        load_rules(f)
    f.write_text("min_relative_wer_improvement: 0.05\nmax_language_regression_abs: 0.01\nmax_term_accuracy_drop: 0\n"
                 "max_latency_p50_worsening_s: 0.2\nrequire_parity: true\nlanguage_tolerance_overrides: {de: 0.005}\n")
    assert load_rules(f).language_tolerance_overrides == {"de": 0.005}


def test_shipped_promotion_yaml_is_an_unfilled_template():
    with pytest.raises(PromotionConfigError):
        load_rules()  # configs/promotion.yaml must stay null until the owner decides


# ---------- registry ----------

@pytest.fixture
def conn():
    return db.connect(":memory:")


PASS = PromotionDecision(True, ["ok"], {})
FAIL = PromotionDecision(False, ["regressed"], {})


def test_promotion_requires_a_passing_decision(conn):
    models.register(conn, "m1", "whisper-small", None, "cfg.yaml")
    with pytest.raises(PermissionError):
        models.promote(conn, "m1", FAIL)
    assert models.champion(conn) is None and models.get(conn, "m1")["status"] == "challenger"


def test_promote_archives_previous_champion_and_keeps_exactly_one(conn):
    for v in ("m1", "m2"):
        models.register(conn, v, "whisper-small", None, "cfg.yaml")
    models.promote(conn, "m1", PASS)
    models.promote(conn, "m2", PASS)
    assert models.champion(conn)["version"] == "m2" and models.get(conn, "m1")["status"] == "archived"
    assert conn.execute("SELECT COUNT(*) FROM model_versions WHERE status='champion'").fetchone()[0] == 1


def test_only_challengers_can_be_promoted_or_rejected(conn):
    models.register(conn, "m1", "b", None, None)
    models.reject(conn, "m1")
    with pytest.raises(ValueError):
        models.promote(conn, "m1", PASS)
    with pytest.raises(KeyError):
        models.promote(conn, "nope", PASS)


def test_lineage_walks_dataset_parents(conn):
    for v, parent in (("ds-001", None), ("ds-002", "ds-001")):
        conn.execute("INSERT INTO dataset_versions VALUES (?, 'now', 'm.jsonl', 1, '{}', 'h', ?)", (v, parent))
    models.register(conn, "m1", "whisper-small", "ds-002", "cfg.yaml")
    models.set_eval_report(conn, "m1", "report.json")
    lin = models.lineage(conn, "m1")
    assert [d["version"] for d in lin["datasets"]] == ["ds-002", "ds-001"]
    assert lin["train_config"] == "cfg.yaml" and lin["eval_report"] == "report.json"


# ---------- guard ----------

def test_training_blocked_while_dictation_runs(tmp_path):
    lock = tmp_path / "d.lock"
    process_lock.acquire(lock)
    with pytest.raises(TrainingBlocked, match="dictation"):
        ensure_can_train(lock, 4.0, free_vram=lambda: 8.0)
    process_lock.release(lock)
    ensure_can_train(lock, 4.0, free_vram=lambda: 8.0)


def test_stale_lock_from_a_dead_process_does_not_block(tmp_path):
    lock = tmp_path / "d.lock"
    lock.write_text("999999999")
    assert not process_lock.is_active(lock)
    lock.write_text("not a pid")
    assert not process_lock.is_active(lock)
    ensure_can_train(lock, 4.0, free_vram=lambda: 8.0)


def test_training_blocked_when_vram_is_short(tmp_path):
    with pytest.raises(TrainingBlocked, match="VRAM"):
        ensure_can_train(tmp_path / "none.lock", 6.0, free_vram=lambda: 3.2)


def test_lock_records_this_pid(tmp_path):
    process_lock.acquire(tmp_path / "x.lock")
    assert (tmp_path / "x.lock").read_text() == str(os.getpid())
