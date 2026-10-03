import json

import numpy as np
import pytest

from amanuensis.store import db, review, sessions
from amanuensis.training import validators as v
from amanuensis.training.dataset import (
    DatasetConfig, EvalLeakError, build_dataset, eval_audio_hashes, load_manifest,
)

TAGS = ["en", "de", "hi", "mr"]
VCFG = v.ValidationConfig(0.5, 30.0, 0.01, 2.0, 30.0)


def tone(seconds: float, seed: int = 0, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * seconds)) * amp).clip(-0.9, 0.9).astype(np.float32)


# ---------- validators ----------

def test_duration_bounds():
    assert v.check_duration(1.0, VCFG) is None
    assert "short" in v.check_duration(0.1, VCFG)
    assert "long" in v.check_duration(31.0, VCFG)


def test_empty_text():
    assert v.check_text_nonempty("hi") is None
    assert v.check_text_nonempty("   ") and v.check_text_nonempty(None)


def test_clipping():
    assert v.check_clipping(tone(1), VCFG) is None
    assert "clipping" in v.check_clipping(np.ones(16000, np.float32), VCFG)
    assert v.check_clipping(np.zeros(0, np.float32), VCFG) == "empty audio"


def test_text_rate_catches_mismatch_both_ways():
    assert v.check_text_rate("a reasonable sentence here", 3.0, VCFG) is None
    assert "short" in v.check_text_rate("a", 10.0, VCFG)
    assert "long" in v.check_text_rate("x" * 400, 2.0, VCFG)


def test_spelling_compliance():
    assert v.check_spelling_compliance("kya hai", {"kyaa": "kya"}) is None
    assert "kyaa" in v.check_spelling_compliance("kyaa hai", {"kyaa": "kya"})


def test_validate_collects_all_problems():
    probs = v.validate("", tone(0.1), 0.1, VCFG, {})
    assert any("empty text" in p for p in probs) and any("short" in p for p in probs)
    assert v.validate("a fine sentence of words", tone(3), 3.0, VCFG, {}) == []


# ---------- dataset builder ----------

@pytest.fixture
def env(tmp_path):
    conn = db.connect(":memory:")
    audio_dir = tmp_path / "audio"
    sid = sessions.start_session(conn, "m", "live")
    cfg = DatasetConfig(tmp_path / "datasets", max_hours_per_group=5.0, val_fraction=0.0, seed=1, replay_min_fraction=0.2)
    return conn, audio_dir, sid, cfg


def add(env, text, tags, status, seed, seconds=3.0):
    conn, audio_dir, sid, _ = env
    i = sessions.log_utterance(conn, audio_dir, sid, tone(seconds, seed), text, text, 500.0)
    if status == "corrected":
        review.save_correction(conn, i, text, tags, TAGS)
    elif status == "approved_as_is":
        review.approve_as_is(conn, i, tags, TAGS)
    elif status == "rejected":
        review.reject(conn, i)
    return i


def test_only_reviewed_utterances_enter_a_manifest(env):
    """Golden rule 2: raw model output and rejected items must never train."""
    conn, audio_dir, _, cfg = env
    ok1 = add(env, "this is a corrected sentence", ["en"], "corrected", 1)
    ok2 = add(env, "this is an approved sentence", ["de"], "approved_as_is", 2)
    add(env, "this was never reviewed at all", ["en"], "raw", 3)
    add(env, "this one was rejected as noise", ["en"], "rejected", 4)
    r = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    assert sorted(e["id"] for e in load_manifest(r.manifest_path)) == [ok1, ok2]


def test_manifest_records_version_hash_lineage(env):
    conn, audio_dir, _, cfg = env
    add(env, "first reviewed utterance text", ["en"], "corrected", 1)
    r1 = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    add(env, "second reviewed utterance text", ["en"], "corrected", 2)
    r2 = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    assert (r1.version, r2.version) == ("ds-001", "ds-002")
    row = conn.execute("SELECT * FROM dataset_versions WHERE version = 'ds-002'").fetchone()
    assert row["parent_version"] == "ds-001" and row["utterance_count"] == 2
    assert row["content_hash"] == r2.content_hash and json.loads(row["hours_by_language"])["en"] > 0


def test_build_is_deterministic(env):
    conn, audio_dir, _, cfg = env
    for i in range(6):
        add(env, f"deterministic sentence number {i}", ["en"], "corrected", i)
    a = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    b = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    assert a.content_hash == b.content_hash


def test_eval_audio_aborts_the_build(env, tmp_path):
    """Golden rule 1: eval audio can never reach training, even if copied into the store."""
    conn, audio_dir, _, cfg = env
    i = add(env, "a sentence that also lives in the eval set", ["en"], "corrected", 7)
    from amanuensis.store.sessions import write_wav

    eval_root = tmp_path / "eval"
    write_wav(eval_root / "e.wav", tone(3.0, 7))
    (eval_root / "manifest.json").write_text(json.dumps([{"audio": "e.wav", "text": "x", "language": "en"}]))
    hashes = eval_audio_hashes(eval_root / "manifest.json", eval_root)
    with pytest.raises(EvalLeakError):
        build_dataset(conn, audio_dir, cfg, VCFG, {}, hashes)
    assert conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] == 0  # nothing was recorded
    assert i


def test_invalid_utterances_rejected_with_reasons(env):
    conn, audio_dir, _, cfg = env
    good = add(env, "a perfectly fine sentence here", ["en"], "corrected", 1)
    short = add(env, "too short audio", ["en"], "corrected", 2, seconds=0.2)
    spell = add(env, "kyaa hai", ["hi"], "corrected", 3)
    r = build_dataset(conn, audio_dir, cfg, VCFG, {"kyaa": "kya"}, set())
    reasons = dict(r.rejected)
    assert short in reasons and spell in reasons and good not in reasons
    assert [e["id"] for e in load_manifest(r.manifest_path)] == [good]


def test_dedup_by_audio_and_by_text(env):
    conn, audio_dir, _, cfg = env
    add(env, "the same words spoken again", ["en"], "corrected", 1)
    add(env, "the same words spoken again", ["en"], "corrected", 1)  # identical audio
    add(env, "The same words spoken again", ["en"], "corrected", 2)  # same text and duration
    add(env, "a different sentence entirely", ["en"], "corrected", 3)
    r = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    assert r.utterance_count == 2 and r.deduped == 2


def test_group_cap_and_replay_warning(env):
    conn, audio_dir, _, cfg = env
    cfg = DatasetConfig(cfg.datasets_dir, max_hours_per_group=7 / 3600, val_fraction=0.0, seed=1, replay_min_fraction=0.5)
    for i in range(4):
        add(env, f"hindi english mixed sentence {i}", ["hi", "en"], "corrected", i)  # 3 s each, cap 7 s
    add(env, "plain english sentence here", ["en"], "corrected", 9)
    r = build_dataset(conn, audio_dir, cfg, VCFG, {}, set())
    assert r.capped == 2 and r.hours_by_group["en+hi"] == pytest.approx(6 / 3600)
    assert any("replay" in w for w in r.warnings)  # en-only is 1 of 3 items < 50%


def test_validation_split_is_deterministic_and_marked(env):
    conn, audio_dir, _, cfg = env
    cfg = DatasetConfig(cfg.datasets_dir, 5.0, val_fraction=0.5, seed=1, replay_min_fraction=0.0)
    for i in range(10):
        add(env, f"split sentence number {i}", ["en"], "corrected", i)
    a = load_manifest(build_dataset(conn, audio_dir, cfg, VCFG, {}, set()).manifest_path)
    b = load_manifest(build_dataset(conn, audio_dir, cfg, VCFG, {}, set()).manifest_path)
    assert [e["split"] for e in a] == [e["split"] for e in b]
    assert {"train", "val"} == {e["split"] for e in a}
