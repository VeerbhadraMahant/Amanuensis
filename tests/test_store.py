import json
import sqlite3
import wave

import numpy as np
import pytest

from amanuensis.store import db, review, sessions
from amanuensis.text.normalizer import canonicalize
from amanuensis.text.spelling import check_spelling

TAGS = ["en", "de", "hi", "mr"]
# SYNTHETIC variants, not owner spellings (see docs/romanization.md stub).
VARIANTS = {"kyaa": "kya", "he": "hai"}


@pytest.fixture
def conn():
    return db.connect(":memory:")


@pytest.fixture
def utt(conn, tmp_path):
    sid = sessions.start_session(conn, "base:small", "live")
    ids = [
        sessions.log_utterance(conn, tmp_path, sid, np.zeros(16000 * (i + 1), np.float32), f"raw {i}", f"norm {i}", 900.0)
        for i in range(3)
    ]
    return ids, tmp_path


def test_migrations_create_all_design_tables_and_are_idempotent(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "utterances", "lexicon", "dataset_versions", "model_versions", "loop_runs"} <= names
    db.migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)


def test_status_check_constraint(conn, utt):
    ids, _ = utt
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE utterances SET status = 'trained_on_raw' WHERE id = ?", (ids[0],))


def test_log_utterance_writes_16k_mono_wav_and_raw_row(conn, utt):
    ids, audio_dir = utt
    row = review.get_utterance(conn, ids[1])
    assert row["status"] == "raw" and row["final_text"] is None and row["duration_s"] == 2.0
    with wave.open(str(audio_dir / row["audio_path"])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 32000)


def test_save_correction_sets_status_final_text_and_tags(conn, utt):
    ids, _ = utt
    review.save_correction(conn, ids[0], "  fixed text ", ["hi", "en"], TAGS)
    r = review.get_utterance(conn, ids[0])
    assert (r["status"], r["final_text"], json.loads(r["language_tags"])) == ("corrected", "fixed text", ["en", "hi"])
    assert r["reviewed_at"]


def test_approve_uses_normalized_hypothesis(conn, utt):
    ids, _ = utt
    review.approve_as_is(conn, ids[0], ["en"], TAGS)
    r = review.get_utterance(conn, ids[0])
    assert (r["status"], r["final_text"]) == ("approved_as_is", "norm 0")


def test_reject_has_no_final_text(conn, utt):
    ids, _ = utt
    review.reject(conn, ids[0])
    r = review.get_utterance(conn, ids[0])
    assert (r["status"], r["final_text"]) == ("rejected", None)


def test_empty_correction_unknown_tag_and_empty_approval_are_refused(conn, utt):
    ids, tmp = utt
    with pytest.raises(ValueError):
        review.save_correction(conn, ids[0], "   ", ["en"], TAGS)
    with pytest.raises(ValueError):
        review.save_correction(conn, ids[0], "x", ["klingon"], TAGS)
    sid = sessions.start_session(conn, "m", "live")
    empty = sessions.log_utterance(conn, tmp, sid, np.zeros(160, np.float32), "", "", None)
    with pytest.raises(ValueError):
        review.approve_as_is(conn, empty, ["en"], TAGS)
    assert review.get_utterance(conn, ids[0])["status"] == "raw"  # refused edits change nothing


def test_unknown_utterance_raises(conn):
    with pytest.raises(KeyError):
        review.reject(conn, 999)


def test_list_is_oldest_first_and_filters_by_status(conn, utt):
    ids, _ = utt
    review.reject(conn, ids[0])
    assert [u["id"] for u in review.list_utterances(conn, "raw")] == ids[1:]
    assert [u["id"] for u in review.list_utterances(conn, None)] == ids


def test_stats(conn, utt):
    ids, _ = utt
    review.save_correction(conn, ids[0], "a", ["en"], TAGS)  # 1 s
    review.approve_as_is(conn, ids[1], ["en", "hi"], TAGS)  # 2 s
    review.reject(conn, ids[2])
    s = review.stats(conn)
    assert s["counts"] == {"raw": 0, "corrected": 1, "approved_as_is": 1, "rejected": 1}
    assert s["correction_rate"] == 0.5
    assert s["reviewed_hours_by_language"]["en"] == pytest.approx(3 / 3600)
    assert s["reviewed_hours_by_language"]["hi"] == pytest.approx(2 / 3600)
    assert s["mean_seconds_per_review"] is not None


def test_stats_empty(conn):
    s = review.stats(conn)
    assert s["correction_rate"] is None and s["mean_seconds_per_review"] is None


def test_canonicalize_keeps_punctuation_and_applies_casing():
    assert canonicalize("Kyaa kar raha he?", VARIANTS) == "kya kar raha hai?"
    assert canonicalize("i am at pccoe, pune.", None, ["PCCOE"]) == "i am at PCCOE, pune."
    assert canonicalize("Unchanged, text.") == "Unchanged, text."


def test_spelling_flags_only_known_variants_with_offsets():
    text = "Kyaa kar raha hai"
    flags = check_spelling(text, VARIANTS)
    assert [(f.word, f.suggestion, text[f.start : f.end]) for f in flags] == [("Kyaa", "kya", "Kyaa")]
    assert check_spelling("kya hai zzz", VARIANTS) == []
    assert check_spelling("anything", {}) == []


def test_rapid_logging_never_overwrites_audio(conn, tmp_path):
    sid = sessions.start_session(conn, "m", "live")
    ids = [sessions.log_utterance(conn, tmp_path, sid, np.full(160, 0.1 * i, np.float32), "r", "n", None) for i in range(1, 6)]
    paths = {review.get_utterance(conn, i)["audio_path"] for i in ids}
    assert len(paths) == 5 and len(list(tmp_path.rglob("*.wav"))) == 5
