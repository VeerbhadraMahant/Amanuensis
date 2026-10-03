import json

from tests.helpers import make_client

from amanuensis.registry import models, results
from amanuensis.store import db, review, sessions
from amanuensis.ui.app import create_app
from tests.test_loop import BETTER, env, pipeline, status  # noqa: F401  (fixture)

TAGS = ["en", "de", "hi", "mr"]


def test_history_reads_reports_and_tolerates_missing_or_broken_ones(tmp_path):
    conn = db.connect(":memory:")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"overall": {"normalized_wer": 0.2}, "per_language": {"en": {"normalized_wer": 0.1}},
                                "second_set": {"overall": {"normalized_wer": 0.3}}}))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    for v, path in (("m1", good), ("m2", bad), ("m3", None)):
        models.register(conn, v, "whisper-small", None, None)
        if path:
            models.set_eval_report(conn, v, str(path))
    h = {v["version"]: v for v in results.history(conn)["versions"]}
    assert h["m1"]["overall_wer"] == 0.2 and h["m1"]["per_language_wer"] == {"en": 0.1} and h["m1"]["second_set_wer"] == 0.3
    assert h["m2"]["has_report"] is False and h["m3"]["overall_wer"] is None


def test_correction_rate_by_day():
    conn = db.connect(":memory:")
    sid = sessions.start_session(conn, "m", "live")
    for status_, day in (("corrected", "2026-10-01"), ("approved_as_is", "2026-10-01"), ("approved_as_is", "2026-10-01"),
                         ("corrected", "2026-10-02"), ("rejected", "2026-10-02")):
        conn.execute("INSERT INTO utterances (session_id, audio_path, duration_s, hypothesis_raw, hypothesis_normalized,"
                     " final_text, status, created_at, reviewed_at) VALUES (?, 'x', 1, 't', 't', 't', ?, ?, ?)",
                     (sid, status_, day, day + "T10:00:00"))
    days = results.history(conn)["correction_rate_by_day"]
    assert [(d["day"], round(d["correction_rate"], 3)) for d in days] == [("2026-10-01", 0.333), ("2026-10-02", 1.0)]


def test_loop_run_records_eval_report_paths_and_second_set(env):  # noqa: F811
    ctx, fakes, ev, conn = env
    ctx.eval_cfg["second_set"] = dict(ctx.eval_cfg)  # same files reused as a stand-in second set
    r = pipeline.run_cycle(ctx, "manual")
    assert r.outcome == "awaiting_owner_approval"
    version = next(iter(status(conn)))
    path = models.get(conn, version)["eval_report_path"]
    saved = json.loads(open(path, encoding="utf-8").read())
    assert saved["overall"] == BETTER["overall"] and "second_set" in saved
    assert any(c[0] == "eval" for c in fakes.calls) and len([c for c in fakes.calls if c[0] == "eval"]) == 4  # 2 models x 2 sets


def test_results_endpoint(tmp_path):
    conn = db.connect(":memory:", check_same_thread=False)
    models.register(conn, "m1", "whisper-small", None, None)
    body = make_client(create_app(conn, tmp_path, TAGS, {})).get("/api/results").json()
    assert body["versions"][0]["version"] == "m1" and body["correction_rate_by_day"] == []
