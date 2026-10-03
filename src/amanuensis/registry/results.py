"""Data behind the results page: metrics per model version over time, and correction rate over time.

Correction rate (share of reviewed utterances that needed an edit) is the most honest everyday metric:
if the model is improving, the owner edits less (docs/systemdesign.md section 9).
"""
import json
import sqlite3
from pathlib import Path


def _read_report(path: str | None) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError):
        return None


def history(conn: sqlite3.Connection) -> dict:
    versions = []
    for m in conn.execute("SELECT * FROM model_versions ORDER BY created_at, rowid"):
        report = _read_report(m["eval_report_path"])
        versions.append({
            "version": m["version"], "status": m["status"], "created_at": m["created_at"], "has_report": report is not None,
            "overall_wer": report["overall"]["normalized_wer"] if report else None,
            "per_language_wer": {k: v["normalized_wer"] for k, v in report["per_language"].items()} if report else {},
            "second_set_wer": report["second_set"]["overall"]["normalized_wer"] if report and "second_set" in report else None,
        })
    days = [
        {"day": r["day"], "corrected": r["corrected"], "approved_as_is": r["approved"],
         "correction_rate": r["corrected"] / (r["corrected"] + r["approved"])}
        for r in conn.execute(
            "SELECT substr(reviewed_at, 1, 10) AS day, SUM(status = 'corrected') AS corrected, "
            "SUM(status = 'approved_as_is') AS approved FROM utterances "
            "WHERE status IN ('corrected', 'approved_as_is') AND reviewed_at IS NOT NULL GROUP BY day ORDER BY day"
        )
    ]
    return {"versions": versions, "correction_rate_by_day": days}
