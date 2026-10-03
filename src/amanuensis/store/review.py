"""Owner review of logged utterances. Only 'corrected' and 'approved_as_is' may ever train (golden rule 2)."""
import json
import sqlite3
from datetime import datetime, timezone

from amanuensis.store.sessions import tags_json

REVIEWED = ("corrected", "approved_as_is")
MAX_REVIEW_GAP_S = 120  # longer gaps between reviews are breaks, not review time


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_tags(tags: list[str], allowed: list[str]) -> None:
    bad = set(tags) - set(allowed)
    if bad:
        raise ValueError(f"unknown language tags: {sorted(bad)}")


def _row(r: sqlite3.Row) -> dict:
    return dict(r)


def list_utterances(conn: sqlite3.Connection, status: str | None = "raw", limit: int = 50, offset: int = 0) -> list[dict]:
    """Oldest first, so the owner reviews in the order they spoke."""
    where, args = ("WHERE status = ?", [status]) if status else ("", [])
    rows = conn.execute(f"SELECT * FROM utterances {where} ORDER BY id LIMIT ? OFFSET ?", [*args, limit, offset])
    return [_row(r) for r in rows]


def get_utterance(conn: sqlite3.Connection, utt_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM utterances WHERE id = ?", (utt_id,)).fetchone()
    return _row(r) if r else None


def _update(conn: sqlite3.Connection, utt_id: int, status: str, final_text: str | None, tags: list[str] | None) -> None:
    sets, args = ["status = ?", "final_text = ?", "reviewed_at = ?"], [status, final_text, _now()]
    if tags is not None:
        sets.append("language_tags = ?")
        args.append(tags_json(tags))
    cur = conn.execute(f"UPDATE utterances SET {', '.join(sets)} WHERE id = ?", [*args, utt_id])
    if cur.rowcount == 0:
        raise KeyError(utt_id)
    conn.commit()


def save_correction(conn, utt_id: int, text: str, tags: list[str], allowed_tags: list[str]) -> None:
    text = text.strip()
    if not text:
        raise ValueError("a correction cannot be empty; reject the utterance instead")
    _check_tags(tags, allowed_tags)
    _update(conn, utt_id, "corrected", text, tags)


def approve_as_is(conn, utt_id: int, tags: list[str], allowed_tags: list[str]) -> None:
    utt = get_utterance(conn, utt_id)
    if utt is None:
        raise KeyError(utt_id)
    if not utt["hypothesis_normalized"].strip():
        raise ValueError("cannot approve an empty transcript; reject it instead")
    _check_tags(tags, allowed_tags)
    _update(conn, utt_id, "approved_as_is", utt["hypothesis_normalized"], tags)


def reject(conn, utt_id: int) -> None:
    _update(conn, utt_id, "rejected", None, None)


def stats(conn: sqlite3.Connection) -> dict:
    counts = {s: n for s, n in conn.execute("SELECT status, COUNT(*) FROM utterances GROUP BY status")}
    corrected, approved = counts.get("corrected", 0), counts.get("approved_as_is", 0)
    hours: dict[str, float] = {}
    for tags, dur in conn.execute(
        "SELECT language_tags, duration_s FROM utterances WHERE status IN ('corrected', 'approved_as_is')"
    ):
        for t in json.loads(tags) or ["untagged"]:
            hours[t] = hours.get(t, 0.0) + dur / 3600
    times = [datetime.fromisoformat(r[0]) for r in conn.execute(
        "SELECT reviewed_at FROM utterances WHERE reviewed_at IS NOT NULL ORDER BY reviewed_at")]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    gaps = [g for g in gaps if g <= MAX_REVIEW_GAP_S]
    return {
        "counts": {s: counts.get(s, 0) for s in ("raw", "corrected", "approved_as_is", "rejected")},
        "reviewed_hours_by_language": hours,
        "correction_rate": corrected / (corrected + approved) if corrected + approved else None,
        "mean_seconds_per_review": sum(gaps) / len(gaps) if gaps else None,
    }
