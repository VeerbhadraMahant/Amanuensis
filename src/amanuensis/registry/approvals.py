"""Owner approval of promotions. The gate passed; a human still presses the button (plan Phase 5)."""
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone

from amanuensis.eval.promotion import PromotionDecision
from amanuensis.registry import models


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request(conn: sqlite3.Connection, version: str, decision: PromotionDecision) -> int:
    """Queue a promotion for the owner. Only a decision that passed the gate can be queued."""
    if not decision.passed:
        raise PermissionError("only a promotion decision that passed the gate can be sent for approval")
    if models.get(conn, version) is None:
        raise KeyError(version)
    cur = conn.execute(
        "INSERT INTO promotion_requests (model_version, decision, created_at) VALUES (?, ?, ?)",
        (version, json.dumps(asdict(decision)), _now()),
    )
    conn.commit()
    return cur.lastrowid


def pending(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM promotion_requests WHERE status = 'pending' ORDER BY id").fetchall()
    return [{**dict(r), "decision": json.loads(r["decision"])} for r in rows]


def _get_pending(conn: sqlite3.Connection, request_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM promotion_requests WHERE id = ?", (request_id,)).fetchone()
    if row is None:
        raise KeyError(request_id)
    if row["status"] != "pending":
        raise ValueError(f"request {request_id} is already {row['status']}")
    return row


def approve(conn: sqlite3.Connection, request_id: int) -> None:
    row = _get_pending(conn, request_id)
    d = json.loads(row["decision"])
    models.promote(conn, row["model_version"], PromotionDecision(**d))
    conn.execute("UPDATE promotion_requests SET status = 'approved', decided_at = ? WHERE id = ?", (_now(), request_id))
    conn.commit()


def decline(conn: sqlite3.Connection, request_id: int) -> None:
    row = _get_pending(conn, request_id)
    models.reject(conn, row["model_version"])
    conn.execute("UPDATE promotion_requests SET status = 'declined', decided_at = ? WHERE id = ?", (_now(), request_id))
    conn.commit()
