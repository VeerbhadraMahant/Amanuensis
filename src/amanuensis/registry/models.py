"""Model registry: champion/challenger tracking with lineage (N4: every model traces to a dataset,
a training config and an eval report)."""
import sqlite3
from datetime import datetime, timezone

from amanuensis.eval.promotion import PromotionDecision


def register(
    conn: sqlite3.Connection,
    version: str,
    base_model: str,
    dataset_version: str | None,
    train_config_path: str | None,
    adapter_path: str | None = None,
    ct2_path: str | None = None,
    status: str = "challenger",
) -> None:
    conn.execute(
        "INSERT INTO model_versions (version, base_model, adapter_path, ct2_path, dataset_version,"
        " train_config_path, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (version, base_model, adapter_path, ct2_path, dataset_version, train_config_path, status,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()


def get(conn: sqlite3.Connection, version: str) -> dict | None:
    r = conn.execute("SELECT * FROM model_versions WHERE version = ?", (version,)).fetchone()
    return dict(r) if r else None


def champion(conn: sqlite3.Connection) -> dict | None:
    r = conn.execute("SELECT * FROM model_versions WHERE status = 'champion'").fetchone()
    return dict(r) if r else None


def set_eval_report(conn: sqlite3.Connection, version: str, path: str) -> None:
    if conn.execute("UPDATE model_versions SET eval_report_path = ? WHERE version = ?", (path, version)).rowcount == 0:
        raise KeyError(version)
    conn.commit()


def reject(conn: sqlite3.Connection, version: str) -> None:
    if conn.execute("UPDATE model_versions SET status = 'rejected' WHERE version = ? AND status = 'challenger'",
                    (version,)).rowcount == 0:
        raise ValueError(f"{version} is not a challenger")
    conn.commit()


def promote(conn: sqlite3.Connection, version: str, decision: PromotionDecision) -> None:
    """Make `version` champion. Needs a passing PromotionDecision: there is no way around the gate."""
    if not decision.passed:
        raise PermissionError(f"promotion refused: {'; '.join(decision.reasons)}")
    row = get(conn, version)
    if row is None:
        raise KeyError(version)
    if row["status"] != "challenger":
        raise ValueError(f"{version} is {row['status']}, only a challenger can be promoted")
    with conn:  # one transaction: there is never zero or two champions
        conn.execute("UPDATE model_versions SET status = 'archived' WHERE status = 'champion'")
        conn.execute("UPDATE model_versions SET status = 'champion' WHERE version = ?", (version,))


def lineage(conn: sqlite3.Connection, version: str) -> dict:
    """Model -> dataset version -> its parent chain, plus config and eval report paths."""
    m = get(conn, version)
    if m is None:
        raise KeyError(version)
    datasets, dv = [], m["dataset_version"]
    while dv:
        d = conn.execute("SELECT * FROM dataset_versions WHERE version = ?", (dv,)).fetchone()
        if d is None:
            break
        datasets.append({"version": d["version"], "content_hash": d["content_hash"], "manifest_path": d["manifest_path"]})
        dv = d["parent_version"]
    return {"model": m, "datasets": datasets, "train_config": m["train_config_path"], "eval_report": m["eval_report_path"]}
