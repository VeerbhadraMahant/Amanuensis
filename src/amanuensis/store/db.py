"""SQLite schema and migrations (docs/systemdesign.md section 6, ADR-005)."""
import sqlite3
from pathlib import Path

MIGRATIONS = [
    """
    CREATE TABLE sessions (
        id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
        model_version TEXT NOT NULL, mode TEXT NOT NULL, notes TEXT
    );
    CREATE TABLE utterances (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        audio_path TEXT NOT NULL, duration_s REAL NOT NULL,
        hypothesis_raw TEXT NOT NULL, hypothesis_normalized TEXT NOT NULL,
        final_text TEXT,
        status TEXT NOT NULL DEFAULT 'raw'
            CHECK (status IN ('raw', 'corrected', 'approved_as_is', 'rejected')),
        language_tags TEXT NOT NULL DEFAULT '[]',
        latency_ms REAL, created_at TEXT NOT NULL, reviewed_at TEXT
    );
    CREATE INDEX idx_utterances_status ON utterances(status);
    CREATE INDEX idx_utterances_session ON utterances(session_id);
    CREATE TABLE lexicon (
        id INTEGER PRIMARY KEY, canonical TEXT NOT NULL, variants TEXT NOT NULL DEFAULT '[]',
        kind TEXT NOT NULL CHECK (kind IN ('name', 'term', 'romanized_word')),
        source TEXT NOT NULL CHECK (source IN ('owner', 'proposed')),
        approved INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE dataset_versions (
        version TEXT PRIMARY KEY, created_at TEXT NOT NULL, manifest_path TEXT NOT NULL,
        utterance_count INTEGER NOT NULL, hours_by_language TEXT NOT NULL,
        content_hash TEXT NOT NULL, parent_version TEXT REFERENCES dataset_versions(version)
    );
    CREATE TABLE model_versions (
        version TEXT PRIMARY KEY, base_model TEXT NOT NULL, adapter_path TEXT, ct2_path TEXT,
        dataset_version TEXT REFERENCES dataset_versions(version),
        train_config_path TEXT, eval_report_path TEXT,
        status TEXT NOT NULL CHECK (status IN ('champion', 'challenger', 'rejected', 'archived')),
        created_at TEXT NOT NULL
    );
    CREATE TABLE loop_runs (
        id INTEGER PRIMARY KEY, trigger TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
        steps TEXT NOT NULL DEFAULT '[]', outcome TEXT, report_path TEXT
    );
    """,
    """
    CREATE TABLE promotion_requests (
        id INTEGER PRIMARY KEY, model_version TEXT NOT NULL REFERENCES model_versions(version),
        decision TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'approved', 'declined')),
        created_at TEXT NOT NULL, decided_at TEXT
    );
    """,
    "ALTER TABLE promotion_requests ADD COLUMN champion_at_request TEXT;",
]


def connect(path: Path | str, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the database, creating it and applying pending migrations.

    Pass check_same_thread=False when one connection is shared by a worker thread (live logging, web UI).
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")  # dictation, the UI and the loop share this file
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {i}")
    conn.commit()
