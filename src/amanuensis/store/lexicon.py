"""Lexicon of names, terms and romanized words. Proposals stay unapproved until the owner accepts them."""
import json
import sqlite3

KINDS = ("name", "term", "romanized_word")


def _clean(canonical: str, variants: list[str], kind: str) -> tuple[str, str]:
    canonical = canonical.strip()
    if not canonical:
        raise ValueError("canonical spelling cannot be empty")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    # Variants are matched lowercase. Only a variant identical to the canonical is redundant:
    # "pccoe" -> "PCCOE" is the useful casing rule and must be kept.
    cleaned = sorted({v.strip().lower() for v in variants if v.strip() and v.strip() != canonical})
    return canonical, json.dumps(cleaned)


def add_entry(
    conn: sqlite3.Connection, canonical: str, variants: list[str], kind: str, source: str = "owner"
) -> int:
    """Owner entries are approved immediately; agent proposals (source='proposed') are not."""
    if source not in ("owner", "proposed"):
        raise ValueError("source must be 'owner' or 'proposed'")
    canonical, variants_json = _clean(canonical, variants, kind)
    cur = conn.execute(
        "INSERT INTO lexicon (canonical, variants, kind, source, approved) VALUES (?, ?, ?, ?, ?)",
        (canonical, variants_json, kind, source, int(source == "owner")),
    )
    conn.commit()
    return cur.lastrowid


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["variants"] = json.loads(d["variants"])
    d["approved"] = bool(d["approved"])
    return d


def list_entries(conn: sqlite3.Connection, approved: bool | None = None) -> list[dict]:
    where, args = ("WHERE approved = ?", [int(approved)]) if approved is not None else ("", [])
    return [_row(r) for r in conn.execute(f"SELECT * FROM lexicon {where} ORDER BY id", args)]


def update_entry(conn: sqlite3.Connection, entry_id: int, canonical: str, variants: list[str], kind: str) -> None:
    canonical, variants_json = _clean(canonical, variants, kind)
    cur = conn.execute(
        "UPDATE lexicon SET canonical = ?, variants = ?, kind = ? WHERE id = ?", (canonical, variants_json, kind, entry_id)
    )
    if cur.rowcount == 0:
        raise KeyError(entry_id)
    conn.commit()


def set_approved(conn: sqlite3.Connection, entry_id: int, approved: bool = True) -> None:
    if conn.execute("UPDATE lexicon SET approved = ? WHERE id = ?", (int(approved), entry_id)).rowcount == 0:
        raise KeyError(entry_id)
    conn.commit()


def delete_entry(conn: sqlite3.Connection, entry_id: int) -> None:
    if conn.execute("DELETE FROM lexicon WHERE id = ?", (entry_id,)).rowcount == 0:
        raise KeyError(entry_id)
    conn.commit()


def approved_terms(conn: sqlite3.Connection) -> list[str]:
    """Canonical spellings of approved entries: used for prompt biasing and casing."""
    return [e["canonical"] for e in list_entries(conn, approved=True)]


def variant_map(conn: sqlite3.Connection) -> dict[str, str]:
    """Approved entries' variants -> canonical (variants are stored lowercase)."""
    return {v: e["canonical"] for e in list_entries(conn, approved=True) for v in e["variants"]}
