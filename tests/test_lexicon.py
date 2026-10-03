import json

import numpy as np
import pytest
from tests.helpers import make_client

from amanuensis.asr.streaming import Streamer
from amanuensis.audio.vad import VadSegmenter
from amanuensis.config import SAMPLE_RATE, WINDOW
from amanuensis.eval.biasing import bias_prompt, compare_biasing
from amanuensis.store import db, lexicon
from amanuensis.text.spelling import check_spelling
from amanuensis.ui.app import create_app
from tests.test_streaming import SCALE, FakeEngine, cfg, fake_vad, speech


@pytest.fixture
def conn():
    return db.connect(":memory:", check_same_thread=False)


def test_owner_entries_are_approved_and_proposals_are_not(conn):
    a = lexicon.add_entry(conn, "PCCOE", ["PCCOE", "pccoe", " P C C O E "], "name")
    p = lexicon.add_entry(conn, "Kubernetes", ["kubernetes"], "term", source="proposed")
    by_id = {e["id"]: e for e in lexicon.list_entries(conn)}
    assert by_id[a]["approved"] and not by_id[p]["approved"]
    assert by_id[a]["variants"] == ["p c c o e", "pccoe"]  # lowercased, deduped, exact canonical removed
    assert lexicon.approved_terms(conn) == ["PCCOE"]  # proposals never reach prompts or casing
    lexicon.set_approved(conn, p)
    assert lexicon.approved_terms(conn) == ["PCCOE", "Kubernetes"]


def test_variant_map_only_from_approved(conn):
    lexicon.add_entry(conn, "PCCOE", ["pccoe"], "name")
    lexicon.add_entry(conn, "Foo", ["fu"], "name", source="proposed")
    assert lexicon.variant_map(conn) == {"pccoe": "PCCOE"}


def test_validation(conn):
    with pytest.raises(ValueError):
        lexicon.add_entry(conn, "  ", [], "name")
    with pytest.raises(ValueError):
        lexicon.add_entry(conn, "x", [], "nonsense")
    with pytest.raises(ValueError):
        lexicon.add_entry(conn, "x", [], "name", source="agent")
    with pytest.raises(KeyError):
        lexicon.set_approved(conn, 99)
    with pytest.raises(KeyError):
        lexicon.update_entry(conn, 99, "x", [], "name")


def test_update_and_delete(conn):
    i = lexicon.add_entry(conn, "Pune", [], "name")
    lexicon.update_entry(conn, i, "Pune City", ["pune"], "term")
    e = lexicon.list_entries(conn)[0]
    assert (e["canonical"], e["variants"], e["kind"], e["approved"]) == ("Pune City", ["pune"], "term", True)
    lexicon.delete_entry(conn, i)
    assert lexicon.list_entries(conn) == []


def test_correctly_cased_term_is_not_flagged_but_wrong_casing_is():
    variants = {"pccoe": "PCCOE"}
    assert check_spelling("at PCCOE today", variants) == []
    assert [f.suggestion for f in check_spelling("at pccoe today", variants)] == ["PCCOE"]


def test_lexicon_api_and_spellcheck_uses_it(tmp_path, conn):
    client = make_client(create_app(conn, tmp_path, ["en"], {}))
    r = client.post("/api/lexicon", json={"canonical": "PCCOE", "variants": ["pccoe"], "kind": "name"})
    assert r.status_code == 200
    assert client.post("/api/lexicon", json={"canonical": "", "kind": "name"}).status_code == 422
    flags = client.post("/api/spellcheck", json={"text": "pccoe"}).json()
    assert flags[0]["suggestion"] == "PCCOE"
    eid = client.get("/api/lexicon").json()[0]["id"]
    assert client.post(f"/api/lexicon/{eid}/delete").status_code == 200
    assert client.post(f"/api/lexicon/{eid}/delete").status_code == 404
    assert client.post("/api/spellcheck", json={"text": "pccoe"}).json() == []


def test_streamer_prompt_has_terms_first_capped_and_no_buffered_text():
    words = speech(["one", "two"], 1.0) + speech(["three", "four"], 4.0)
    c = cfg(max_prompt_terms=2)
    engine = FakeEngine(words)
    s = Streamer(engine, VadSegmenter(fake_vad(words), 0.5, 0.6), c, lambda u: None,
                 bias_terms=lambda: ["Alpha", "Beta", "Gamma"])
    audio = (np.arange(int(7 * SAMPLE_RATE) // WINDOW * WINDOW) / SCALE).astype(np.float32)
    for i in range(0, len(audio), WINDOW):
        s.feed(audio[i : i + WINDOW])
    s.close()
    assert engine.prompts[0] == "Alpha, Beta."  # capped at max_prompt_terms, no context yet
    assert "Gamma" not in " ".join(p or "" for p in engine.prompts)
    assert any(p and p.startswith("Alpha, Beta. ") and "one two" in p for p in engine.prompts)


def test_zero_terms_means_no_bias_prompt():
    assert bias_prompt([], 20) is None
    assert bias_prompt(["A", "B"], 0) is None
    assert bias_prompt(["A", "B", "C"], 2) == "A, B."


def test_compare_biasing_reports_improvement(tmp_path):
    entries = [{"audio": "a.wav", "text": "i study at PCCOE", "language": "en"},
               {"audio": "b.wav", "text": "no terms here", "language": "de"}]

    def transcribe(path, prompt):
        if path.endswith("a.wav"):
            return "i study at PCCOE" if prompt and "PCCOE" in prompt else "i study at pick o"
        return "no terms here"

    r = compare_biasing(transcribe, entries, tmp_path, ["PCCOE"], 20)
    assert r["baseline"]["overall"]["term_accuracy"] == 0.0 and r["biased"]["overall"]["term_accuracy"] == 1.0
    assert r["delta"]["term_accuracy"] == 1.0 and r["delta"]["normalized_wer"] < 0
    assert r["baseline"]["per_language"]["de"]["normalized_wer"] == 0.0  # unaffected slice stays put
    json.dumps(r)  # report is serializable
