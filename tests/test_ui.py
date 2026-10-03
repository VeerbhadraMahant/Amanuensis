import numpy as np
import pytest
from fastapi.testclient import TestClient

from amanuensis.store import db, sessions
from amanuensis.ui.app import create_app

TAGS = ["en", "de", "hi", "mr"]


@pytest.fixture
def client(tmp_path):
    conn = db.connect(":memory:", check_same_thread=False)
    sid = sessions.start_session(conn, "base:small", "live")
    for i in range(2):
        sessions.log_utterance(conn, tmp_path, sid, np.zeros(16000, np.float32), f"raw {i}", f"norm {i}", 800.0)
    return TestClient(create_app(conn, tmp_path, TAGS, {"kyaa": "kya"}))


def test_index_and_config(client):
    assert "Amanuensis" in client.get("/").text
    assert client.get("/api/config").json() == {"tags": TAGS}


def test_list_and_audio(client):
    items = client.get("/api/utterances").json()
    assert [u["hypothesis_raw"] for u in items] == ["raw 0", "raw 1"]
    audio = client.get(f"/api/audio/{items[0]['id']}")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert client.get("/api/audio/999").status_code == 404


def test_review_flow_and_stats(client):
    a, b = [u["id"] for u in client.get("/api/utterances").json()]
    assert client.post(f"/api/utterances/{a}/approve", json={"tags": ["en"]}).status_code == 200
    assert client.post(f"/api/utterances/{b}/correct", json={"text": "fixed", "tags": ["de"]}).status_code == 200
    assert client.get("/api/utterances").json() == []
    s = client.get("/api/stats").json()
    assert s["counts"]["corrected"] == 1 and s["counts"]["approved_as_is"] == 1 and s["correction_rate"] == 0.5


def test_validation_errors(client):
    a = client.get("/api/utterances").json()[0]["id"]
    assert client.post(f"/api/utterances/{a}/correct", json={"text": "", "tags": ["en"]}).status_code == 422
    assert client.post(f"/api/utterances/{a}/correct", json={"text": "x", "tags": ["zz"]}).status_code == 422
    assert client.post("/api/utterances/999/reject").status_code == 404


def test_audio_path_outside_audio_dir_is_not_served(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (tmp_path / "secret.wav").write_bytes(b"RIFFsecret")
    conn = db.connect(":memory:", check_same_thread=False)
    sid = sessions.start_session(conn, "m", "live")
    uid = sessions.log_utterance(conn, audio_dir, sid, np.zeros(160, np.float32), "r", "n", None)
    conn.execute("UPDATE utterances SET audio_path = ? WHERE id = ?", ("../secret.wav", uid))
    conn.commit()
    assert TestClient(create_app(conn, audio_dir, TAGS, {})).get(f"/api/audio/{uid}").status_code == 404


def test_spellcheck_endpoint(client):
    flags = client.post("/api/spellcheck", json={"text": "kyaa hai"}).json()
    assert flags == [{"word": "kyaa", "suggestion": "kya", "start": 0, "end": 4}]
