import json

import pytest

from amanuensis.eval.manifest import ManifestHashError, content_hash, verify
from amanuensis.eval.metrics import cer, normalized_wer, term_accuracy, wer
from amanuensis.eval.report import Utterance, per_language_report


def test_wer_zero_and_one_error():
    assert wer(["a b c"], ["a b c"]) == 0.0
    assert wer(["a b c d"], ["a b x d"]) == pytest.approx(0.25)


def test_cer():
    assert cer(["abcd"], ["abxd"]) == pytest.approx(0.25)


def test_normalized_wer_ignores_case_punct_and_variants():
    assert normalized_wer(["Kya hai."], ["kyaa hai"], {"kyaa": "kya"}) == 0.0
    assert wer(["Kya hai."], ["kyaa hai"]) > 0


def test_term_accuracy():
    refs = ["i study at pccoe", "nothing here"]
    hyps = ["i study at pccoe", "nothing here"]
    assert term_accuracy(refs, hyps, ["PCCOE"]) == 1.0
    assert term_accuracy(refs, ["i study at pcc", "x"], ["PCCOE"]) == 0.0
    assert term_accuracy(["no terms"], ["no terms"], ["PCCOE"]) is None


def test_per_language_report_splits_slices():
    rep = per_language_report(
        [Utterance("en", "a b", "a b"), Utterance("de", "a b", "a x")],
        terms=["a"],
    )
    assert rep["en"]["wer"] == 0.0 and rep["de"]["wer"] == pytest.approx(0.5)
    assert rep["en"]["n"] == 1 and rep["de"]["term_accuracy"] == 1.0


def test_manifest_hash_detects_tamper(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"audio")
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps([{"audio": "a.wav", "text": "x", "language": "en"}]), encoding="utf-8")
    h = content_hash(m, tmp_path)
    verify(m, tmp_path, h)
    (tmp_path / "a.wav").write_bytes(b"tampered")
    with pytest.raises(ManifestHashError):
        verify(m, tmp_path, h)


def test_freeze_records_the_hash_once_and_never_overwrites(tmp_path):
    from amanuensis.eval.manifest import freeze

    (tmp_path / "a.wav").write_bytes(b"audio")
    (tmp_path / "manifest.json").write_text(json.dumps([{"audio": "a.wav", "text": "x", "language": "en"}]), encoding="utf-8")
    digest = freeze(tmp_path)
    assert (tmp_path / "manifest.sha256").read_text() == digest == content_hash(tmp_path / "manifest.json", tmp_path)
    verify(tmp_path / "manifest.json", tmp_path, digest)
    (tmp_path / "a.wav").write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        freeze(tmp_path)  # cannot be used to bless a modified eval set
    assert (tmp_path / "manifest.sha256").read_text() == digest
