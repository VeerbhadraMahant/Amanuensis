import pytest

from amanuensis.eval.report import Utterance, build_report
from amanuensis.training.parity import compare
from amanuensis.training.train_lora import load_train_config


def test_parity_compare_pass_and_fail():
    same = compare(["hello world", "good day"], ["Hello world", "good day"], 0.02)
    assert same.passed and same.wer == 0.0 and same.exact_match_share == 1.0 and same.n == 2
    off = compare(["one two three four"], ["one two three five"], 0.02)
    assert not off.passed and off.wer == pytest.approx(0.25) and off.exact_match_share == 0.0


def test_shipped_training_config_loads_and_is_machine_agnostic():
    cfg = load_train_config()
    assert cfg.device in ("cuda", "cpu") and cfg.runs_dir.parts[0] == "models"  # models/ is gitignored
    assert cfg.batch_size * cfg.grad_accum >= 1


def test_build_report_has_overall_per_language_and_latency():
    utts = [Utterance("en", "a b", "a b"), Utterance("de", "c d", "c x")]
    r = build_report(utts, None, ["a"], 1.2)
    assert r["overall"]["normalized_wer"] == pytest.approx(0.25)
    assert r["per_language"]["de"]["normalized_wer"] == 0.5 and r["latency_p50_s"] == 1.2
    assert r["overall"]["term_accuracy"] == 1.0
