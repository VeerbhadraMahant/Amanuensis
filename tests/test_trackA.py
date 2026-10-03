import json

import numpy as np
import pytest
import torch

from trackA import data
from trackA.decode import beam_search, greedy
from trackA.features import frame, hz_to_mel, log_mel, mel_filterbank, mel_to_hz
from trackA.model import CtcModel, out_lengths
from trackA.text import BLANK, VOCAB, decode, encode

SR = 16000


def sine(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------- features ----------

def test_mel_scale_roundtrip_and_monotonic():
    f = np.array([0.0, 100.0, 1000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f)
    assert np.all(np.diff(hz_to_mel(f)) > 0)


def test_filterbank_shape_nonnegative_and_each_filter_covers_some_bins():
    fb = mel_filterbank()
    assert fb.shape == (80, 201) and (fb >= 0).all() and (fb.max(axis=1) > 0).all()
    assert fb.max() <= 1.0 + 1e-9


def test_framing_counts_and_overlap():
    x = np.arange(1000, dtype=np.float32)
    fr = frame(x, 400, 160)
    assert fr.shape == (1 + (1000 - 400) // 160, 400) and fr[1, 0] == 160 and fr[0, 160] == fr[1, 0]
    assert frame(np.zeros(100, np.float32), 400, 160).shape == (1, 400)  # shorter than a window is padded


def test_log_mel_shape_and_a_sine_peaks_in_the_right_mel_bin():
    feats = log_mel(sine(1000, 1.0), normalize=False)
    assert feats.shape == (1 + (SR - 400) // 160, 80)
    fb = mel_filterbank()
    expected = int(np.argmax(fb[:, round(1000 / (SR / 400))]))  # the filter that covers 1 kHz
    assert abs(int(feats.mean(axis=0).argmax()) - expected) <= 1


def test_normalization_gives_zero_mean_unit_variance_per_bin():
    f = log_mel(np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1)
    assert np.allclose(f.mean(axis=0), 0, atol=1e-4) and np.allclose(f.std(axis=0), 1, atol=1e-2)


def test_our_log_mel_tracks_whispers_extractor():
    """Independent cross-check against faster-whisper's numpy log-mel. Whisper uses a different mel filterbank, so bin m
    covers different frequencies in the two and per-bin correlation is meaningless. Frame timing and overall level are
    comparable: the per-frame mean log energy of an amplitude-modulated signal must correlate strongly, and only when
    the frames are aligned (Whisper centres its frames: reflect-pad n_fft // 2 on both sides)."""
    from faster_whisper.feature_extractor import FeatureExtractor

    rng = np.random.default_rng(1)
    t = np.arange(2 * SR) / SR
    envelope = 0.1 + 0.9 * np.abs(np.sin(2 * np.pi * 1.5 * t))
    audio = (envelope * (np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 2500 * t)) + 0.02 * rng.standard_normal(2 * SR)).astype(np.float32)
    theirs = FeatureExtractor(feature_size=80)(audio, padding=False).mean(axis=0)
    ours = log_mel(np.pad(audio, (200, 200), mode="reflect"), normalize=False).mean(axis=1)
    n = min(len(theirs), len(ours))
    assert np.corrcoef(ours[:n], theirs[:n])[0, 1] > 0.95
    shifted = np.corrcoef(ours[2:n], theirs[: n - 2])[0, 1]  # two frames out of step must look clearly worse
    assert shifted < 0.85


# ---------- text and decoding ----------

def test_text_roundtrip_and_unknowns_dropped():
    assert decode(encode("It's a test")) == "it's a test"
    assert decode(encode("héllo, wörld 42!")) == "hllo wrld "
    assert VOCAB[BLANK] == "<blank>" and BLANK not in encode("anything at all")


def lp(rows):
    return np.log(np.array(rows, dtype=np.float64))


def test_greedy_collapses_repeats_and_blanks():
    a, b = encode("a")[0], encode("b")[0]
    probs = np.full((6, len(VOCAB)), 1e-6)
    for t, s in enumerate([a, a, BLANK, a, b, b]):
        probs[t, s] = 1.0
    assert decode(greedy(np.log(probs / probs.sum(axis=1, keepdims=True)))) == "aab"  # blank separates the two a groups


def test_beam_search_merges_alignments_where_greedy_fails():
    """P(blank)=0.6 per frame so greedy emits nothing (p=0.216), but the text 'a' has p=1-0.216=0.784 summed over alignments."""
    a = encode("a")[0]
    probs = np.full((3, len(VOCAB)), 1e-9)
    probs[:, BLANK], probs[:, a] = 0.6, 0.4
    probs /= probs.sum(axis=1, keepdims=True)
    assert greedy(np.log(probs)) == [] and beam_search(np.log(probs), beam=8) == [a]


def test_beam_search_equals_greedy_on_confident_input():
    ids = [BLANK, 3, 3, BLANK, 5, 6, 6, BLANK, 4]
    probs = np.full((len(ids), len(VOCAB)), 0.001)
    for t, i in enumerate(ids):
        probs[t, i] = 0.9
    logp = np.log(probs / probs.sum(axis=1, keepdims=True))
    assert beam_search(logp, 8) == greedy(logp) == [3, 5, 6, 4]


def ctc_logprob(logp: np.ndarray, labels: list[int]) -> float:
    """Exact log P(labels | input), summed over all alignments (torch's CTC loss is the negative of this)."""
    if not labels:
        return float(logp[:, BLANK].sum())
    loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp).float().unsqueeze(1), torch.tensor([labels]),
                                        torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=BLANK, reduction="sum")
    return -float(loss)


def test_wide_beam_never_finds_a_less_probable_sequence_than_greedy():
    rng = np.random.default_rng(5)
    wins = 0
    for _ in range(25):
        raw = rng.random((14, 7)) ** 3  # peaky-ish distributions
        logp = np.log(raw / raw.sum(axis=1, keepdims=True))
        g, b = greedy(logp), beam_search(logp, beam=16)
        assert ctc_logprob(logp, b) >= ctc_logprob(logp, g) - 1e-5
        wins += ctc_logprob(logp, b) > ctc_logprob(logp, g) + 1e-5
    assert wins >= 1  # and it does strictly better at least sometimes, so the search is doing real work


# ---------- model ----------

@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_out_lengths_matches_real_output(arch):
    m = CtcModel(d_model=32, layers=1, heads=2, arch=arch).eval()
    for t in (37, 100, 101, 400):
        logp, olens = m(torch.randn(1, t, 80), torch.tensor([t]))
        assert logp.shape[1] == int(olens[0]) == int(out_lengths(torch.tensor([t]))[0])


@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_padding_does_not_change_a_sequences_output(arch):
    torch.manual_seed(0)
    m = CtcModel(d_model=32, layers=2, heads=2, dropout=0.0, arch=arch).eval()
    short, long_ = torch.randn(1, 60, 80), torch.randn(1, 100, 80)
    alone, _ = m(short, torch.tensor([60]))
    batch = torch.zeros(2, 100, 80)
    batch[0, :60], batch[1] = short[0], long_[0]
    together, olens = m(batch, torch.tensor([60, 100]))
    n = int(olens[0])
    assert torch.allclose(alone[0, :n], together[0, :n], atol=1e-4)


@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_ctc_overfits_a_tiny_set_which_shows_the_training_path_works(arch):
    torch.manual_seed(0)
    targets = ["ab", "ba", "abc", "cab"]
    rng = np.random.default_rng(0)
    feats = [torch.from_numpy(rng.standard_normal((80, 80)).astype(np.float32)) for _ in targets]
    ids = [torch.tensor(encode(t)) for t in targets]
    m = CtcModel(d_model=64, layers=2, heads=2, dropout=0.0, arch=arch)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    x, lens = torch.stack(feats), torch.tensor([80] * 4)
    first = last = None
    for step in range(800 if arch == "gru" else 250):  # the GRU memorizes random features more slowly
        logp, olens = m(x, lens)
        loss = ctc(logp.transpose(0, 1), torch.cat(ids), olens, torch.tensor([len(i) for i in ids]))
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < 0.1 * first
    m.eval()
    logp, olens = m(x, lens)
    hyps = [decode(greedy(logp[i, : olens[i]].detach().numpy())) for i in range(4)]
    # Memorizing random noise is easy for the transformer and slow for the GRU; the loss drop above is the real check.
    assert sum(h == t for h, t in zip(hyps, targets)) >= (2 if arch == "gru" else 3)


# ---------- data ----------

def make_corpus(tmp_path, n_utts=6):
    chap = tmp_path / "19" / "198"
    chap.mkdir(parents=True)
    lines = []
    for i in range(n_utts):
        uid = f"19-198-{i:04d}"
        (chap / f"{uid}.flac").write_bytes(b"x")
        lines.append(f"{uid} HELLO WORLD NUMBER {i}")
    (chap / "19-198.trans.txt").write_text("\n".join(lines))
    return tmp_path


def test_list_items_parses_librispeech_layout(tmp_path):
    items = data.list_items(make_corpus(tmp_path))
    assert len(items) == 6 and items[0].text == "HELLO WORLD NUMBER 0" and items[0].audio.name == "19-198-0000.flac"


def test_list_items_skips_transcripts_whose_audio_is_missing(tmp_path):
    corpus = make_corpus(tmp_path)
    (corpus / "19" / "198" / "19-198-0003.flac").unlink()
    assert [i.audio.name for i in data.list_items(corpus)] == [f"19-198-000{k}.flac" for k in (0, 1, 2, 4, 5)]


def test_cache_selection_is_nested_and_stops_at_the_hour_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda path, sampling_rate: sine(300, 1.0))  # 1 s each
    items = data.list_items(make_corpus(tmp_path))
    index = data.build_cache(items, tmp_path / "cache", max_hours=3.5 / 3600)
    assert len(index) == 4  # stops once 3.5 s are reached
    small, large = data.select_hours(index, 1.5 / 3600), data.select_hours(index, 3.5 / 3600)
    assert [e["audio"] for e in small] == [e["audio"] for e in large][: len(small)] and len(small) < len(large)
    again = data.build_cache(items, tmp_path / "cache", max_hours=6 / 3600)  # resumes, does not redo the first four
    assert [e["npy"] for e in again[:4]] == [e["npy"] for e in index]
    assert json.loads((tmp_path / "cache" / "index.json").read_text())[0]["ids"] == encode("HELLO WORLD NUMBER 0")


def test_collate_pads_and_concatenates_targets(tmp_path, monkeypatch):
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda path, sampling_rate: sine(300, 1.0))
    index = data.build_cache(data.list_items(make_corpus(tmp_path, 3)), tmp_path / "c")
    feats, lens, targets, tlens = data.collate(tmp_path / "c", index)
    assert feats.shape[0] == 3 and feats.shape[2] == 80 and lens.tolist() == [e["frames"] for e in index]
    assert targets.numel() == int(tlens.sum())


def test_length_buckets_cover_everything_and_respect_the_frame_cap():
    entries = [{"frames": f, "ids": []} for f in (100, 120, 300, 310, 900, 50)]
    batches = data.length_buckets(entries, max_frames=700, shuffle=False)
    assert sorted(id(e) for b in batches for e in b) == sorted(id(e) for e in entries)
    for b in batches:
        assert len(b) * max(e["frames"] for e in b) <= 700 or len(b) == 1


# ---------- evaluation on the owner's English slice ----------

def test_vocab_normalize_reduces_references_to_what_the_model_can_say():
    from trackA.evaluate import vocab_normalize

    assert vocab_normalize("It's 3 o'clock, Pune!") == "it's o'clock pune"


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError):
        CtcModel(arch="lstm")


def test_evaluate_english_uses_only_the_en_slice_and_runs_end_to_end(tmp_path):
    from amanuensis.store.sessions import write_wav
    from trackA.evaluate import evaluate_english

    entries = []
    for i, lang in enumerate(("en", "de", "en")):
        write_wav(tmp_path / f"{i}.wav", sine(300 + 50 * i, 1.0))
        entries.append({"audio": f"{i}.wav", "text": f"sample sentence {i}", "language": lang})
    torch.manual_seed(0)
    res = evaluate_english(CtcModel(d_model=32, layers=1, heads=2).eval(), entries, tmp_path, beam=2)
    assert res["n"] == 2 and set(res) == {"n", "wer_greedy", "wer_beam2"} and res["wer_greedy"] >= 0
    with pytest.raises(ValueError):
        evaluate_english(CtcModel(d_model=32, layers=1, heads=2).eval(), [entries[1]], tmp_path)
