"""Parity check (ADR-004): the CT2 model must transcribe like the merged HF model before it is used.

Both models are run in fp32 on CPU with greedy decoding so any difference comes from conversion.
"""
from dataclasses import dataclass
from pathlib import Path

from amanuensis.eval.metrics import normalized_wer


@dataclass(frozen=True)
class ParityResult:
    wer: float  # CT2 transcripts scored against the merged-HF transcripts
    exact_match_share: float
    passed: bool
    n: int


def compare(hf_texts: list[str], ct2_texts: list[str], max_wer: float) -> ParityResult:
    wer = normalized_wer(hf_texts, ct2_texts)
    same = sum(a.strip().lower() == b.strip().lower() for a, b in zip(hf_texts, ct2_texts, strict=True))
    return ParityResult(wer, same / len(hf_texts), wer <= max_wer, len(hf_texts))


def parity_check(hf_dir: Path, ct2_dir: Path, audio_paths: list[Path], language: str, max_wer: float) -> ParityResult:
    """language is the short code for faster-whisper (e.g. 'en') and its long name for HF ('english')."""
    import torch
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    names = {"en": "english", "de": "german", "hi": "hindi", "mr": "marathi"}
    processor = WhisperProcessor.from_pretrained(hf_dir)
    hf = WhisperForConditionalGeneration.from_pretrained(hf_dir, torch_dtype=torch.float32).eval()
    ct2 = WhisperModel(str(ct2_dir), device="cpu", compute_type="float32")
    hf_texts, ct2_texts = [], []
    for path in audio_paths:
        audio = decode_audio(str(path), sampling_rate=16000)
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = hf.generate(feats, language=names[language], task="transcribe", num_beams=1, do_sample=False)
        hf_texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        segs, _ = ct2.transcribe(audio, language=language, beam_size=1, temperature=0.0,
                                 condition_on_previous_text=False, without_timestamps=True)
        ct2_texts.append(" ".join(s.text for s in segs))
    return compare(hf_texts, ct2_texts, max_wer)
