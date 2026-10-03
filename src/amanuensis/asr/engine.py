import numpy as np

from amanuensis.asr.types import Word
from amanuensis.config import SAMPLE_RATE, StreamingConfig
from amanuensis.logging import log


class FasterWhisperEngine:
    """faster-whisper wrapper. Falls back to a small CPU model if no GPU (visible via .fallback)."""

    def __init__(self, cfg: StreamingConfig) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        self._language = cfg.language
        gpu = cfg.device in ("auto", "cuda") and ctranslate2.get_cuda_device_count() > 0
        self.fallback = not gpu
        if gpu:
            self._model = WhisperModel(cfg.model, device="cuda", compute_type=cfg.compute_type)
        else:
            log("gpu_unavailable_cpu_fallback", model=cfg.cpu_fallback_model)
            self._model = WhisperModel(cfg.cpu_fallback_model, device="cpu", compute_type="int8")
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), None)  # warm-up: first decode is slow

    def transcribe(self, audio: np.ndarray, prompt: str | None) -> list[Word]:
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            initial_prompt=prompt,
            word_timestamps=True,
            beam_size=1,
            temperature=0.0,  # no temperature fallback: decode time must stay bounded
            # A mid-sentence cut plus a prompt can make Whisper emit 224 junk tokens (~3 s).
            # Cap tokens by audio length (12/s leaves room for romanized Hindi/Marathi).
            max_new_tokens=int(len(audio) / SAMPLE_RATE * 12) + 16,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        words = (Word(w.word.strip(), float(w.start), float(w.end)) for s in segments for w in (s.words or []))
        return [w for w in words if any(c.isalnum() for c in w.text)]
