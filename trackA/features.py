"""Log-mel features from raw audio, implemented by hand with numpy (no librosa/torchaudio).

Pipeline: pre-emphasis-free framing -> Hann window -> power spectrum (rFFT) -> triangular mel filterbank -> log.
Defaults match the usual ASR setup: 25 ms window (400), 10 ms hop (160), 80 mel bins at 16 kHz.
"""
import numpy as np


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)  # HTK mel scale


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int = 16000, n_fft: int = 400, n_mels: int = 80, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular filters, equally spaced on the mel scale."""
    fmax = fmax or sr / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(fft_freqs)))
    for m in range(n_mels):
        left, centre, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (fft_freqs - left) / (centre - left)
        down = (right - fft_freqs) / (right - centre)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def frame(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(frames, n_fft) overlapping windows. The tail shorter than one window is dropped."""
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    n = 1 + (len(audio) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx]


def log_mel(audio: np.ndarray, sr: int = 16000, n_fft: int = 400, hop: int = 160, n_mels: int = 80, normalize: bool = True) -> np.ndarray:
    """(frames, n_mels) float32. With normalize, each mel bin is standardized over the utterance."""
    window = np.hanning(n_fft + 1)[:-1]  # periodic Hann
    spec = np.fft.rfft(frame(audio.astype(np.float64), n_fft, hop) * window, axis=1)
    power = spec.real**2 + spec.imag**2
    mel = power @ mel_filterbank(sr, n_fft, n_mels).T
    feats = np.log(np.maximum(mel, 1e-10))
    if normalize:
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-5)
    return feats.astype(np.float32)
