import sounddevice as sd

from amanuensis.audio.ring_buffer import RingBuffer
from amanuensis.config import SAMPLE_RATE, WINDOW


def open_microphone(ring: RingBuffer) -> sd.InputStream:
    """16 kHz mono stream writing 32 ms blocks into the ring buffer. Caller starts/stops it."""

    def callback(indata, frames, time_info, status) -> None:
        ring.write(indata[:, 0])

    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=WINDOW, callback=callback)
