"""Live dictation into the overlay. Usage: uv run python -m amanuensis.live"""
import threading
import time

from amanuensis.asr.engine import FasterWhisperEngine
from amanuensis.asr.streaming import Streamer
from amanuensis.audio.capture import open_microphone
from amanuensis.audio.ring_buffer import RingBuffer
from amanuensis.audio.vad import SileroProb, VadSegmenter
from amanuensis.config import SAMPLE_RATE, WINDOW, load_streaming
from amanuensis.logging import log
from amanuensis.output.overlay import Overlay


def main() -> None:
    cfg = load_streaming()
    engine = FasterWhisperEngine(cfg)
    overlay = Overlay("CPU fallback: reduced accuracy" if engine.fallback else "")
    ring = RingBuffer(30 * SAMPLE_RATE)
    seg = VadSegmenter(SileroProb(), cfg.vad_threshold, cfg.min_silence_s)
    streamer = Streamer(engine, seg, cfg, overlay.push)
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set():
            w = ring.read(WINDOW)
            if w is None:
                time.sleep(0.005)
            else:
                streamer.feed(w)

    t = threading.Thread(target=worker, daemon=True)
    with open_microphone(ring):
        t.start()
        overlay.run()
        stop.set()
        t.join()
    streamer.close()
    log("live_stopped", ring_overruns=ring.overruns, stats=streamer.stats.summary())


if __name__ == "__main__":
    main()
