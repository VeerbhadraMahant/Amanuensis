"""Live dictation into the overlay. Usage: uv run python -m amanuensis.live"""
import threading
import time

from amanuensis.asr.engine import FasterWhisperEngine
from amanuensis.asr.streaming import Streamer
from amanuensis.audio.capture import open_microphone
from amanuensis.audio.ring_buffer import RingBuffer
from amanuensis.audio.vad import SileroProb, VadSegmenter
from amanuensis.config import SAMPLE_RATE, WINDOW, load_paths, load_streaming, load_variants
from amanuensis.logging import log
from amanuensis.output.overlay import Overlay
from amanuensis.store import db, sessions
from amanuensis.text.normalizer import canonicalize


def main() -> None:
    cfg = load_streaming()
    paths = load_paths()
    conn = db.connect(paths.db_path, check_same_thread=False)
    session_id = sessions.start_session(conn, f"base:{cfg.model}", "live")
    variants = load_variants(paths.variants_file)
    engine = FasterWhisperEngine(cfg)
    overlay = Overlay("CPU fallback: reduced accuracy" if engine.fallback else "")
    ring = RingBuffer(30 * SAMPLE_RATE)
    seg = VadSegmenter(SileroProb(), cfg.vad_threshold, cfg.min_silence_s)

    def on_utterance(audio, raw, latency_ms) -> None:
        sessions.log_utterance(conn, paths.audio_dir, session_id, audio, raw, canonicalize(raw, variants), latency_ms)

    streamer = Streamer(engine, seg, cfg, overlay.push, on_utterance=on_utterance)
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
    sessions.end_session(conn, session_id)
    log("live_stopped", ring_overruns=ring.overruns, stats=streamer.stats.summary())


if __name__ == "__main__":
    main()
