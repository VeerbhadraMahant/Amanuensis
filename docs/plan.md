# Amanuensis: Plan

This plan builds the system in an order where every layer rests on something already measured and trusted. Agents come late on purpose: if the multi-agent loop is built before the scripts it orchestrates work reliably, you end up debugging the ML and the orchestration at the same time and can't tell which one is broken.

Each phase lists why it exists, its tasks, the dev subagent that owns most of the work, an exit criterion, and what you learn. Tick checkboxes as tasks finish and add a one-line note.

Rough timeline at part-time pace: 10 to 14 weeks for Phases 0 to 5. Track A runs alongside whenever you want a break.

## Ongoing habit (starts in Phase 2, never stops)

Dictate real things daily (notes, messages, project logs) for 15 to 30 minutes and review the day's utterances in the correction UI. Aim for a mix of all four languages that reflects how you actually speak. The model can only improve as fast as reviewed data accumulates.

---

## Phase 0: Foundation and baseline

**Why:** Without a fixed test set and a spelling convention, no later result means anything. This phase defines what "better" will mean.

- [x] Create repo, `uv` project, layout from `CLAUDE.md`, `.gitignore` covering `data/` and `models/` — Python pinned to 3.11 via uv; layout created.
- [x] Create dev subagents in `.claude/agents/` (use `/agents` in Claude Code, then edit the generated files to match the table in `CLAUDE.md`) — files written by hand to match the CLAUDE.md table; may need session restart to load.
- [ ] Write `docs/romanization.md` (human task): canonical spellings for your 150 to 200 most frequent Hindi and Marathi words, rules for long vowels, nasals, and common English loanwords — **BLOCKED (owner): stub created, spelling must come from you.**
- [x] Write ADR-001 to ADR-007 from `systemdesign.md` as separate files — ADR-001 to 007 written as short files; expand as decisions are tested.
- [ ] Record the frozen eval set (human task): about 20 minutes each for Hindi-English, Marathi-English, English, German, in natural speech, not read text — **BLOCKED (owner): needs your recordings.**
- [ ] Hand-transcribe the eval set following the spelling guide; build its manifest and record its hash — **BLOCKED (owner): needs your transcripts; `eval/manifest.py` is ready to hash them.**
- [x] Implement `eval/`: normalizer, WER, CER, lexicon term accuracy, per-language report (owner: `eval-engineer`) — normalizer, WER, CER, term accuracy, per-language report, manifest hash check; normalizer takes variants as input (none hardcoded).
- [x] Unit tests for the normalizer using real variants from the spelling guide (owner: `test-runner`) — 14 tests pass, but variants are SYNTHETIC; add real ones from romanization.md.
- [ ] Baseline: run Whisper small, medium, and large-v3-turbo (int8) on the eval set; record WER per language, decode speed, and VRAM use — **BLOCKED (owner): `python -m amanuensis.eval.baseline` is written but untested without eval data; VRAM use not yet recorded.**

**Exit:** a baseline report with per-language numbers for three model sizes, and a decision on which model to build Phase 1 around.

**You learn:** ASR evaluation, why normalization matters for WER, the gap between off-the-shelf models and your speech.

---

## Phase 1: Real-time pipeline (no training)

**Why:** Real-time streaming is the hardest engineering piece and is independent of fine-tuning. Getting it working on the base model means every later model improvement shows up immediately in daily use.

- [x] Audio capture with ring buffer (owner: `realtime-audio-engineer`) — RingBuffer + sounddevice callback; mic opens and captures ~16 kHz on this machine.
- [x] Silero VAD segmentation with configurable silence threshold — reuses the ONNX Silero model bundled with faster-whisper (no torch); it is stateless per call, so each window is scored with 1 s of context.
- [x] faster-whisper engine wrapper with config-driven model and compute type — config-driven; CPU fallback to tiny with a visible flag. Learned: a mid-sentence cut plus a prompt can make Whisper emit 224 junk tokens (~3 s), so decode uses temperature 0 and max_new_tokens scaled by audio length.
- [x] Streaming loop with LocalAgreement commit and buffer trimming — Learned: the prompt must contain only text that already left the buffer, otherwise Whisper skips words still in the audio (this dropped words until fixed).
- [x] Prompt with recent committed text — see note above; tested by test_prompt_never_contains_text_still_in_the_buffer.
- [x] Overlay window showing committed and tentative text — tkinter always-on-top; renders committed/tentative in a smoke test; not yet used with live speech.
- [x] Latency instrumentation: time to first partial, commit latency p50 and p95, decode time per cycle — first partial, commit p50/p95, decode time per cycle; falls-behind detection widens the re-decode interval.
- [x] Replay mode: feed a WAV file through the streaming pipeline as if live, so latency and accuracy are testable without a microphone — `python -m amanuensis.asr.replay FILE.wav [--ref TEXT]`; reports streaming vs offline WER.
- [ ] Run `code-reviewer` and `test-runner` in parallel on the finished pipeline (first practice of parallel subagents) — NOT DONE as subagents; I self-reviewed the diff and ran pytest (28 pass). Run the subagents when they are loaded.

**Status (Phase 1):** pipeline built and measured on ONE synthetic TTS clip (11.7 s, Whisper small int8, RTX 4060): first partial p50 0.98 s, commit p50 1.28 s / p95 1.65 s, decode p95 0.73 s, streaming WER == offline WER (0.0). Still owed: speak into the overlay live, and re-measure on the real eval set once it exists. ADR-002 en/de language detection is not implemented yet; the language is fixed in `configs/streaming.yaml`.

**Exit:** live dictation works in the overlay; latency measured on eval audio in replay mode; streaming WER compared with offline WER on the same audio.

**You learn:** streaming ASR, the latency vs stability trade-off, profiling GPU inference.

---

## Phase 2: Logging, storage, and correction UI

**Why:** This is the data engine. The faster you can correct, the faster the model improves. Treat UI speed as a core metric.

- [ ] SQLite schema and migrations from `systemdesign.md` section 6 (owner: main session, since it touches every plane)
- [ ] Utterance logging at the end of each VAD segment: audio, raw and normalized hypothesis, model version, latency
- [ ] Correction UI: list, playback, edit, approve as is, reject, language tags, keyboard shortcuts
- [ ] Live spelling check against the guide inside the editor
- [ ] Stats page: reviewed hours per language, correction rate, review throughput
- [ ] Start the daily dictation habit

**Exit:** you can review an utterance in under about 5 seconds on average, and reviewed data is accumulating in all four languages.

**You learn:** human-in-the-loop data design, why label quality bounds model quality.

---

## Phase 3: Lexicon and biasing

**Why:** Names and terms are cheaper to fix with biasing than with training. This also sets a stronger baseline that fine-tuning must beat.

- [ ] Lexicon table and UI to add, edit, and approve entries
- [ ] Prompt biasing with lexicon terms in the decoding prompt (owner: `realtime-audio-engineer`)
- [ ] Normalizer rules for lexicon casing and known variants (owner: `eval-engineer`)
- [ ] Measure lexicon term accuracy and overall WER with and without biasing on the eval set

**Exit:** measured improvement on lexicon term accuracy; new "biased baseline" recorded.

**You learn:** prompt conditioning in encoder-decoder ASR, and its limits (too many prompt terms can cause hallucinations; measure this).

---

## Phase 4: Manual fine-tuning pipeline

**Why:** Build every deterministic tool the agents will later use, and run the full cycle by hand at least twice. You need to know what normal looks like before you can automate recovery from abnormal.

Start this phase once you have roughly 2 to 3 hours of reviewed speech. Expect clearer gains at 5 hours and beyond.

- [ ] Dataset builder: pull reviewed utterances, validate, deduplicate, balance languages, mix in English and German replay data, write a versioned manifest (owner: `ml-training-engineer`)
- [ ] Data validators as standalone functions with tests (duration bounds, empty text, clipping, text-length vs audio-length, spelling compliance)
- [ ] LoRA training script for Whisper small first, config-driven, 8-bit loading, gradient checkpointing, VRAM check and dictation-process check before start
- [ ] Merge adapter, convert to CTranslate2, parity check against merged HF model
- [ ] Model registry: champion and challenger tracking with lineage
- [ ] Promotion rule as a pure function reading `configs/promotion.yaml` (owner: `eval-engineer`)
- [ ] First manual cycle on Whisper small; write up results
- [ ] Repeat on Whisper medium if VRAM allows; compare
- [ ] Evaluate ADR-002 (language token strategy) on the code-switched slices; revise if needed

**Exit:** at least one fine-tuned model beats the biased baseline on normalized WER for the code-switched slices without breaching the German and English tolerance, produced entirely by scripts.

**You learn:** parameter-efficient fine-tuning, memory-constrained training, catastrophic forgetting, model lineage.

---

## Phase 5: Multi-agent improvement loop

**Why:** Now the tools are trusted, so the agent layer can focus on orchestration, delegation, failure recovery, and reporting. This is the main multi-agent learning phase.

- [ ] Read the current Claude Agent SDK docs on subagents, tool definitions, and permissions before writing anything; note anything that changed since this plan was written
- [ ] Wrap Phase 4 functions as tools with narrow signatures; no tool exposes raw audio or write access to eval data or promotion config
- [ ] Implement specialists one at a time, each tested alone before being added to the orchestrator: Evaluator first (read-mostly, safest), then Curator, Trainer, Lexicographer, Gatekeeper
- [ ] Orchestrator with step budget, wall-clock budget, retry policy, and run report
- [ ] Failure injection tests: corrupt an audio file, force an out-of-memory, break the conversion step, tamper with the eval manifest hash; confirm the loop detects each and stops or recovers correctly
- [ ] Owner approval step for promotion, surfaced in the correction UI
- [ ] Run three or more full loop cycles on real data; compare agent runs with your Phase 4 manual runs

**Exit:** the loop runs end to end from a trigger, produces a readable report, refuses bad data and bad models, and survives injected failures.

**You learn:** orchestrator and specialist patterns, tool design and least privilege for agents, deterministic gates vs agent judgment, observability for agent systems.

---

## Phase 6: Polish and extensions (pick based on interest)

- [ ] System-wide text injection adapter for your OS
- [ ] Async LLM cleanup pass on finalized utterances (removing disfluencies, enforcing spelling), off by default, measured separately
- [ ] Second held-out eval set to detect overfitting to the first
- [ ] Results page: WER per version per language over time, correction rate over time
- [ ] Short write-up or blog post with the result charts

---

## Track A: from-scratch study (parallel, independent)

- [ ] Log-mel feature pipeline from raw audio, implemented by hand
- [ ] Small CTC model trained on a LibriSpeech subset
- [ ] Greedy CTC decoding, then a small beam search
- [ ] WER vs training hours curve (e.g. 10, 25, 50, 100 hours)
- [ ] Evaluate on your English eval slice and compare with base and fine-tuned Whisper
- [ ] Write up the comparison

**You learn:** the fundamentals underneath the product, and evidence for why pretrained models dominate at small data scale.

---

## Resume framing (fill in with real numbers as they arrive)

"Built Amanuensis, a personalized real-time multilingual dictation system (English, German, code-switched Hindi and Marathi) with a human-in-the-loop correction pipeline and a multi-agent improvement loop that curates data, fine-tunes Whisper with LoRA on an 8 GB GPU, and promotes models through per-language regression gates. Reduced normalized WER on code-switched speech from X% to Y% while keeping English and German within Z points of baseline, at a p50 commit latency of N seconds."

The numbers are what make that line credible. Without the frozen eval set from Phase 0, there are no numbers.