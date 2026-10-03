# Amanuensis: System Design

## 1. Purpose

Amanuensis is a real-time dictation system personalized to one speaker. It starts from a pretrained multilingual speech model and improves on the owner's voice, accent, vocabulary, and romanized spelling through a human-in-the-loop correction cycle, orchestrated offline by a team of agents.

The name: an amanuensis was the scribe who took down dictation for scholars and artists in Renaissance Europe. That is exactly the job.

## 2. Requirements

### Functional

- F1. Real-time transcription of live microphone input, with text appearing while the owner speaks.
- F2. Languages: English, German, Hindi, Marathi, including mid-sentence switching between English and Hindi or Marathi.
- F3. Output entirely in Latin script. Hindi and Marathi are romanized according to `docs/romanization.md`.
- F4. Adherence to a custom lexicon of names and technical terms.
- F5. Every utterance is logged (audio plus hypothesis) for later review.
- F6. A correction UI where the owner reviews, edits, or approves utterances.
- F7. An offline improvement loop that turns reviewed data into a fine-tuned model and promotes it only when it beats the current one.
- F8. Inserting text into other applications (system-wide dictation), behind a platform adapter.

### Non-functional

- N1. Latency: first partial text within about 1 s of speech onset; committed text within about 2 s (p50). These are starting targets, revised after Phase 1 measurements.
- N2. Runs fully offline on an 8 GB GPU for dictation.
- N3. Privacy: audio never leaves the machine.
- N4. Reproducibility: every model version traces back to a dataset version, a training config, and an eval report.
- N5. No silent regressions: a model that gets worse on any language beyond tolerance is never promoted.

### Out of scope (for now)

- Speaker diarization or multi-speaker use.
- Devanagari output.
- Mobile deployment.

## 3. Key design decisions

Each of these gets a full ADR in `docs/adr/`.

**ADR-001: Single Latin-script output space.** All four languages share one script. This removes the need for per-language model routing and makes code-switching a normal sequence-to-sequence problem. Cost: off-the-shelf models do not produce romanized Hindi or Marathi reliably, so fine-tuning is required, not optional, and spelling must be standardized by the owner.

**ADR-002: Language token strategy.** Whisper conditions on a language token. Initial approach: decode non-German speech with the English token (the romanized output is Latin-script anyway) and restrict automatic language detection to English vs German. Training targets use the same convention. Alternatives (a dedicated new token, per-session mode switch) are evaluated in Phase 4 if code-switched WER stalls.

**ADR-003: No LLM in the hot path.** The real-time path is deterministic and local. LLM-based cleanup, if added, runs asynchronously on finalized utterances and never blocks output.

**ADR-004: Train in Hugging Face, serve in CTranslate2.** Training uses transformers + PEFT LoRA. For serving, the adapter is merged into the base weights and converted to CTranslate2 for faster-whisper. A parity check confirms the converted model matches the merged HF model on an eval subset before the converted model is used.

**ADR-005: SQLite plus filesystem storage.** Audio as 16 kHz mono WAV files on disk; metadata, transcripts, and lineage in SQLite. Simple, inspectable, enough for one user.

**ADR-006: Frozen, hash-verified eval set.** A fixed set of owner recordings per language, hand-transcribed, never used for training. Its manifest hash is checked before every evaluation.

**ADR-007: Agents orchestrate, code decides.** Multi-agent orchestration lives in the offline improvement loop. Agents plan, call tools, recover from failures, and write reports. Pass/fail decisions (data validity rules, promotion thresholds) are deterministic code with config-defined thresholds that agents cannot modify.

## 4. Architecture overview

The system has three planes.

1. **Real-time plane:** microphone to text on screen, local and deterministic.
2. **Data plane:** session logs, the correction UI, the lexicon, and versioned datasets.
3. **Improvement plane:** the multi-agent loop that curates, trains, evaluates, and promotes.

Dev-time subagents in Claude Code (described in `CLAUDE.md`) are separate from runtime agents. They help build the system; the runtime agents are part of it.

## 5. Real-time plane

### Data flow

1. **Capture.** sounddevice streams 16 kHz mono audio into a ring buffer in small frames (about 30 ms).
2. **VAD.** Silero VAD marks speech and silence. Silence longer than a configured threshold closes an utterance.
3. **Streaming buffer.** While speech continues, the ASR engine re-decodes the growing audio buffer at a fixed interval (about 0.5 to 1 s).
4. **Local agreement.** Words are committed only when two consecutive decodes agree on the same prefix. This is the LocalAgreement approach from the whisper_streaming work (Machacek et al., 2023). Uncommitted words are shown as tentative text.
5. **Buffer trimming.** Once committed text reaches a sentence or long-pause boundary, the corresponding audio is dropped from the buffer to bound latency and compute. Committed text is passed back as the decoding prompt for context.
6. **Prompt biasing.** The decoding prompt includes recent committed text plus a short list of lexicon terms relevant to the current session.
7. **Normalization.** A deterministic normalizer maps known spelling variants to the canonical forms in `romanization.md` and applies lexicon casing (e.g. "pccoe" to "PCCOE").
8. **Output.** Committed text goes to the active output adapter (overlay window first, system-wide text injection later). Tentative text is shown only in the overlay.
9. **Logging.** When an utterance closes, its audio, raw hypothesis, normalized text, model version, timestamps, and latency metrics are written to the store with status `raw`.

### VRAM budget

- Dictation runs one model in int8 at a time. Whisper small and medium fit comfortably; larger models are benchmarked in Phase 0 before being considered.
- Training never runs while dictation is active. The trainer checks for the dictation process and GPU memory before starting.

### Failure handling

- If decoding falls behind real time (decode time exceeds the re-decode interval for N consecutive cycles), the engine increases the interval and logs a warning.
- If the GPU is unavailable, the engine falls back to CPU with a smaller model and shows a visible indicator.

## 6. Data plane

### Schema (SQLite)

- **sessions**: id, started_at, ended_at, model_version, mode, notes
- **utterances**: id, session_id, audio_path, duration_s, hypothesis_raw, hypothesis_normalized, final_text, status (`raw`, `corrected`, `approved_as_is`, `rejected`), language_tags, latency_ms, created_at, reviewed_at
- **lexicon**: id, canonical, variants (JSON), kind (`name`, `term`, `romanized_word`), source (`owner`, `proposed`), approved (bool)
- **dataset_versions**: version, created_at, manifest_path, utterance_count, hours_by_language (JSON), content_hash, parent_version
- **model_versions**: version, base_model, adapter_path, ct2_path, dataset_version, train_config_path, eval_report_path, status (`champion`, `challenger`, `rejected`, `archived`), created_at
- **loop_runs**: id, trigger, started_at, finished_at, steps (JSON), outcome, report_path

### Correction UI

- Lists recent utterances with audio playback, the hypothesis, and an editable text field.
- Actions: save correction, approve as is, reject (noise, unclear, not worth training on).
- Language tags per utterance (multi-select), used for balancing and per-language metrics.
- Keyboard-first: the owner should be able to review an utterance in a few seconds. Correction speed directly limits how fast the model can improve.
- Shows a live spelling check against `romanization.md` and flags non-canonical spellings in the corrected text before saving.

### Eval set

- Recorded and transcribed by the owner in Phase 0, following the spelling guide.
- Four slices: code-switched Hindi-English, code-switched Marathi-English, English, German. Roughly 20 minutes each to start.
- Stored in `data/eval_frozen/` with a manifest and a recorded content hash.

## 7. Improvement plane (multi-agent loop)

### Why agents here

The improvement loop is the part of the system with real orchestration needs: several distinct stages, each with its own tools and permissions, long-running jobs that can fail in messy ways (out-of-memory, bad audio, conversion errors), and a report that a human needs to read. That makes it a reasonable place to practice multi-agent orchestration.

An honest caveat: the same loop could run as a plain script. The deterministic tools are built and run by hand first (Phase 4). The agent layer is added on top (Phase 5) and must never be required for the system to work. If the agents are removed, the scripts still run.

### Trigger

The loop starts when either condition holds: the number of newly reviewed utterances since the last dataset version exceeds a threshold, or the owner triggers it manually. It runs only when dictation is not active.

### Agents

All runtime agents are built with the Claude Agent SDK. The orchestrator delegates to specialist subagents, each with its own system prompt and a restricted tool set. Tools are thin Python wrappers around the deterministic code in `training/`, `eval/`, `store/`, and `registry/`.

**Orchestrator.** Plans the run, delegates to specialists in order, handles retries, decides when to stop, and assembles the final report. Tools: read loop state, invoke specialists, write report. Cannot touch data or models directly.

**Curator.** Builds the next dataset version. Pulls reviewed utterances, runs validation (duration bounds, empty transcripts, audio clipping, transcript-length vs audio-length sanity, spelling-guide compliance), deduplicates, balances languages, and mixes in a replay sample of German and English to prevent forgetting. Tools: query store, run validators, write manifest. Cannot read the eval set.

**Lexicographer.** Reviews recent corrections for recurring names, terms, and spelling variants. Proposes lexicon additions and normalizer rules. Proposals are stored as `approved = false` until the owner accepts them in the UI. Tools: query corrections, write proposals.

**Trainer.** Launches a LoRA fine-tuning job from a config, monitors logs, and responds to known failures (on out-of-memory, retries with smaller batch and more gradient accumulation, up to a limit). Then merges, converts to CTranslate2, and runs the parity check. Tools: check GPU, start job, read logs, merge, convert, parity check.

**Evaluator.** Verifies the eval set hash, runs the challenger and the champion on the frozen set, and computes metrics per language: raw WER, normalized WER, CER, lexicon term accuracy, and latency on a fixed replay of eval audio. Tools: hash check, run eval, write eval report. Cannot touch `training/` or the dataset.

**Gatekeeper.** Applies the promotion rules from `configs/promotion.yaml` to the evaluator's report. The rule itself is code; the agent's job is to call it, explain the outcome in plain language, and highlight the biggest wins and regressions. Until the system has completed several clean cycles, promotion also requires owner approval.

### Promotion rules (initial, in config)

- Overall normalized WER improves by at least a minimum relative margin over the champion.
- No language slice gets worse by more than a small absolute tolerance. German and English are watched specifically for forgetting.
- Lexicon term accuracy does not drop.
- Latency p50 does not worsen beyond a tolerance.
- Parity check passed.

### Run report

Every loop run produces a Markdown report: dataset version and composition, training config and curve summary, per-language metrics vs champion, promotion decision with reasons, failures and retries, and lexicon proposals awaiting approval.

### Guardrails

- Tool-level permissions enforce the golden rules: no agent has a tool that writes to `data/eval_frozen/` or `configs/promotion.yaml`.
- Agents receive metrics, logs, and paths. They never receive raw audio.
- Every agent action is logged to `loop_runs.steps`.
- A run has a step budget and a wall-clock budget. Exceeding either stops the run cleanly with a report.

## 8. Track A: from-scratch study

A separate, offline study under `trackA/`, not part of the product. A small CTC model (convolutional front end plus a few transformer or LSTM layers) trained from scratch on a public English corpus (LibriSpeech train-clean-100 subset) using log-mel features and greedy CTC decoding. Goals: understand features, alignment-free training, and decoding; measure how WER scales with data; and compare against fine-tuned Whisper on the owner's English eval slice. The comparison is the deliverable: it shows, with numbers, why the product uses a pretrained model.

## 9. Observability

- Real-time plane: per-utterance latency, decode time per cycle, buffer length, GPU memory, fallbacks.
- Data plane: reviewed hours per language, correction rate (share of utterances needing edits), review throughput.
- Improvement plane: per-version metrics history, which becomes the project's main result chart.

Correction rate over time is the most honest everyday metric: if the model is improving, the owner should be editing less.

## 10. Risks and what to revisit

- **Inconsistent spelling in corrections.** Mitigated by the spelling guide, UI checks, and the normalizer. Biggest single risk to training quality.
- **Not enough reviewed data.** Correction is tedious. Mitigated by a fast UI and reading prepared scripts to bootstrap.
- **Catastrophic forgetting of German and English.** Mitigated by replay mixing and per-language gates.
- **Overfitting to the eval set** through repeated tuning against it. Mitigation: refresh a second held-out set periodically and report on both; the original frozen set stays as the long-term benchmark.
- **Whisper hallucination on silence or noise.** Mitigated by VAD gating and rejecting utterances below a speech-ratio threshold.
- **Streaming latency vs accuracy trade-off.** Tuned with measurements; a natively streaming architecture is the fallback if Whisper streaming cannot meet targets.

As the project grows, revisit: a second model tier for long-form dictation, an async LLM cleanup pass, and moving training to a larger GPU.