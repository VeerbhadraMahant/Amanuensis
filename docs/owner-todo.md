# What needs you

Everything below is blocked on a decision, a recording, or a live run that only you can do. All the code around
them is built and tested (see `docs/plan.md` for the per-task status). Roughly in the order you will hit them.

## 1. Content only you can write (blocks Phases 0, 3, 4 on real data)

1. **`docs/romanization.md`**: canonical spellings for your 150 to 200 most frequent Hindi and Marathi words, rules
   for long vowels, nasals and English loanwords. It is a stub. Nothing was invented in its place.
   Then derive `configs/spelling_variants.yaml` (a flat `variant: canonical` map, lowercase). Until it exists the
   spelling check and the normalizer's variant mapping do nothing.
2. **Record and hand-transcribe the frozen eval set**: about 20 minutes each of Hindi-English, Marathi-English,
   English and German, in natural speech, following the spelling guide. Put it in `data/eval_frozen/` with a
   `manifest.json` (a list of `{"audio", "text", "language"}`), then freeze it once:
   `uv run python -m amanuensis.eval.manifest data/eval_frozen` (writes `manifest.sha256`, refuses to overwrite).
3. **`configs/promotion.yaml`**: every threshold is deliberately `null`, and promotion refuses to run until you set
   them: minimum relative WER improvement, maximum regression per language (and optional stricter German/English
   tolerances), maximum lexicon accuracy drop, maximum p50 latency worsening, whether parity is required.
4. **Confirm the language-tag convention.** `configs/review.yaml` uses `en, de, hi, mr` as a placeholder for the
   per-utterance tags, but the eval slices are named by the manifest's `language` field (for example `hi-en`,
   `mr-en`). The promotion gate compares eval slices; training balances by tags. Decide the names and whether they
   should match.
5. Optionally a **second held-out eval set** (template in `configs/eval.yaml`), and a **lexicon** of your names and
   terms (Lexicon page in the UI).

## 2. Things to run once you have the above

1. **Baseline** (Phase 0 exit): `uv run python -m amanuensis.eval.baseline`. Records WER per language for Whisper
   small, medium, large-v3-turbo. It does not record VRAM yet. Then decide which model to build around
   (`configs/streaming.yaml: model`).
2. **Live dictation**: `uv run python -m amanuensis.live` (add `--inject` to type into the focused window). Only
   synthetic speech and an idle microphone were tested, so confirm the overlay, latency and accuracy with your voice.
   Streaming was measured on one synthetic clip: first partial p50 0.98 s, commit p50 1.28 s.
3. **Use the correction UI**: `uv run python -m amanuensis.ui.app`, then http://127.0.0.1:8765. The page logic was
   only syntax-checked and its API tested; the Chrome extension was not connected, so nobody has clicked through it.
4. **The habit** (Phase 2 onward): 15 to 30 minutes of dictation a day and review of the day's utterances.
5. **First real fine-tuning cycles** (Phase 4) once you have roughly 2 to 3 hours of reviewed speech:
   `uv run python -m amanuensis.loop.pipeline --manual`. Whisper medium and the ADR-002 language-token question wait
   for this. The machinery was only run on 24 synthetic clips for 6 steps (no WER change, as expected).
6. **A live agent run** (Phase 5): `uv run python -m amanuensis.loop.agents --manual`. It spends API money (capped by
   `agent_max_budget_usd`, default 5.0) and has never run against a model. Check two things in the first run: that the
   permission callback really blocks the orchestrator from specialist tools, and that each specialist works alone.

## 3. Decisions

* **Privacy of correction text.** The Lexicographer agent reads up to 200 recent corrections (text, never audio) and
  those go to the Claude API. Audio never does. If you do not want text to leave the machine, do not run the agent
  layer; the script pipeline needs no agents and proposes no lexicon entries.
* **Async LLM cleanup of finished utterances** (Phase 6) was not built: it is an API cost and privacy choice, and it
  should be measured separately.
* **ADR-002 language detection** (English vs German restricted detection) is not implemented. The language is fixed
  to English in `configs/streaming.yaml`, so German speech will be transcribed with the English token until you decide.
* **Untuned defaults** to look at once real data exists: `configs/validation.yaml`, `dataset.yaml`, `training.yaml`
  (learning rate, LoRA rank), `loop.yaml` (`new_reviewed_threshold: 200`).
* **Track A** stopped at 10 h (72.8% WER) and 25 h (56.6%) versus Whisper small at 3.7% on the same utterances. The 50 h and 100 h points need the rest of LibriSpeech and many GPU hours
  (commands in `trackA/README.md`). The write-up or blog post (Phase 6) needs your real results.

## 4. Known limits

* The agent layer is wired against the Agent SDK docs but untested live; the unit tests cover structure and the
  permission logic only.
* `bitsandbytes` 8-bit training is verified on this GPU for Whisper small (2.9 GB peak at batch 4). Medium is
  untested.
* Windows only for text injection. The training stack lives in the optional `train` dependency group
  (`uv sync --group train`).
