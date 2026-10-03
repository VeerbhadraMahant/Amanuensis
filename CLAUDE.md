# CLAUDE.md: Amanuensis

Amanuensis is a personal, real-time, multilingual speech-to-text system that adapts to one speaker (the owner) over time. It transcribes English, German, and code-switched Hindi and Marathi, and writes everything in the Latin script (romanized Hindi and Marathi, e.g. "kya kar raha hai"). It improves itself through a correction loop: the owner corrects transcripts, and an offline multi-agent pipeline curates that data, fine-tunes the model, evaluates it, and promotes it only if it is measurably better.

This is a learning and portfolio project. Clarity, measurability, and honest evaluation matter more than feature count.

Read `docs/systemdesign.md` for architecture and `docs/plan.md` for the current phase before starting any non-trivial task.

## Golden rules (never violate)

1. **The eval set is frozen and sacred.** Nothing in `data/eval_frozen/` may ever enter a training manifest, be edited, or be regenerated. Its manifest is hash-verified before every evaluation. If a task seems to require changing it, stop and ask the owner.
2. **Only reviewed data trains the model.** An utterance enters training only if the owner explicitly corrected it or explicitly marked it correct. Never train on raw model output. That is how a self-improving system becomes self-degrading.
3. **No LLM or network call in the real-time hot path.** Audio capture, VAD, ASR, normalization, and output injection must run locally and deterministically. LLM agents live only in the offline improvement loop.
4. **`docs/romanization.md` is the source of truth for spelling.** All training targets, normalizers, and lexicon entries follow it. If you find a conflict, flag it instead of picking a spelling.
5. **Promotion is gated by numbers, not judgment.** A new model replaces the champion only if it passes the thresholds in `configs/promotion.yaml`. Agents may not edit that file or the eval data.
6. **Audio never goes to git or to any remote API.** `data/` and `models/` are gitignored. Agents in the improvement loop receive metrics, logs, and file paths, never raw audio.

## How to work

Adapted from the Karpathy-inspired guidelines (multica-ai/andrej-karpathy-skills, MIT), merged with this project's needs. They favor caution over speed. If they conflict with a golden rule, the golden rule wins.

### 1. Think before coding

- State your assumptions before implementing. If a request has more than one reasonable reading, list the readings and ask which one is meant instead of silently picking one.
- If something is unclear, stop and name exactly what is unclear. Do not paper over confusion with plausible-looking code.
- If a simpler approach exists than the one requested, say so. Push back when a request conflicts with `docs/systemdesign.md`, the current phase in `docs/plan.md`, or the golden rules.
- For anything touching more than one top-level module, write a short plan and wait for approval.
- Project-specific: never guess a romanized spelling, a language-tag convention, or a promotion threshold. Ask.

### 2. Simplicity first

- Write the minimum code that solves the current task. No features, options, or abstractions that the task did not ask for.
- No abstraction layers for code used in one place. No error handling for situations that cannot occur.
- If a solution comes out several times longer than it needs to be, rewrite it shorter before presenting it.
- Build only the current phase. The architecture in `systemdesign.md` describes where the project is going, not what to build today. Do not scaffold the improvement loop while working on Phase 1, and do not add config knobs for values nobody tunes yet.
- Test: would an experienced engineer reviewing this diff call it overcomplicated? If yes, simplify.

### 3. Surgical changes

- Touch only what the task requires. Every changed line should trace back to the request.
- Do not reformat, rename, re-comment, or refactor neighboring code that works, even if you would have written it differently. Match the existing style.
- If you notice unrelated problems (dead code, a bug elsewhere, a questionable design), mention them in your summary. Do not fix them unasked.
- Clean up after yourself: remove imports, variables, and helpers that your own change made unused. Leave pre-existing ones alone unless asked.
- Project-specific: never modify files under `data/eval_frozen/`, `configs/promotion.yaml`, or `docs/romanization.md` as a side effect of another task.

### 4. Goal-driven execution

- Turn every task into a verifiable goal before starting. Examples:
  - "Fix the streaming bug" becomes "add a replay test that reproduces it, then make it pass."
  - "Make the normalizer handle X" becomes "add test cases for X from the spelling guide, then make them pass."
  - "Improve latency" becomes "record p50/p95 commit latency in replay mode before and after; the change must improve one without hurting WER."
  - "Refactor Y" becomes "tests pass before and after, with no behavior change."
- For multi-step tasks, state a brief plan where each step has its own check.
- Loop until the check passes. Do not report a task done on the basis that the code looks right.
- Any claimed improvement in accuracy or latency must come with the numbers, measured on the frozen eval set or in replay mode.

**These guidelines are working if:** diffs contain only what was asked, fewer changes get rewritten for being overbuilt, and clarifying questions arrive before implementation rather than after a mistake.

## Stack

- Python 3.11, managed with `uv`
- Inference: faster-whisper (CTranslate2), int8 or int8_float16 on GPU
- Training: Hugging Face transformers + PEFT (LoRA), bitsandbytes for 8-bit loading
- VAD: Silero VAD
- Audio I/O: sounddevice
- Metrics: jiwer (WER, CER) plus our own normalizer
- Storage: SQLite for metadata, filesystem for audio (16 kHz mono WAV)
- Correction UI: FastAPI backend with a minimal local web frontend
- Improvement loop agents: Claude Agent SDK (Python)
- Config: YAML in `configs/`, loaded into typed dataclasses
- Tests: pytest

## Hardware constraints

- Local GPU has 8 GB VRAM. Dictation and training cannot run at the same time on it.
- The trainer must check GPU memory before starting and refuse to run if the dictation process is active.
- Training code must be machine-agnostic (paths and device from config) so it can run on a larger GPU if one is available.
- Default model for iteration: Whisper small. Target model: Whisper medium with LoRA. Larger models only for inference experiments, decided by Phase 0 benchmarks.

## Repository layout

```
amanuensis/
  CLAUDE.md
  docs/
    systemdesign.md
    plan.md
    romanization.md        # spelling guide, owned by the human
    adr/                   # architecture decision records
  .claude/agents/          # Claude Code dev subagents (build-time)
  configs/                 # all tunables: streaming, training, promotion, paths
  src/amanuensis/
    audio/                 # capture, ring buffer, VAD segmentation
    asr/                   # engine wrapper, streaming + local agreement
    text/                  # normalizer, lexicon, prompt biasing
    output/                # text injection adapters, overlay
    store/                 # SQLite schema, session logging, dataset access
    ui/                    # correction app
    training/              # manifest building, LoRA training, merge, CT2 convert
    eval/                  # metrics, frozen-set runner, reports
    registry/              # model versions, champion/challenger
    loop/                  # improvement loop: orchestrator, agent defs, tools
  trackA/                  # from-scratch CTC study (independent of product)
  tests/
  data/                    # gitignored
  models/                  # gitignored
```

## Conventions

- Type hints everywhere. Small pure functions where possible, especially in `text/` and `eval/`.
- Paths, model names, and any threshold that is tuned or used by evaluation or promotion live in `configs/`. Plain internal constants stay in code until someone actually needs to tune them.
- Every module in `eval/` and `text/` has unit tests. The normalizer has a test file of real spelling variants from `romanization.md`.
- Log with structured logging (JSON lines) so the improvement loop agents can read logs.
- Significant design choices get an ADR in `docs/adr/` (context, options, decision, consequences).
- Do not add a dependency without stating why the standard library or an existing dependency is not enough.
- After finishing a task from `docs/plan.md`, tick its checkbox and add a one-line note on what was learned or changed.

## Dev subagents (`.claude/agents/`)

Use these for focused work and for parallel tasks. Each has the minimum tools it needs.

| Agent | Owns | Tools | Notes |
|---|---|---|---|
| `realtime-audio-engineer` | `audio/`, `asr/`, `output/` | Read, Edit, Write, Bash, Grep, Glob | Latency is its primary metric. Must report p50/p95 latency after changes. |
| `ml-training-engineer` | `training/`, `registry/` | Read, Edit, Write, Bash, Grep, Glob | Must respect VRAM limits and log every run's config. |
| `eval-engineer` | `eval/`, `text/normalizer` | Read, Edit, Write, Bash, Grep, Glob | May not edit `training/`. Keeps evaluation independent of the thing being evaluated. |
| `code-reviewer` | none (read-only) | Read, Grep, Glob, Bash (git diff only) | Reviews every change against the golden rules and the "How to work" principles, flagging out-of-scope edits and overbuilt code. Run before every commit. |
| `test-runner` | `tests/` | Read, Edit, Bash, Grep, Glob | Runs tests, fixes failures without weakening test intent. |

When to delegate: give a subagent a task when it is self-contained and its output can be summarized (a module, a test suite, a review). Keep cross-cutting design decisions in the main session.

Parallel pattern to practice: after a feature lands, run `code-reviewer` and `test-runner` in parallel, then fix issues in the main session.