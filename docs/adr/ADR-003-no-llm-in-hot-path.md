# ADR-003: No LLM in the hot path

Status: accepted

## Context
Dictation must be fast, offline, and deterministic.

## Options considered
Inline LLM cleanup; async LLM cleanup; none.

## Decision
Capture, VAD, ASR, normalization and output are local and deterministic. Any LLM cleanup runs asynchronously on finalized utterances.

## Consequences
Predictable latency and privacy. Cleanup quality is deferred to Phase 6.
