# ADR-002: Language token strategy

Status: accepted

## Context
Whisper conditions on a language token; romanized Hindi/Marathi has no native token.

## Options considered
Use English token for non-German speech; new dedicated token; per-session mode switch.

## Decision
Decode non-German speech with the English token; restrict language detection to English vs German. Training targets use the same convention.

## Consequences
Simple start. Alternatives are evaluated in Phase 4 if code-switched WER stalls.
