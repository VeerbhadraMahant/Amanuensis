# ADR-001: Single Latin-script output space

Status: accepted

## Context
Four languages (English, German, Hindi, Marathi) must be output together, with Hindi/Marathi code-switched.

## Options considered
Devanagari output for Hindi/Marathi; per-language routing; one shared Latin script.

## Decision
All output is Latin script; Hindi and Marathi romanized per docs/romanization.md.

## Consequences
No per-language routing and code-switching is plain seq2seq. Off-the-shelf models do not romanize reliably, so fine-tuning is required and spelling must be standardized by the owner.
