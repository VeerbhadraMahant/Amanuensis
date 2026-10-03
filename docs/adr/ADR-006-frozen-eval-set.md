# ADR-006: Frozen, hash-verified eval set

Status: accepted

## Context
Without a fixed test set no result is meaningful and overfitting to it is easy.

## Options considered
Rolling eval; frozen eval; frozen eval plus periodic second held-out set.

## Decision
Owner-recorded, hand-transcribed set in data/eval_frozen/, never trained on, manifest hash verified before every evaluation.

## Consequences
Eval is trustworthy but small; a second held-out set is added later (Phase 6).
