# ADR-004: Train in Hugging Face, serve in CTranslate2

Status: accepted

## Context
LoRA training tooling is strongest in transformers+PEFT; fast inference is in faster-whisper (CTranslate2).

## Options considered
Train and serve in HF; train in HF and serve in CT2.

## Decision
Train with transformers+PEFT, merge the adapter, convert to CT2, and run a parity check against the merged HF model before use.

## Consequences
Extra conversion step and a mandatory parity gate.
