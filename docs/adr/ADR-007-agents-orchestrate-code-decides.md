# ADR-007: Agents orchestrate, code decides

Status: accepted

## Context
The improvement loop needs orchestration but pass/fail must be trustworthy.

## Options considered
Agents decide everything; scripts only; agents orchestrate with deterministic gates.

## Decision
Agents plan, call tools, recover from failures and report. Validity rules and promotion thresholds are deterministic code with config agents cannot edit.

## Consequences
The system works without agents; agents never weaken a gate.
