# ADR-008: Owner-approved promotion and least-privilege agent tools

Status: accepted

## Context
ADR-007 says agents orchestrate and code decides. Phase 5 had to turn that into mechanisms: what an agent can
call, who can promote a model, and what happens when a run goes wrong. A code review of the first version found
that a passing gate could still go stale or be queued twice, and that a latency of NaN passed the latency gate.

## Options considered
1. Give agents the deterministic functions directly (paths, thresholds as arguments) and trust the prompts.
2. Let a passing gate promote automatically.
3. A closed set of narrow tools, a gate that only queues an owner approval, and budgets and an audit log under
   every call.

## Decision
Option 3.
- **Closed tool set** (`loop/agents.py: TOOLS`, pinned by a test). No tool takes a path, a threshold, or a file
  to write, and none returns audio or audio paths. Built-in tools (Bash, Read, Write, Edit) are disabled for
  runtime agents.
- **Roles**: each specialist lists only its own tools. `can_use_tool` additionally refuses specialist tools from
  the orchestrator itself, so it can only delegate.
- **The gate only queues.** `decide_promotion` applies the owner's thresholds (`configs/promotion.yaml`, which
  ships all-null and refuses to load until the owner fills it). A pass creates a pending approval that records
  the champion it was compared against; a fail rejects the challenger. Only the owner, in the correction UI, can
  promote. `models.promote` itself requires a passing decision object.
- **Fail closed**: unmeasured or NaN latency fails the gate; an approval is refused if the champion changed since
  the request; promotion and the request status commit in one transaction; a model is only trained from a
  dataset manifest the builder registered and that is unchanged.
- **Every call is a `LoopRun.step`**: step and wall-clock budgets, an audit trail in `loop_runs.steps`, and a
  Markdown report, whether a script or an agent drives it. The plain-script pipeline is the reference and needs
  no agents.

## Consequences
- Safety does not depend on a model following its prompt: the worst an agent can do is waste its budget or write
  an unapproved lexicon proposal.
- The agent layer is not exercised against a live model by the unit tests (structure and permission logic only).
  Whether `can_use_tool` is consulted for every subagent call must be confirmed in a live run.
- The Lexicographer receives correction text (not audio) through the API. That is a privacy choice for the owner.
- Owner approval adds a manual step to every promotion, by design, until several clean cycles have passed.
