"""Optional agent layer over the deterministic loop tools (ADR-007: agents orchestrate, code decides).

Least privilege, enforced in layers so no single one has to be perfect:
  1. Tools: only our in-process MCP tools plus the "Agent" delegation tool. Built-ins (Bash, Read, Write, ...)
     are disabled, so agents cannot touch files, eval data or configs/promotion.yaml at all.
  2. Roles: each AgentDefinition lists only its own tools. can_use_tool additionally refuses specialist tools
     from the orchestrator itself (agent_id is None), so it can only delegate.
  3. Tool code: every tool enforces the golden rules itself and none takes a path, a threshold or raw audio.
     Passing the gate only queues an owner approval: no agent can promote a model.
  4. Budgets: LoopRun step and wall-clock budgets apply to every tool call, plus an API spend cap.
This module was wired against claude-agent-sdk 0.2.x docs; its behavior under a live model is NOT covered by
the unit tests (they check structure and the permission logic).

Usage: uv run python -m amanuensis.loop.agents [--manual]
"""
import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    AgentDefinition, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, create_sdk_mcp_server, tool,
)
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from amanuensis.loop import tools as T
from amanuensis.loop.report import write_report
from amanuensis.loop.run import BudgetExceeded, LoopRun

SERVER = "loop"


def _mcp(name: str) -> str:
    return f"mcp__{SERVER}__{name}"


@dataclass(frozen=True)
class ToolSpec:
    description: str
    schema: dict
    call: Callable[[T.LoopContext, LoopRun, dict], Any]


# The complete, closed set of tools agents can ever call. test_agents pins it.
TOOLS: dict[str, ToolSpec] = {
    "check_trigger": ToolSpec("Report whether a run should start (data threshold, dictation not active).", {"manual": bool},
                              lambda c, r, a: T.check_trigger(c, bool(a.get("manual")))),
    "read_loop_state": ToolSpec("Current run progress: steps taken, budget left, model and dataset versions.", {},
                                lambda c, r, a: T.loop_state(c, r)),
    "write_report": ToolSpec("Write the Markdown run report and close the run. Call once, last.", {"outcome": str},
                             lambda c, r, a: _close(c, r, str(a["outcome"]))),
    "build_dataset": ToolSpec("Validate, deduplicate and balance reviewed utterances into a new dataset version.", {},
                              lambda c, r, a: T.build_dataset(c)),
    "train_challenger": ToolSpec("LoRA-train a challenger on the latest dataset. batch_size/grad_accum 0 = config default. "
                                 "Out-of-memory is retried automatically with a smaller batch.", {"batch_size": int, "grad_accum": int},
                                 lambda c, r, a: T.train_challenger(c, a.get("batch_size") or None, a.get("grad_accum") or None)),
    "read_train_log": ToolSpec("Status, peak VRAM and the last loss values of the challenger's training run.", {"tail": int},
                               lambda c, r, a: T.read_train_log(c, int(a.get("tail") or 10))),
    "convert_and_check_parity": ToolSpec("Merge the adapter, convert to CTranslate2, run the parity check.", {},
                                         lambda c, r, a: T.convert_and_check_parity(c)),
    "verify_eval_hash": ToolSpec("Verify the frozen eval set against its recorded hash. Run before any evaluation.", {},
                                 lambda c, r, a: T.verify_eval_hash(c)),
    "evaluate": ToolSpec("Evaluate 'champion' or 'challenger' on the frozen eval set; returns metrics and a report path.",
                         {"which": str}, lambda c, r, a: T.evaluate(c, str(a["which"]), r.id)),
    "decide_promotion": ToolSpec("Apply the owner's promotion thresholds to the two eval reports. A pass queues owner approval; "
                                 "a fail rejects the challenger. This tool cannot promote.", {},
                                 lambda c, r, a: T.decide_promotion(c)),
    "recent_corrections": ToolSpec("Recent corrections as text pairs (heard vs corrected). No audio.", {"limit": int},
                                   lambda c, r, a: T.recent_corrections(c, int(a.get("limit") or 50))),
    "propose_lexicon": ToolSpec("Propose a lexicon entry. Stored unapproved; only the owner can approve it.",
                                {"canonical": str, "variants": list, "kind": str},
                                lambda c, r, a: T.propose_lexicon(c, str(a["canonical"]), list(a.get("variants") or []), str(a["kind"]))),
}

ROLE_TOOLS: dict[str, list[str]] = {
    "orchestrator": ["check_trigger", "read_loop_state", "write_report"],
    "curator": ["build_dataset"],
    "trainer": ["train_challenger", "read_train_log", "convert_and_check_parity"],
    "evaluator": ["verify_eval_hash", "evaluate"],
    "lexicographer": ["recent_corrections", "propose_lexicon"],
    "gatekeeper": ["decide_promotion"],
}
SPECIALIST_TOOLS = {_mcp(t) for role, ts in ROLE_TOOLS.items() if role != "orchestrator" for t in ts}
ORCHESTRATOR_TOOLS = {_mcp(t) for t in ROLE_TOOLS["orchestrator"]}


def _close(ctx: T.LoopContext, run: LoopRun, outcome: str) -> dict:
    path = write_report(ctx, run, outcome)
    run.finish(outcome, path)
    return {"report_path": str(path)}


SPECIALIST_PROMPTS = {
    "curator": "You are the Curator. Build the next dataset version with build_dataset, then report: version, utterance "
               "count, hours per language group, how many were rejected and why, and any warnings. You cannot see audio or the eval set.",
    "trainer": "You are the Trainer. Call train_challenger (defaults unless the orchestrator asks otherwise), check read_train_log, "
               "then convert_and_check_parity. If a step fails, report the exact error and stop. Do not retry anything else.",
    "evaluator": "You are the Evaluator. Call verify_eval_hash FIRST. If it fails, stop and report the failure. Otherwise evaluate "
                 "'challenger' and then 'champion' and report the per-language normalized WER, term accuracy and p50 latency.",
    "lexicographer": "You are the Lexicographer. Read recent_corrections, find names, terms or spellings the owner corrected repeatedly, "
                     "and propose_lexicon entries. Proposals are unapproved until the owner accepts them. Propose only what the data shows.",
    "gatekeeper": "You are the Gatekeeper. Call decide_promotion once. Explain the outcome in plain language, highlight the biggest "
                  "win and the biggest regression, and say clearly that a passing model still needs the owner's approval.",
}
SPECIALIST_DESCRIPTIONS = {
    "curator": "Builds the next dataset version from reviewed utterances.",
    "trainer": "Trains the challenger with LoRA, merges, converts and checks parity.",
    "evaluator": "Verifies the eval hash and evaluates champion and challenger.",
    "lexicographer": "Proposes lexicon additions from recent corrections.",
    "gatekeeper": "Applies the owner's promotion thresholds and explains the result.",
}
ORCHESTRATOR_PROMPT = (
    "You orchestrate one run of the Amanuensis improvement loop. Work through these steps in order, delegating each to the named "
    "specialist with the Agent tool: 1) check_trigger yourself; stop if it says not to run. 2) evaluator: verify the eval hash. "
    "3) curator. 4) trainer. 5) evaluator: evaluate challenger and champion. 6) gatekeeper. 7) lexicographer (optional). "
    "8) write_report with a one-line outcome. If any step fails, do not improvise around it: stop, and write_report with the failure. "
    "You cannot promote a model: only the owner can, in the correction UI. You never see audio. Respect the step budget; "
    "read_loop_state shows what is left."
)


def build_agents(model: str) -> dict[str, AgentDefinition]:
    return {
        role: AgentDefinition(
            description=SPECIALIST_DESCRIPTIONS[role], prompt=SPECIALIST_PROMPTS[role],
            tools=[_mcp(t) for t in ROLE_TOOLS[role]], model=model, maxTurns=12, background=False,
        )
        for role in SPECIALIST_PROMPTS
    }


async def can_use_tool(tool_name: str, input_data: dict, context: ToolPermissionContext):
    """Anything not listed is denied. The orchestrator (agent_id None) may not call specialist tools."""
    if tool_name in SPECIALIST_TOOLS:
        if context.agent_id is None:
            return PermissionResultDeny(message="The orchestrator must delegate this to a specialist agent.")
        return PermissionResultAllow(updated_input=input_data)
    if tool_name == "Agent" or tool_name in ORCHESTRATOR_TOOLS:
        return PermissionResultAllow(updated_input=input_data)
    return PermissionResultDeny(message=f"{tool_name} is not available in the improvement loop.")


def _text(obj: Any, error: bool = False) -> dict:
    out = {"content": [{"type": "text", "text": json.dumps(obj, default=str)}]}
    return out | {"is_error": True} if error else out


def make_handler(ctx: T.LoopContext, run: LoopRun, name: str, spec: ToolSpec, lock: threading.Lock):
    """The async MCP handler for one tool. Calls are serialized (one GPU) and go through LoopRun.step
    for the budgets and the audit log."""

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        def work():
            with lock:
                return run.step(name, lambda: spec.call(ctx, run, args))

        try:
            return _text(await asyncio.to_thread(work))
        except BudgetExceeded as e:
            return _text({"error": str(e), "instruction": "stop now and write_report"}, error=True)
        except Exception as e:  # the failure is already logged by run.step; the agent sees the message
            return _text({"error": f"{type(e).__name__}: {e}"}, error=True)

    return handler


def build_server(ctx: T.LoopContext, run: LoopRun):
    """In-process MCP server exposing exactly TOOLS."""
    lock = threading.Lock()
    return create_sdk_mcp_server(
        SERVER, tools=[tool(n, s.description, s.schema)(make_handler(ctx, run, n, s, lock)) for n, s in TOOLS.items()]
    )


def build_options(ctx: T.LoopContext, run: LoopRun) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=ORCHESTRATOR_PROMPT,
        tools=["Agent"],  # no Bash/Read/Write/Edit: agents act only through our tools
        allowed_tools=["Agent", *sorted(ORCHESTRATOR_TOOLS)],  # specialist tools go through can_use_tool
        can_use_tool=can_use_tool,
        mcp_servers={SERVER: build_server(ctx, run)},
        agents=build_agents(ctx.loop_cfg.agent_model),
        model=ctx.loop_cfg.agent_model,
        max_turns=60,
        max_budget_usd=ctx.loop_cfg.agent_max_budget_usd,
        setting_sources=[],  # do not load the dev-time CLAUDE.md or user settings into runtime agents
        env={"CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "1", "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "1"},
    )


async def run_agent_loop(ctx: T.LoopContext, trigger: str = "manual") -> dict:
    run = LoopRun(ctx.conn, ctx.loop_cfg, f"agents:{trigger}")
    ctx.state.clear()
    result: dict = {"run_id": run.id}
    try:
        async with ClaudeSDKClient(build_options(ctx, run)) as client:
            await client.query(f"Run the improvement loop now. Trigger: {trigger}.")
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    result |= {"is_error": message.is_error, "subtype": message.subtype, "cost_usd": message.total_cost_usd,
                               "turns": message.num_turns, "summary": message.result}
    finally:
        if not run.conn.execute("SELECT finished_at FROM loop_runs WHERE id = ?", (run.id,)).fetchone()[0]:
            # The orchestrator did not close the run (budget, error, crash): close it ourselves so a report always exists.
            outcome = f"agent run ended without a report ({result.get('subtype', 'no result')})"
            run.finish(outcome, write_report(ctx, run, outcome))
    return result


if __name__ == "__main__":
    import sys

    from amanuensis.loop.pipeline import build_context

    print(json.dumps(asyncio.run(run_agent_loop(build_context(), "manual" if "--manual" in sys.argv else "threshold")), indent=2, default=str))
