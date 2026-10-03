"""Structure and permission logic of the agent layer. These do NOT call a model."""
import asyncio
import inspect
import json

import pytest
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from amanuensis.loop import agents, tools
from amanuensis.loop.run import LoopConfig, LoopRun
from tests.test_loop import env  # noqa: F401  (fixture)

BUILTINS = {"Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch", "NotebookEdit"}


def test_tool_set_is_closed_and_pinned():
    """Adding a tool is a deliberate act: update this list, and re-check it against golden rules 1, 2, 5 and 6."""
    assert set(agents.TOOLS) == {
        "check_trigger", "read_loop_state", "write_report", "build_dataset", "train_challenger", "read_train_log",
        "convert_and_check_parity", "verify_eval_hash", "evaluate", "decide_promotion", "recent_corrections", "propose_lexicon",
    }
    assert set(agents.TOOLS) == {t for ts in agents.ROLE_TOOLS.values() for t in ts}  # no tool without an owner role


def test_no_tool_takes_a_path_threshold_or_audio():
    """Least privilege (rules 1, 5, 6): nothing an agent can pass in could redirect a write or change a threshold."""
    banned = ("path", "file", "dir", "audio", "threshold", "tolerance", "rules", "manifest", "hash")
    for fn in (tools.check_trigger, tools.build_dataset, tools.train_challenger, tools.read_train_log, tools.convert_and_check_parity,
               tools.verify_eval_hash, tools.evaluate, tools.decide_promotion, tools.recent_corrections, tools.propose_lexicon):
        for param in list(inspect.signature(fn).parameters)[1:]:
            assert not any(b in param.lower() for b in banned), f"{fn.__name__}({param})"
    for name, spec in agents.TOOLS.items():
        assert not any(b in k.lower() for k in spec.schema for b in banned), name


def test_agents_have_no_builtin_tools_and_only_their_own():
    defs = agents.build_agents("sonnet")
    assert set(defs) == {"curator", "trainer", "evaluator", "lexicographer", "gatekeeper"}
    for role, d in defs.items():
        assert not set(d.tools) & BUILTINS
        assert d.tools == [agents._mcp(t) for t in agents.ROLE_TOOLS[role]]
        assert d.background is False and d.maxTurns


def test_only_the_gatekeeper_can_decide_and_nobody_can_promote():
    owners = [r for r, ts in agents.ROLE_TOOLS.items() if "decide_promotion" in ts]
    assert owners == ["gatekeeper"]
    assert not any("promot" in t and t != "decide_promotion" for t in agents.TOOLS)  # no promote/approve tool exists
    assert not any("approve" in t for t in agents.TOOLS)


def test_evaluator_alone_touches_the_eval_set_and_curator_cannot_evaluate():
    assert "evaluate" not in agents.ROLE_TOOLS["curator"] and "evaluate" not in agents.ROLE_TOOLS["trainer"]
    assert "train_challenger" not in agents.ROLE_TOOLS["evaluator"]


def ctx_for(agent_id):
    return ToolPermissionContext(tool_use_id="t1", agent_id=agent_id)


def decide(tool, agent_id):
    return asyncio.run(agents.can_use_tool(tool, {}, ctx_for(agent_id)))


def test_orchestrator_cannot_call_specialist_tools_but_subagents_can():
    for t in agents.SPECIALIST_TOOLS:
        assert isinstance(decide(t, None), PermissionResultDeny)
        assert isinstance(decide(t, "sub-1"), PermissionResultAllow)
    assert isinstance(decide("Agent", None), PermissionResultAllow)
    assert isinstance(decide(agents._mcp("write_report"), None), PermissionResultAllow)


def test_everything_unlisted_is_denied():
    for t in ("Bash", "Write", "Edit", "Read", "WebFetch", "mcp__other__tool", "mcp__loop__made_up"):
        assert isinstance(decide(t, "sub-1"), PermissionResultDeny) and isinstance(decide(t, None), PermissionResultDeny)


def test_options_wiring(env):  # noqa: F811
    ctx, _, _, conn = env
    opts = agents.build_options(ctx, LoopRun(conn, ctx.loop_cfg, "t"))
    assert opts.tools == ["Agent"] and opts.setting_sources == [] and opts.can_use_tool is agents.can_use_tool
    assert opts.max_budget_usd == ctx.loop_cfg.agent_max_budget_usd and opts.max_turns
    assert set(opts.allowed_tools) == {"Agent", *agents.ORCHESTRATOR_TOOLS}
    assert not set(opts.allowed_tools) & agents.SPECIALIST_TOOLS  # specialists' tools are never pre-approved
    assert opts.env["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"] == "1" and "loop" in opts.mcp_servers


def test_wrapped_tool_runs_through_the_run_log_and_budget(env):  # noqa: F811
    ctx, _, _, conn = env
    run = LoopRun(conn, ctx.loop_cfg, "t")
    handlers = _handlers(ctx, run)
    out = asyncio.run(handlers["check_trigger"]({"manual": True}))
    payload = json.loads(out["content"][0]["text"])
    assert payload["should_run"] is True and "is_error" not in out
    assert [s["name"] for s in run.steps] == ["check_trigger"]  # agent actions are logged like script steps


def test_wrapped_tool_failure_is_returned_not_raised(env):  # noqa: F811
    ctx, _, _, conn = env
    run = LoopRun(conn, ctx.loop_cfg, "t")
    out = asyncio.run(_handlers(ctx, run)["evaluate"]({"which": "bogus"}))  # no challenger yet / invalid arg
    assert out["is_error"] is True and "error" in json.loads(out["content"][0]["text"])
    assert run.steps[0]["ok"] is False


def test_budget_exhaustion_tells_the_agent_to_stop(env):  # noqa: F811
    ctx, _, _, conn = env
    cfg = LoopConfig(200, 1, 14400, 2, "small", ctx.loop_cfg.reports_dir, 3, 2, "sonnet", 5.0)
    run = LoopRun(conn, cfg, "t")
    h = _handlers(ctx, run)
    asyncio.run(h["read_loop_state"]({}))
    out = asyncio.run(h["read_loop_state"]({}))
    assert out["is_error"] and "write_report" in json.loads(out["content"][0]["text"])["instruction"]


def test_write_report_closes_the_run(env):  # noqa: F811
    ctx, _, _, conn = env
    run = LoopRun(conn, ctx.loop_cfg, "t")
    asyncio.run(_handlers(ctx, run)["write_report"]({"outcome": "test outcome"}))
    row = conn.execute("SELECT outcome, report_path FROM loop_runs WHERE id = ?", (run.id,)).fetchone()
    assert row["outcome"] == "test outcome" and row["report_path"]


def _handlers(ctx, run):
    """The real handlers, as built for the MCP server (same factory the server uses)."""
    import threading

    lock = threading.Lock()
    return {name: agents.make_handler(ctx, run, name, spec, lock) for name, spec in agents.TOOLS.items()}


def test_server_builds_with_every_tool(env):  # noqa: F811
    ctx, _, _, conn = env
    server = agents.build_server(ctx, LoopRun(conn, ctx.loop_cfg, "t"))
    assert server["type"] == "sdk" and server["name"] == "loop"
