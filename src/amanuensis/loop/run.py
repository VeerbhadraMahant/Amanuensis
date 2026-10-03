"""One loop run: logged steps, step budget, wall-clock budget (docs/systemdesign.md section 7, Guardrails).

Every tool call, whether made by a script or by an agent, goes through LoopRun.step, so the budgets and the
audit trail hold no matter who is driving.
"""
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class LoopConfig:
    new_reviewed_threshold: int
    step_budget: int
    wall_clock_budget_s: float
    oom_retries: int
    baseline_model: str
    reports_dir: Path
    latency_clips: int
    parity_clips: int
    agent_model: str
    agent_max_budget_usd: float


def load_loop_config(path: Path = Path("configs/loop.yaml")) -> LoopConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LoopConfig(**{f.name: Path(raw[f.name]) if f.name == "reports_dir" else raw[f.name] for f in fields(LoopConfig)})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _brief(result: Any, limit: int = 600) -> str:
    try:
        text = json.dumps(result, default=str)
    except (TypeError, ValueError):
        text = repr(result)
    return text if len(text) <= limit else text[:limit] + "..."


class LoopRun:
    def __init__(self, conn: sqlite3.Connection, cfg: LoopConfig, trigger: str, clock: Callable[[], float] = time.monotonic):
        self.conn, self.cfg, self.trigger, self._clock = conn, cfg, trigger, clock
        self._t0 = clock()
        self.steps: list[dict] = []
        cur = conn.execute("INSERT INTO loop_runs (trigger, started_at, steps) VALUES (?, ?, '[]')", (trigger, _now()))
        conn.commit()
        self.id = cur.lastrowid

    def step(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run fn under the budgets and record it. A failing step is recorded and re-raised."""
        if len(self.steps) >= self.cfg.step_budget:
            raise BudgetExceeded(f"step budget of {self.cfg.step_budget} reached")
        if self._clock() - self._t0 >= self.cfg.wall_clock_budget_s:
            raise BudgetExceeded(f"wall-clock budget of {self.cfg.wall_clock_budget_s:.0f}s reached")
        t = self._clock()
        record: dict = {"name": name, "at": _now()}
        try:
            result = fn()
            record |= {"ok": True, "result": _brief(result)}
            return result
        except Exception as e:
            record |= {"ok": False, "error": f"{type(e).__name__}: {e}"}
            raise
        finally:
            record["seconds"] = round(self._clock() - t, 2)
            self.steps.append(record)
            self.conn.execute("UPDATE loop_runs SET steps = ? WHERE id = ?", (json.dumps(self.steps), self.id))
            self.conn.commit()

    def finish(self, outcome: str, report_path: Path | None) -> None:
        self.conn.execute(
            "UPDATE loop_runs SET finished_at = ?, outcome = ?, report_path = ? WHERE id = ?",
            (_now(), outcome, str(report_path) if report_path else None, self.id),
        )
        self.conn.commit()
