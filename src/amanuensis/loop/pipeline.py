"""The improvement loop as a plain script. It needs no agents: they are an optional layer on top (ADR-007).

Usage: uv run python -m amanuensis.loop.pipeline [--manual]
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

from amanuensis.eval.manifest import ManifestHashError
from amanuensis.eval.promotion import PromotionConfigError
from amanuensis.loop import tools
from amanuensis.loop.report import write_report
from amanuensis.loop.run import BudgetExceeded, LoopRun
from amanuensis.logging import log
from amanuensis.registry import approvals, models
from amanuensis.training.dataset import EvalLeakError
from amanuensis.training.guard import TrainingBlocked


@dataclass(frozen=True)
class RunResult:
    run_id: int
    outcome: str
    report_path: Path | None


def run_cycle(ctx: tools.LoopContext, trigger: str = "manual") -> RunResult:
    run = LoopRun(ctx.conn, ctx.loop_cfg, trigger)
    ctx.state.clear()
    outcome = "failed"
    try:
        t = run.step("check_trigger", lambda: tools.check_trigger(ctx, trigger == "manual"))
        if not t["should_run"]:
            outcome = f"skipped: {t['reason']}"
            return _finish(ctx, run, outcome)
        run.step("verify_eval_hash", lambda: tools.verify_eval_hash(ctx))  # fail fast on a tampered eval set
        built = run.step("build_dataset", lambda: tools.build_dataset(ctx))
        if built["utterance_count"] == 0:
            return _finish(ctx, run, "stopped: no usable reviewed data")
        run.step("train", lambda: tools.train_challenger(ctx))
        try:
            run.step("convert_and_check_parity", lambda: tools.convert_and_check_parity(ctx))
        except BudgetExceeded:
            raise
        except Exception:
            _reject_challenger(ctx)
            raise
        run.step("evaluate_challenger", lambda: tools.evaluate(ctx, "challenger", run.id))
        run.step("evaluate_champion", lambda: tools.evaluate(ctx, "champion", run.id))
        decision = run.step("decide_promotion", lambda: tools.decide_promotion(ctx))
        outcome = "awaiting_owner_approval" if decision["passed"] else "rejected_by_gate"
    except BudgetExceeded as e:
        _reject_challenger(ctx)
        outcome = f"stopped: {e}"
    except ManifestHashError:
        outcome = "aborted: frozen eval set failed its hash check"
    except EvalLeakError as e:
        outcome = f"aborted: eval leak ({e})"
    except TrainingBlocked as e:
        outcome = f"blocked: {e}"
    except PromotionConfigError as e:
        _reject_challenger(ctx)
        outcome = f"blocked: {e}"
    except Exception as e:
        _reject_challenger(ctx)
        outcome = f"failed: {type(e).__name__}: {e}"
    return _finish(ctx, run, outcome)


def _reject_challenger(ctx: tools.LoopContext) -> None:
    """A challenger from a run that did not finish must not linger as a candidate.
    One already waiting for the owner's decision is left alone."""
    version = ctx.state.get("challenger")
    if not version or models.get(ctx.conn, version)["status"] != "challenger":
        return
    if any(p["model_version"] == version for p in approvals.pending(ctx.conn)):
        return
    models.reject(ctx.conn, version)


def _finish(ctx: tools.LoopContext, run: LoopRun, outcome: str) -> RunResult:
    report = write_report(ctx, run, outcome)
    run.finish(outcome, report)
    log("loop_run_finished", run_id=run.id, outcome=outcome, report=str(report))
    return RunResult(run.id, outcome, report)


def build_context() -> tools.LoopContext:
    import yaml

    from amanuensis.config import load_paths, load_streaming, load_variants
    from amanuensis.loop.run import load_loop_config
    from amanuensis.store import db
    from amanuensis.training.dataset import load_dataset_config
    from amanuensis.training.train_lora import load_train_config
    from amanuensis.training.validators import load_validation

    paths, tcfg_path = load_paths(), Path("configs/training.yaml")
    return tools.LoopContext(
        conn=db.connect(paths.db_path, check_same_thread=False), paths=paths, sc=load_streaming(),
        train_cfg=load_train_config(tcfg_path), dataset_cfg=load_dataset_config(), vcfg=load_validation(),
        loop_cfg=load_loop_config(), eval_cfg=yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")),
        variants=load_variants(paths.variants_file), training_config_path=tcfg_path,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manual", action="store_true", help="run even if the new-data threshold is not reached")
    a = p.parse_args()
    r = run_cycle(build_context(), "manual" if a.manual else "threshold")
    print(f"{r.outcome}\nreport: {r.report_path}")
