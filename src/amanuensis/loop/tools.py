"""Deterministic loop tools: thin, narrow wrappers over training/, eval/, store/ and registry/.

Used directly by the script pipeline (loop/pipeline.py) and, wrapped, by the agents (loop/agents.py).
Design rules (golden rules 1, 2, 5, 6):
  * No tool takes a path to eval data, a promotion threshold, or a file to write.
  * No tool returns raw audio or audio file paths: only counts, metrics, text and report paths.
  * Passing the promotion gate only queues an owner approval; nothing here can promote a model.
"""
import dataclasses
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from amanuensis import process_lock
from amanuensis.config import PathsConfig, StreamingConfig
from amanuensis.eval import manifest as eval_manifest
from amanuensis.eval.promotion import PromotionRules, decide, load_rules
from amanuensis.eval.runner import evaluate_model
from amanuensis.loop.run import LoopConfig
from amanuensis.registry import approvals, models
from amanuensis.store import lexicon
from amanuensis.training import dataset as ds
from amanuensis.training import guard
from amanuensis.training.merge_convert import merge_and_convert
from amanuensis.training.parity import parity_check
from amanuensis.training.train_lora import TrainConfig, train
from amanuensis.training.validators import ValidationConfig


@dataclass
class LoopContext:
    conn: object
    paths: PathsConfig
    sc: StreamingConfig
    train_cfg: TrainConfig
    dataset_cfg: ds.DatasetConfig
    vcfg: ValidationConfig
    loop_cfg: LoopConfig
    eval_cfg: dict
    variants: dict[str, str]
    training_config_path: Path
    # Injectable so failure-injection tests do not need a GPU:
    rules_loader: Callable[[], PromotionRules] = load_rules
    trainer: Callable = train
    converter: Callable = merge_and_convert
    parity: Callable = parity_check
    evaluator: Callable = evaluate_model
    state: dict = field(default_factory=dict)  # run-scoped: dataset_version, challenger, reports, decision


def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


def check_trigger(ctx: LoopContext, manual: bool) -> dict:
    last = ctx.conn.execute("SELECT created_at FROM dataset_versions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    since = last["created_at"] if last else ""
    new = ctx.conn.execute(
        f"SELECT COUNT(*) FROM utterances WHERE {ds.REVIEWED_SQL} AND reviewed_at > ?", (since,)
    ).fetchone()[0]
    dictating = process_lock.is_active(ctx.paths.lock_file)
    if dictating:
        reason = "dictation is running"
    elif manual:
        reason = "manual trigger"
    elif new >= ctx.loop_cfg.new_reviewed_threshold:
        reason = f"{new} newly reviewed utterances (threshold {ctx.loop_cfg.new_reviewed_threshold})"
    else:
        reason = f"only {new} newly reviewed utterances (threshold {ctx.loop_cfg.new_reviewed_threshold})"
    ok = not dictating and (manual or new >= ctx.loop_cfg.new_reviewed_threshold)
    return {"should_run": ok, "reason": reason, "new_reviewed": new, "dictation_active": dictating}


def verify_eval_hash(ctx: LoopContext) -> dict:
    root, mpath = Path(ctx.eval_cfg["eval_dir"]), Path(ctx.eval_cfg["manifest"])
    eval_manifest.verify(mpath, root, Path(ctx.eval_cfg["manifest_hash_file"]).read_text().strip())
    return {"ok": True}


def build_dataset(ctx: LoopContext) -> dict:
    root, mpath = Path(ctx.eval_cfg["eval_dir"]), Path(ctx.eval_cfg["manifest"])
    res = ds.build_dataset(ctx.conn, ctx.paths.audio_dir, ctx.dataset_cfg, ctx.vcfg, ctx.variants, ds.eval_audio_hashes(mpath, root), ds.eval_texts(mpath))
    ctx.state["dataset_version"] = res.version
    ctx.state["dataset"] = {"version": res.version, "hours_by_group": res.hours_by_group, "manifest": str(res.manifest_path)}
    return {
        "version": res.version, "utterance_count": res.utterance_count, "hours_by_group": res.hours_by_group,
        "rejected_count": len(res.rejected), "rejected_reasons": dict(Counter(r.split(" (")[0] for _, rs in res.rejected for r in rs)),
        "deduped": res.deduped, "capped": res.capped, "warnings": res.warnings,
    }


def train_challenger(ctx: LoopContext, batch_size: int | None = None, grad_accum: int | None = None) -> dict:
    """Train on the current dataset. Out-of-memory is retried with half the batch and double accumulation."""
    version = ctx.state["dataset_version"]
    manifest = Path(ctx.conn.execute("SELECT manifest_path FROM dataset_versions WHERE version = ?", (version,)).fetchone()[0])
    ds.verify_registered(ctx.conn, manifest)  # never train on a manifest the builder did not write
    cfg = dataclasses.replace(ctx.train_cfg, batch_size=batch_size or ctx.train_cfg.batch_size,
                              grad_accum=grad_accum or ctx.train_cfg.grad_accum)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for attempt in range(ctx.loop_cfg.oom_retries + 1):
        name = f"{version}-{stamp}-a{attempt}"
        try:
            run_dir = ctx.trainer(cfg, manifest, ctx.paths.audio_dir, name, ctx.paths.lock_file, ctx.training_config_path)
            break
        except guard.TrainingBlocked:
            raise
        except Exception as e:
            if not _is_oom(e) or attempt == ctx.loop_cfg.oom_retries:
                raise
            ctx.state.setdefault("retries", []).append(f"{name}: out of memory at batch {cfg.batch_size}, retrying smaller")
            cfg = dataclasses.replace(cfg, batch_size=max(1, cfg.batch_size // 2), grad_accum=cfg.grad_accum * 2)
    models.register(ctx.conn, name, cfg.base_model, version, str(run_dir / "train_config.yaml"), adapter_path=str(run_dir / "adapter"))
    ctx.state |= {"challenger": name, "run_dir": run_dir, "train": {"batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum, "attempts": attempt + 1}}
    return {"model_version": name, "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum, "attempts": attempt + 1}


def read_train_log(ctx: LoopContext, tail: int = 10) -> dict:
    run_json = json.loads((ctx.state["run_dir"] / "run.json").read_text(encoding="utf-8"))
    return {"status": run_json["status"], "steps": run_json["steps"], "peak_vram_gb": run_json["peak_vram_gb"], "history": run_json["history"][-tail:]}


def convert_and_check_parity(ctx: LoopContext) -> dict:
    run_dir = ctx.state["run_dir"]
    hf_dir, ct2_dir = ctx.converter(ctx.train_cfg, run_dir)
    ctx.conn.execute("UPDATE model_versions SET ct2_path = ? WHERE version = ?", (str(ct2_dir), ctx.state["challenger"]))
    ctx.conn.commit()
    clips = [e for e in ds.load_manifest(Path(ctx.state["dataset"]["manifest"])) if e["split"] == "val"] or \
        ds.load_manifest(Path(ctx.state["dataset"]["manifest"]))
    paths = [ctx.paths.audio_dir / e["audio"] for e in clips[: ctx.loop_cfg.parity_clips]]
    result = ctx.parity(hf_dir, ct2_dir, paths, "en", ctx.train_cfg.parity_max_wer)
    ctx.state["parity_ok"] = result.passed
    return {"ct2_ready": True, "parity_passed": result.passed, "parity_wer": result.wer, "parity_clips": result.n}


def evaluate(ctx: LoopContext, which: str, run_id: int) -> dict:
    if which not in ("champion", "challenger"):
        raise ValueError("which must be 'champion' or 'challenger'")
    if which == "challenger":
        model = models.get(ctx.conn, ctx.state["challenger"])["ct2_path"]
    else:
        champ = models.champion(ctx.conn)
        model = champ["ct2_path"] if champ else ctx.loop_cfg.baseline_model
    terms = lexicon.approved_terms(ctx.conn)
    report = ctx.evaluator(model, ctx.eval_cfg, ctx.sc, ctx.variants, terms, ctx.loop_cfg.latency_clips)
    second = ctx.eval_cfg.get("second_set")
    if second:  # reported next to the primary set to expose overfitting to it; never used by the gate
        report["second_set"] = ctx.evaluator(model, second, ctx.sc, ctx.variants, terms, 0)
    ctx.loop_cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.loop_cfg.reports_dir / f"run-{run_id}-{which}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ctx.state[f"report_{which}"] = report
    version = ctx.state["challenger"] if which == "challenger" else (models.champion(ctx.conn) or {}).get("version")
    if version:
        models.set_eval_report(ctx.conn, version, str(path))
    return {"report_path": str(path), "overall": report["overall"], "latency_p50_s": report["latency_p50_s"],
            "per_language_wer": {k: v["normalized_wer"] for k, v in report["per_language"].items()}}


def decide_promotion(ctx: LoopContext) -> dict:
    """Apply the owner's thresholds. A pass queues an owner approval; a fail rejects the challenger.
    Idempotent: calling it again returns the first result instead of queuing or rejecting twice."""
    if "decision_result" in ctx.state:
        return ctx.state["decision_result"]
    decision = decide(ctx.state["report_champion"], ctx.state["report_challenger"], ctx.rules_loader(), ctx.state.get("parity_ok", False))
    ctx.state["decision"] = decision
    version = ctx.state["challenger"]
    if decision.passed:
        result = {"passed": True, "reasons": decision.reasons, "deltas": decision.deltas,
                  "approval_request_id": approvals.request(ctx.conn, version, decision)}
    else:
        models.reject(ctx.conn, version)
        result = {"passed": False, "reasons": decision.reasons, "deltas": decision.deltas}
    ctx.state["decision_result"] = result
    return result


def recent_corrections(ctx: LoopContext, limit: int = 50) -> list[dict]:
    """Text only (what was heard vs. what the owner wrote). No audio, no audio paths."""
    rows = ctx.conn.execute(
        "SELECT id, hypothesis_normalized AS heard, final_text AS corrected FROM utterances "
        "WHERE status = 'corrected' AND final_text != hypothesis_normalized ORDER BY id DESC LIMIT ?", (min(limit, 200),)
    )
    return [dict(r) for r in rows]


def propose_lexicon(ctx: LoopContext, canonical: str, variants: list[str], kind: str) -> dict:
    """Always stored unapproved: only the owner can approve (in the UI)."""
    return {"entry_id": lexicon.add_entry(ctx.conn, canonical, variants, kind, source="proposed"), "approved": False}


def loop_state(ctx: LoopContext, run) -> dict:
    return {"run_id": run.id, "steps_taken": len(run.steps), "step_budget": run.cfg.step_budget,
            "completed": [s["name"] for s in run.steps if s["ok"]], "failed": [s["name"] for s in run.steps if not s["ok"]],
            "dataset_version": ctx.state.get("dataset_version"), "challenger": ctx.state.get("challenger"),
            "parity_ok": ctx.state.get("parity_ok"), "retries": ctx.state.get("retries", [])}
