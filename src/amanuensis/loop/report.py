"""Markdown run report (docs/systemdesign.md section 7, Run report)."""
import json
from pathlib import Path

from amanuensis.loop.run import LoopRun
from amanuensis.loop.tools import LoopContext
from amanuensis.store import lexicon


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def write_report(ctx: LoopContext, run: LoopRun, outcome: str) -> Path:
    s, lines = ctx.state, []
    add = lines.append
    add(f"# Loop run {run.id}")
    add(f"\n**Trigger:** {run.trigger}  \n**Outcome:** {outcome}\n")

    add("## Dataset")
    d = s.get("dataset")
    if d:
        add(f"Version `{d['version']}`. Hours by language group: "
            + (", ".join(f"{k}: {v:.3f} h" for k, v in d["hours_by_group"].items()) or "none") + ".")
    else:
        add("No dataset was built in this run.")

    add("\n## Training")
    if "train" in s:
        t = s["train"]
        add(f"Challenger `{s['challenger']}`: batch size {t['batch_size']} x accumulation {t['grad_accum']}, "
            f"{t['attempts']} attempt(s). Config is saved with the run.")
        run_json = s["run_dir"] / "run.json"
        if run_json.exists():
            hist = [h for h in json.loads(run_json.read_text(encoding="utf-8"))["history"] if "loss" in h]
            if hist:
                add(f"Loss {hist[0]['loss']:.3f} at step {hist[0]['step']} to {hist[-1]['loss']:.3f} at step {hist[-1]['step']}.")
    else:
        add("No model was trained in this run.")

    add("\n## Metrics vs champion (normalized WER, lower is better)")
    champ, chall = s.get("report_champion"), s.get("report_challenger")
    if champ and chall:
        add("\n| Slice | Champion | Challenger | Change |\n|---|---|---|---|")
        for lang in sorted(champ["per_language"]):
            a = champ["per_language"][lang]["normalized_wer"]
            b = chall["per_language"].get(lang, {}).get("normalized_wer")
            add(f"| {lang} | {_pct(a)} | {_pct(b)} | {'n/a' if b is None else f'{(b - a) * 100:+.1f} pts'} |")
        a, b = champ["overall"]["normalized_wer"], chall["overall"]["normalized_wer"]
        add(f"| **overall** | {_pct(a)} | {_pct(b)} | {(b - a) * 100:+.1f} pts |")
        add(f"\nLexicon term accuracy: champion {_pct(champ['overall']['term_accuracy'])}, challenger "
            f"{_pct(chall['overall']['term_accuracy'])}. Latency p50: champion {champ['latency_p50_s']:.2f}s, "
            f"challenger {chall['latency_p50_s']:.2f}s.")
    else:
        add("Not evaluated in this run.")

    add("\n## Promotion decision")
    dec = s.get("decision")
    if dec:
        add(("**Passed the gate.** Awaiting your approval in the correction UI (Approvals). " if dec.passed
             else "**Rejected by the gate.** ") + "Reasons: " + "; ".join(dec.reasons))
    else:
        add("No decision was made.")

    add("\n## Failures and retries")
    failed = [st for st in run.steps if not st["ok"]]
    if failed or s.get("retries"):
        for r in s.get("retries", []):
            add(f"- retry: {r}")
        for st in failed:
            add(f"- step `{st['name']}` failed: {st['error']}")
    else:
        add("None.")

    add("\n## Lexicon proposals awaiting approval")
    pending = [e for e in lexicon.list_entries(ctx.conn, approved=False)]
    add("\n".join(f"- `{e['canonical']}` ({e['kind']}), variants: {', '.join(e['variants']) or 'none'}" for e in pending) or "None.")

    add("\n## Steps")
    add("\n".join(f"- {st['name']}: {'ok' if st['ok'] else 'FAILED'} ({st['seconds']}s)" for st in run.steps) or "None.")

    ctx.loop_cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.loop_cfg.reports_dir / f"run-{run.id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
