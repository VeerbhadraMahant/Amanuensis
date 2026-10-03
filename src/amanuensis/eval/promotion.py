"""Promotion gate (golden rule 5, ADR-007): a pure function over two eval reports and config thresholds.

Agents may call decide() but never edit configs/promotion.yaml or the eval data.
Report shape (see eval.report.build_report):
  {"overall": {"normalized_wer", "term_accuracy"},
   "per_language": {lang: {"normalized_wer", "term_accuracy", ...}},
   "latency_p50_s": float}
"""
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


class PromotionConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromotionRules:
    min_relative_wer_improvement: float  # overall normalized WER must drop by at least this fraction
    max_language_regression_abs: float  # no language may worsen (absolute WER) by more than this
    max_term_accuracy_drop: float  # lexicon term accuracy may not fall by more than this
    max_latency_p50_worsening_s: float
    require_parity: bool
    language_tolerance_overrides: dict[str, float] = field(default_factory=dict)  # e.g. a stricter German tolerance


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: list[str]
    deltas: dict[str, float]  # per-language normalized WER change (negative = better), for the report


def load_rules(path: Path = Path("configs/promotion.yaml")) -> PromotionRules:
    """Refuses to load until the owner has filled in every threshold. No defaults on purpose."""
    if not path.exists():
        raise PromotionConfigError(f"{path} does not exist")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [f.name for f in fields(PromotionRules) if f.name != "language_tolerance_overrides" and raw.get(f.name) is None]
    if missing:
        raise PromotionConfigError(f"owner has not set: {', '.join(missing)} in {path}")
    return PromotionRules(**{f.name: raw[f.name] for f in fields(PromotionRules) if f.name in raw})


def decide(champion: dict, challenger: dict, rules: PromotionRules, parity_ok: bool) -> PromotionDecision:
    reasons: list[str] = []
    cw, nw = champion["overall"]["normalized_wer"], challenger["overall"]["normalized_wer"]
    needed = cw * (1 - rules.min_relative_wer_improvement)
    if not nw <= needed:
        reasons.append(f"overall normalized WER {nw:.4f} does not beat champion {cw:.4f} by "
                       f"{rules.min_relative_wer_improvement:.1%} (needs <= {needed:.4f})")

    deltas: dict[str, float] = {}
    for lang, c in champion["per_language"].items():
        if lang not in challenger["per_language"]:
            reasons.append(f"challenger report has no '{lang}' slice")
            continue
        d = challenger["per_language"][lang]["normalized_wer"] - c["normalized_wer"]
        deltas[lang] = d
        tol = rules.language_tolerance_overrides.get(lang, rules.max_language_regression_abs)
        if d > tol:
            reasons.append(f"'{lang}' regressed by {d:.4f} (tolerance {tol:.4f})")

    ct, nt = champion["overall"].get("term_accuracy"), challenger["overall"].get("term_accuracy")
    if ct is not None:
        if nt is None or ct - nt > rules.max_term_accuracy_drop:
            reasons.append(f"lexicon term accuracy fell from {ct:.3f} to {nt if nt is None else round(nt, 3)}")

    worse = challenger["latency_p50_s"] - champion["latency_p50_s"]
    if worse > rules.max_latency_p50_worsening_s:
        reasons.append(f"p50 latency worsened by {worse:.3f}s (tolerance {rules.max_latency_p50_worsening_s:.3f}s)")

    if rules.require_parity and not parity_ok:
        reasons.append("CT2 parity check did not pass")

    return PromotionDecision(not reasons, reasons or ["all promotion gates passed"], deltas)
