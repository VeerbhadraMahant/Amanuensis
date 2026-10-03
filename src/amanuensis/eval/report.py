from collections import defaultdict
from dataclasses import asdict, dataclass

from amanuensis.eval.metrics import cer, normalized_wer, term_accuracy, wer


@dataclass(frozen=True)
class Utterance:
    language: str  # slice name, e.g. "hi-en", "mr-en", "en", "de"
    ref: str
    hyp: str


@dataclass(frozen=True)
class SliceReport:
    n: int
    wer: float
    normalized_wer: float
    cer: float
    term_accuracy: float | None


def per_language_report(
    utts: list[Utterance],
    variants: dict[str, str] | None = None,
    terms: list[str] | None = None,
) -> dict[str, dict]:
    groups: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        groups[u.language].append(u)
    out = {}
    for lang, g in sorted(groups.items()):
        refs, hyps = [u.ref for u in g], [u.hyp for u in g]
        out[lang] = asdict(
            SliceReport(
                n=len(g),
                wer=wer(refs, hyps),
                normalized_wer=normalized_wer(refs, hyps, variants),
                cer=cer(refs, hyps),
                term_accuracy=term_accuracy(refs, hyps, terms) if terms else None,
            )
        )
    return out


def build_report(
    utts: list[Utterance],
    variants: dict[str, str] | None,
    terms: list[str] | None,
    latency_p50_s: float,
) -> dict:
    """Full eval report: the input to the promotion gate (eval.promotion.decide)."""
    overall = per_language_report([Utterance("all", u.ref, u.hyp) for u in utts], variants, terms)["all"]
    return {
        "overall": {"normalized_wer": overall["normalized_wer"], "term_accuracy": overall["term_accuracy"]},
        "per_language": per_language_report(utts, variants, terms),
        "latency_p50_s": latency_p50_s,
    }
