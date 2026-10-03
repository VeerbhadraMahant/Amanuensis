from dataclasses import dataclass, field

import numpy as np


def _pcts(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "p50": None, "p95": None}
    return {"n": len(xs), "p50": float(np.percentile(xs, 50)), "p95": float(np.percentile(xs, 95))}


@dataclass
class LatencyStats:
    first_partial_s: list[float] = field(default_factory=list)  # speech onset -> first text shown
    commit_s: list[float] = field(default_factory=list)  # word end -> word committed
    decode_s: list[float] = field(default_factory=list)  # one decode cycle

    def summary(self) -> dict:
        return {
            "first_partial_s": _pcts(self.first_partial_s),
            "commit_s": _pcts(self.commit_s),
            "decode_s": _pcts(self.decode_s),
        }
