"""Metrics over predictions. Pure functions of lists, so they're easy to test.

Conventions: a prediction of None (invalid or unparseable output) counts as wrong.
`p` is the probability the system gave its chosen label; items with p=None are
ranked last for coverage and put in the lowest calibration bin.
"""

import math
import statistics


def accuracy(gold: list, pred: list) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold: list, pred: list) -> float:
    """Unweighted mean F1 over the classes present in gold."""
    scores = []
    for c in sorted(set(gold)):
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores)


def within_one(gold: list, pred: list, order: list[str]) -> float:
    """Ordinal accuracy allowing one level of error (e.g. medium for high)."""
    idx = {label: i for i, label in enumerate(order)}
    return sum(
        p is not None and abs(idx[g] - idx[p]) <= 1 for g, p in zip(gold, pred)
    ) / len(gold)


def invalid_rate(pred: list) -> float:
    return sum(p is None for p in pred) / len(pred)


def _ranked(correct: list[bool], score: list[float | None]) -> list[bool]:
    """Correctness sorted by score, most confident first; None scores last.
    Ties are broken by original order, so results are deterministic."""
    order = sorted(
        range(len(correct)),
        key=lambda i: (score[i] is None, -(score[i] or 0.0), i),
    )
    return [correct[i] for i in order]


def ranked_pairs(correct: list[bool], score: list[float | None]) -> list[list]:
    """[[p, 1|0], ...] most confident first: enough for a client to recompute any
    coverage/accuracy trade-off (e.g. a target-accuracy slider)."""
    order = sorted(
        range(len(correct)),
        key=lambda i: (score[i] is None, -(score[i] or 0.0), i),
    )
    return [
        [None if score[i] is None else round(score[i], 4), int(correct[i])]
        for i in order
    ]


def coverage_curve(
    correct: list[bool], score: list[float | None], points: int = 20
) -> list[dict]:
    """Accuracy on the most-confident fraction of items, for coverage 1/points..1."""
    ranked = _ranked(correct, score)
    n = len(ranked)
    out = []
    for k in range(1, points + 1):
        m = max(1, round(n * k / points))
        out.append({"coverage": m / n, "accuracy": sum(ranked[:m]) / m})
    return out


def coverage_at_accuracy(
    correct: list[bool], score: list[float | None], target: float
) -> float:
    """Largest share of items that can be automated (most confident first) while
    accuracy on the automated share stays >= target. 0.0 if never reached."""
    ranked = _ranked(correct, score)
    best, hits = 0.0, 0
    for m, ok in enumerate(ranked, start=1):
        hits += ok
        if hits / m >= target:
            best = m / len(ranked)
    return best


def accuracy_at_coverage(
    correct: list[bool], score: list[float | None], coverage: float
) -> float:
    ranked = _ranked(correct, score)
    m = max(1, round(len(ranked) * coverage))
    return sum(ranked[:m]) / m


def calibration(correct: list[bool], score: list[float | None], bins: int = 10) -> dict:
    """Expected calibration error plus the reliability-diagram bins."""
    table = [
        {"lo": b / bins, "hi": (b + 1) / bins, "n": 0, "conf": 0.0, "acc": 0.0}
        for b in range(bins)
    ]
    for ok, s in zip(correct, score):
        s = 0.0 if s is None else min(max(s, 0.0), 1.0)
        b = min(int(s * bins), bins - 1)
        table[b]["n"] += 1
        table[b]["conf"] += s
        table[b]["acc"] += ok
    n = len(correct)
    ece = 0.0
    for row in table:
        if row["n"]:
            row["conf"] /= row["n"]
            row["acc"] /= row["n"]
            ece += row["n"] / n * abs(row["acc"] - row["conf"])
    return {"ece": ece, "bins": [r for r in table if r["n"]]}


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def ops_summary(records: list[dict]) -> dict:
    lat = [r["latency_ms"] for r in records]
    costs = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    return {
        "calls": len(records),
        "latency_ms_p50": percentile(lat, 50),
        "latency_ms_p95": percentile(lat, 95),
        "input_tokens_mean": statistics.mean(r["input_tokens"] for r in records),
        "output_tokens_mean": statistics.mean(r["output_tokens"] for r in records),
        "cost_per_1k_usd": statistics.mean(costs) * 1000 if costs else None,
        "retried_calls": sum(r.get("attempts", 1) > 1 for r in records),
    }
