"""Cache -> results. Regenerates every number the dashboard shows from cached responses.

Only cache entries matching the CURRENT request (current labels.py and prompt) count,
so results always describe the code in the repo. Systems without cached records for a
split (e.g. Jev before an API key is available) are reported as unavailable.
"""

import json
import statistics
import time
from pathlib import Path

from triage import config, data, metrics
from triage.backends import SYSTEMS, Backend, JevBackend, TogetherBackend
from triage.runner import cached_record
from triage.tasks import TASKS, Task

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
GOLD_FILE = data.SPLITS_DIR / "tickets_gold_labels.jsonl"
REPEATS = (0, 1, 2)
TARGETS = (0.9, 0.95)
EXPLORER_ROWS = {"tickets": 60, "banking77": 40}


def offline_backend(system: str) -> Backend:
    """A backend for building payloads and parsing only (no API key needed)."""
    if system == "jev":
        return JevBackend(config.JevRoute("", None, "jev-latest", "offline"))
    return TogetherBackend(system, config.together_models()[system])


def load_gold() -> dict[str, dict] | None:
    """Hand-adjudicated ticket labels from P2, keyed by id (None until P2 is done)."""
    if not GOLD_FILE.exists():
        return None
    rows = [
        json.loads(line)
        for line in GOLD_FILE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return {r["id"]: r for r in rows}


def collect(
    backend: Backend, task: Task, rows: list[dict]
) -> dict[int, list[tuple[dict, dict, dict]]]:
    """repeat -> [(row, record, prediction)] for repeats where every row is cached."""
    out = {}
    for rep in REPEATS:
        recs = [cached_record(backend, task, row, rep) for row in rows]
        if all(recs):
            out[rep] = [
                (row, rec, backend.parse(task, rec["response"]))
                for row, rec in zip(rows, recs)
            ]
    return out


def field_metrics(task: Task, field, runs: dict, gold_of) -> dict:
    per_rep = []
    for items in runs.values():
        gold = [gold_of(row)[field.name] for row, _, _ in items]
        pred = [p[field.name]["label"] for _, _, p in items]
        per_rep.append((gold, pred, [p[field.name] for _, _, p in items]))

    accs = [metrics.accuracy(g, p) for g, p, _ in per_rep]
    gold, pred, preds = per_rep[0]  # curves and calibration from the primary run
    correct = [g == p for g, p in zip(gold, pred)]
    score = [x["p"] for x in preds]
    out = {
        "accuracy": statistics.mean(accs),
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "macro_f1": statistics.mean(metrics.macro_f1(g, p) for g, p, _ in per_rep),
        "invalid_rate": statistics.mean(metrics.invalid_rate(p) for _, p, _ in per_rep),
        "classes": len(field.options),
        "majority_baseline": max(gold.count(c) for c in set(gold)) / len(gold),
        "curve": metrics.coverage_curve(correct, score),
        "calibration": metrics.calibration(correct, score),
        "ranked": metrics.ranked_pairs(correct, score),
    }
    for t in TARGETS:
        out[f"coverage_at_{round(t * 100)}"] = metrics.coverage_at_accuracy(
            correct, score, t
        )
    if field.kind == "score":
        out["within_one"] = statistics.mean(
            metrics.within_one(g, p, list(field.options)) for g, p, _ in per_rep
        )
    # Jev also reports its own concentration-based confidence; evaluate it as a second signal.
    native = [x.get("confidence") for x in preds]
    if any(c is not None for c in native):
        out["native_confidence"] = {
            "curve": metrics.coverage_curve(correct, native),
            "ranked": metrics.ranked_pairs(correct, native),
            **{
                f"coverage_at_{round(t * 100)}": metrics.coverage_at_accuracy(
                    correct, native, t
                )
                for t in TARGETS
            },
        }
    return out


def build(split: str) -> dict:
    gold_labels = load_gold()
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": split,
        "prices": {m: vars(p) for m, p in config.PRICES.items()},
        "systems": {},
        "runs": [],
        "gold_status": "adjudicated" if gold_labels else "pending (P2)",
    }
    for system in SYSTEMS:
        backend = offline_backend(system)
        report["systems"][system] = {"model": backend.model, "datasets": []}
        for dataset, task in TASKS.items():
            rows = data.read_split(dataset, split).to_dict(orient="records")
            runs = collect(backend, task, rows)
            if not runs:
                continue
            report["systems"][system]["datasets"].append(dataset)
            label_sets = [("original", rows, task.gold)]
            if dataset == "tickets" and gold_labels:
                gold_rows = [r for r in rows if r["id"] in gold_labels]
                label_sets.append(
                    ("adjudicated", gold_rows, lambda r: gold_labels[r["id"]])
                )
            for label_set, subset, gold_of in label_sets:
                ids = {r["id"] for r in subset}
                sub_runs = {
                    rep: [x for x in items if x[0]["id"] in ids]
                    for rep, items in runs.items()
                }
                report["runs"].append(
                    {
                        "system": system,
                        "model": backend.model,
                        "dataset": dataset,
                        "labels": label_set,
                        "n": len(subset),
                        "repeats": sorted(sub_runs),
                        "fields": {
                            f.name: field_metrics(task, f, sub_runs, gold_of)
                            for f in task.fields
                            if f.scored
                        },
                        "ops": metrics.ops_summary(
                            [rec for _, rec, _ in sub_runs[min(sub_runs)]]
                        ),
                    }
                )
    return report


def explorer(split: str) -> dict:
    """A small side-by-side sample for the dashboard's ticket explorer."""
    out = {}
    for dataset, task in TASKS.items():
        rows = data.read_split(dataset, split).to_dict(orient="records")
        if dataset == "tickets" and "in_gold" in rows[0]:
            rows = [r for r in rows if r["in_gold"]] + [
                r for r in rows if not r["in_gold"]
            ]
        items = []
        for row in rows:
            entry = {
                "id": row["id"],
                "text": task.state(row),
                "gold": task.gold(row),
                "systems": {},
            }
            for system in SYSTEMS:
                backend = offline_backend(system)
                rec = cached_record(backend, task, row)
                if rec is None:
                    continue
                pred = backend.parse(task, rec["response"])
                entry["systems"][system] = {
                    "latency_ms": rec["latency_ms"],
                    "fields": {
                        name: {
                            k: v
                            for k, v in val.items()
                            if k in ("label", "p", "confidence")
                        }
                        for name, val in pred.items()
                    },
                }
            if entry["systems"]:
                items.append(entry)
            if len(items) >= EXPLORER_ROWS[dataset]:
                break
        out[dataset] = items
    return out


def write(split: str) -> list[Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    paths = []
    for name, payload in (("metrics", build(split)), ("explorer", explorer(split))):
        path = RESULTS_DIR / f"{name}_{split}.json"
        path.write_text(
            json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8"
        )
        paths.append(path)
    return paths
