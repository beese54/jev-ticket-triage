"""Per-queue accuracy on the adjudicated gold tickets (test, repeat 0), per system.

Usage: uv run python analysis/per_queue.py   -> results/per_queue_gold.json
Shows where the queue differences between systems come from.
"""

import json
from collections import Counter
from pathlib import Path

from triage import data
from triage.backends import SYSTEMS
from triage.report import load_gold, offline_backend
from triage.runner import cached_record
from triage.tasks import TICKETS

OUT = Path(__file__).resolve().parents[1] / "results" / "per_queue_gold.json"
gold = load_gold()
rows = [
    r for r in data.read_split("tickets", "test").to_dict("records") if r["id"] in gold
]
report = {"n": len(rows), "per_queue": {}, "predicted_distribution": {}}
preds = {}
for s in SYSTEMS:
    b = offline_backend(s)
    recs = {r["id"]: cached_record(b, TICKETS, r) for r in rows}
    if all(recs.values()):
        preds[s] = {
            i: b.parse(TICKETS, rec["response"])["queue"]["label"]
            for i, rec in recs.items()
        }
for q in sorted({g["queue"] for g in gold.values()}):
    ids = [r["id"] for r in rows if gold[r["id"]]["queue"] == q]
    report["per_queue"][q] = {
        "n": len(ids),
        **{s: sum(p[i] == q for i in ids) for s, p in preds.items()},
    }
for s, p in preds.items():
    report["predicted_distribution"][s] = dict(Counter(p.values()).most_common())
OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"{'gold queue':<32} {'n':>3} " + " ".join(f"{s:>15}" for s in preds))
for q, v in report["per_queue"].items():
    print(f"{q:<32} {v['n']:>3} " + " ".join(f"{v[s]:>15}" for s in preds))
