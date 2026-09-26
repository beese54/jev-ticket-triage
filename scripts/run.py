"""Run one system over one split, reusing cached calls.

Usage:
    uv run python scripts/run.py --system together-small --dataset tickets --split dev --limit 10
    uv run python scripts/run.py --system jev --dataset banking77 --split test

Systems: jev, together-small, together-large. Prints accuracy on scored fields as
a quick sanity check; full metrics come from the P6 report.
"""

import argparse
import asyncio
import statistics

from triage import config, data
from triage.backends import SYSTEMS, get_backend
from triage.runner import run
from triage.tasks import TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--dataset", required=True, choices=list(TASKS))
    ap.add_argument("--split", required=True, choices=["dev", "test", "holdout"])
    ap.add_argument("--limit", type=int, help="first N rows only (by id)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--repeat", type=int, default=0, help="repeat index >0 re-runs cached requests"
    )
    args = ap.parse_args()

    task = TASKS[args.dataset]
    rows = data.read_split(args.dataset, args.split).to_dict(orient="records")
    if args.limit:
        rows = rows[: args.limit]

    backend = get_backend(args.system)
    budget = (
        config.together_budget_usd() if args.system.startswith("together") else None
    )
    records, stats = asyncio.run(
        run(backend, task, rows, args.concurrency, budget, args.repeat)
    )

    print(
        f"{backend.system} ({backend.model}) {args.dataset}/{args.split}: "
        f"{len(rows)} rows | cached {stats.cached} | new calls {stats.called} | "
        f"failed {stats.failed} | new spend ${stats.spent_usd:.4f}"
    )
    for err in stats.errors[:5]:
        print("  error:", err)

    by_id = {r["id"]: r for r in rows}
    for f in (f for f in task.fields if f.scored):
        hits = [
            backend.parse(task, rec["response"])[f.name]["label"]
            == task.gold(by_id[rec["id"]])[f.name]
            for rec in records
        ]
        if hits:
            print(
                f"  {f.name:<9} accuracy vs original labels: {sum(hits) / len(hits):.1%} (n={len(hits)})"
            )
    if records:
        lat = [r["latency_ms"] for r in records]
        print(
            f"  latency p50 {statistics.median(lat):.0f} ms (includes cached records' original timings)"
        )


if __name__ == "__main__":
    main()
