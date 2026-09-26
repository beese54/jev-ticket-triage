"""Run the whole evaluation for one system, in protocol order. Cached calls are reused.

Usage:
    uv run python scripts/run_all.py --system jev
    uv run python scripts/run_all.py --system together-large --skip-holdout

Steps: dev (full, as a sanity check with the frozen descriptions) -> test
(Banking77 once; tickets x3 repeats, because LLM answers vary between runs) -> holdout.
Then rebuild results and the dashboard with scripts/report.py and build_dashboard.py.
"""

import argparse
import asyncio

from triage import config, data
from triage.backends import SYSTEMS, get_backend
from triage.runner import run
from triage.tasks import TASKS

TICKET_TEST_REPEATS = (0, 1, 2)


def step(backend, dataset: str, split: str, repeat: int, concurrency: int) -> None:
    task = TASKS[dataset]
    rows = data.read_split(dataset, split).to_dict(orient="records")
    budget = (
        config.together_budget_usd() if backend.system.startswith("together") else None
    )
    _, stats = asyncio.run(run(backend, task, rows, concurrency, budget, repeat))
    print(
        f"{backend.system:<15} {dataset:<9} {split:<7} repeat {repeat}: {len(rows)} rows | "
        f"cached {stats.cached} | new {stats.called} | failed {stats.failed} | "
        f"${stats.spent_usd:.4f}"
    )
    for err in stats.errors[:3]:
        print("   error:", err)
    if stats.failed:
        raise SystemExit(
            "failed calls: rerun this script to retry them (successes are cached)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip-holdout", action="store_true")
    args = ap.parse_args()
    backend = get_backend(args.system)

    for dataset in TASKS:
        step(backend, dataset, "dev", 0, args.concurrency)
    step(backend, "banking77", "test", 0, args.concurrency)
    for rep in TICKET_TEST_REPEATS:
        step(backend, "tickets", "test", rep, args.concurrency)
    if not args.skip_holdout:
        for dataset in TASKS:
            step(backend, dataset, "holdout", 0, args.concurrency)


if __name__ == "__main__":
    main()
