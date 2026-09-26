"""Regenerate results/metrics_<split>.json and results/explorer_<split>.json from the cache.

Usage:
    uv run python scripts/report.py --split test

No API keys are needed: everything is computed from cached responses.
"""

import argparse
import json

from triage import report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "holdout"])
    args = ap.parse_args()

    paths = report.write(args.split)
    metrics = json.loads(paths[0].read_text(encoding="utf-8"))
    print(f"gold labels: {metrics['gold_status']}")
    for run in metrics["runs"]:
        fields = ", ".join(
            f"{name} {m['accuracy']:.1%} (F1 {m['macro_f1']:.2f}, auto@90% {m['coverage_at_90']:.0%})"
            for name, m in run["fields"].items()
        )
        ops = run["ops"]
        cost = ops["cost_per_1k_usd"]
        cost_str = f"${cost:.3f}/1k" if cost is not None else "cost n/a"
        print(
            f"{run['system']:<15} {run['dataset']:<9} [{run['labels']}] n={run['n']} "
            f"repeats={run['repeats']} | {fields} | p50 {ops['latency_ms_p50']:.0f} ms | "
            f"{cost_str}"
        )
    for p in paths:
        print("wrote", p.relative_to(report.ROOT))


if __name__ == "__main__":
    main()
