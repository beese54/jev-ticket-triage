"""Do repeated identical requests give the same answers? (repeat 0 vs 1 vs 2)

Usage: uv run python analysis/determinism.py [system ...]
Reports, per field, the share of rows whose label is identical across all repeats
and the largest change in p. Writes results/determinism.json.
"""

import json
import sys
from pathlib import Path

from triage import data
from triage.backends import SYSTEMS, get_backend
from triage.runner import cached_record
from triage.tasks import TASKS

OUT = Path(__file__).resolve().parents[1] / "results" / "determinism.json"
systems = sys.argv[1:] or [s for s in SYSTEMS if s != "jev"]
report = {}
for system in systems:
    backend = get_backend(system)
    for dataset, task in TASKS.items():
        rows = data.read_split(dataset, "dev").to_dict(orient="records")
        runs = []
        for row in rows:
            recs = [cached_record(backend, task, row, rep) for rep in (0, 1, 2)]
            if all(recs):
                runs.append([backend.parse(task, r["response"]) for r in recs])
        if not runs:
            continue
        fields = {}
        for f in task.fields:
            labels = [[p[f.name]["label"] for p in preds] for preds in runs]
            ps = [[p[f.name]["p"] for p in preds] for preds in runs]
            stable = sum(len({str(x) for x in ls}) == 1 for ls in labels) / len(runs)
            max_dp = max(
                (max(v) - min(v) for v in ps if all(x is not None for x in v)),
                default=0.0,
            )
            fields[f.name] = {
                "label_identical": round(stable, 4),
                "max_p_change": round(max_dp, 4),
            }
        report[f"{system}/{dataset}"] = {"rows": len(runs), "fields": fields}
        worst = min(v["label_identical"] for v in fields.values())
        print(
            f"{system:<15} {dataset:<10} n={len(runs)}  lowest label agreement {worst:.0%}  "
            + ", ".join(
                f"{k} {v['label_identical']:.0%}, max dp {v['max_p_change']:.2f}"
                for k, v in fields.items()
            )
        )
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(report, indent=2) + "\n")
