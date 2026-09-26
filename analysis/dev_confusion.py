"""Confusion matrix of a system's cached dev predictions vs the original labels.

Usage: uv run python analysis/dev_confusion.py <system> <field> [dataset]
Only cache entries for the CURRENT request (current labels.py) are counted.
"""

import sys

import pandas as pd

from triage import data
from triage.backends import get_backend
from triage.runner import cached_record
from triage.tasks import TASKS

system, field = sys.argv[1], sys.argv[2]
dataset = sys.argv[3] if len(sys.argv) > 3 else "tickets"
task, backend = TASKS[dataset], get_backend(system)
rows = data.read_split(dataset, "dev").to_dict(orient="records")
gold, pred = [], []
for row in rows:
    rec = cached_record(backend, task, row)
    if rec is None:
        continue
    gold.append(task.gold(row)[field])
    pred.append(backend.parse(task, rec["response"])[field]["label"])
g, p = pd.Series(gold, name="gold"), pd.Series(pred, name="pred").fillna("<invalid>")
print(f"{system} {dataset}/dev {field}: n={len(g)} accuracy={(g == p).mean():.0%}")
print(pd.crosstab(g, p).to_string())
