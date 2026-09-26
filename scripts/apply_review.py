"""Turn the P2 review export into gold labels: data/splits/tickets_gold_labels.jsonl.

Usage: save the text copied from docs/review.html as data/gold/review_decisions.json, then
    uv run python scripts/apply_review.py

A gold ticket is included when the reviewer reached it (it is in the export) and every field
is settled: either the original and Claude's blind label agree, or the reviewer decided it.
Tickets past the furthest one reached are left out even if fully agreed, so a partial review
stays a random sample. Also reports how often the reviewer sided
with each source, which measures how noisy the original labels are and how reliable
Claude's first pass was.
"""

import json
from pathlib import Path

from triage import data, labels

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data" / "gold"
FIELDS = ("queue", "priority", "type")
ALLOWED = {"queue": labels.QUEUES, "priority": labels.PRIORITIES, "type": labels.TYPES}


def main() -> None:
    claude = {
        r["id"]: r
        for r in map(
            json.loads,
            (GOLD_DIR / "claude_labels.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    review = json.loads(
        (GOLD_DIR / "review_decisions.json").read_text(encoding="utf-8")
    )
    decided = {d["id"]: d for d in review["decisions"]}
    test = data.read_split("tickets", "test")
    gold = test[test["in_gold"]].sort_values("id").to_dict(orient="records")

    out, sided = [], {f: {"original": 0, "claude": 0, "other": 0} for f in FIELDS}
    for row in gold:
        if row["id"] not in decided:
            continue
        c, d = claude[row["id"]], decided[row["id"]]
        final = {}
        for f in FIELDS:
            if row[f] == c[f]:
                final[f] = row[f]
            elif f in d:
                assert d[f] in ALLOWED[f], f"{row['id']}: invalid {f} {d[f]!r}"
                final[f] = d[f]
                side = (
                    "original"
                    if d[f] == row[f]
                    else "claude"
                    if d[f] == c[f]
                    else "other"
                )
                sided[f][side] += 1
        if len(final) == len(FIELDS):
            out.append({"id": row["id"], **final})

    path = data.SPLITS_DIR / "tickets_gold_labels.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in out:
            fh.write(json.dumps(rec) + "\n")
    print(
        f"wrote {path.relative_to(ROOT)}: {len(out)} of {len(gold)} gold tickets settled"
    )
    for f, s in sided.items():
        n = sum(s.values())
        if n:
            print(
                f"  {f:<9} disputed decisions {n}: reviewer chose original {s['original'] / n:.0%}, "
                f"Claude {s['claude'] / n:.0%}, other {s['other'] / n:.0%}"
            )
    (GOLD_DIR / "review_summary.json").write_text(
        json.dumps({"settled": len(out), "of": len(gold), "sided": sided}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
