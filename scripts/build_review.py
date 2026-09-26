"""Build the blind P2 review page: docs/review.html.

All 150 gold tickets are shown. Fields where the original label and Claude's blind label
agree need no action. Each disputed field shows the two labels as "Option A/B" in a seeded
random order, so the reviewer can't tell which source is which. Tickets are shuffled
(seeded) and only tickets up to the furthest one reached are exported, so a partly
finished review is still a random sample: fully agreed tickets are not over-represented.

Usage: uv run python scripts/build_review.py
"""

import json
import random
from pathlib import Path

from triage import data, labels

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = ROOT / "data" / "gold" / "claude_labels.jsonl"
FIELDS = ("queue", "priority", "type")
SEED = 20260926
VERSION = "gold-review-v1"


def main() -> None:
    claude = {
        r["id"]: r
        for r in map(json.loads, CLAUDE.read_text(encoding="utf-8").splitlines())
    }
    test = data.read_split("tickets", "test")
    gold = test[test["in_gold"]].sort_values("id").to_dict(orient="records")
    rng = random.Random(SEED)

    tickets = []
    for row in gold:
        c = claude[row["id"]]
        options, disputed = {}, []
        for f in FIELDS:
            if row[f] == c[f]:
                options[f] = [row[f]]
            else:
                pair = [row[f], c[f]]
                rng.shuffle(pair)
                options[f], disputed = pair, [*disputed, f]
        tickets.append(
            {
                "id": row["id"],
                "subject": row["subject"],
                "body": row["body"],
                "options": options,
                "disputed": disputed,
            }
        )
    rng.shuffle(tickets)

    payload = {
        "version": VERSION,
        "labels": {
            "queue": labels.QUEUES,
            "priority": labels.PRIORITIES,
            "type": labels.TYPES,
        },
        "tickets": tickets,
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    template = (ROOT / "dashboard" / "review_template.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" in template
    out = ROOT / "docs" / "review.html"
    out.write_text(
        template.replace("/*__DATA__*/null", blob), encoding="utf-8", newline="\n"
    )
    n_fields = sum(len(t["disputed"]) for t in tickets)
    print(
        f"wrote {out.relative_to(ROOT)}: {len(tickets)} tickets, {n_fields} disputed fields"
    )


if __name__ == "__main__":
    main()
