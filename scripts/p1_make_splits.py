"""P1: download both datasets, clean them, and freeze dev/test/holdout splits.

Usage:
    uv run python scripts/p1_make_splits.py

Writes data/splits/*.jsonl and data/splits/manifest.json. Re-running produces
byte-identical output (fixed seed); the manifest's source hashes detect upstream changes.
"""

import json

from triage import data


def label_counts(df, col):
    return {k: int(v) for k, v in df[col].value_counts().sort_index().items()}


def main() -> None:
    b77_paths = data.download_banking77()
    tickets_path = data.download_tickets()

    b77 = data.load_banking77(b77_paths["test"])
    b77_splits = data.split_banking77(b77)

    tickets, dropped = data.load_tickets(tickets_path)
    ticket_splits = data.split_tickets(tickets)

    manifest = {
        "seed": data.SEED,
        "sources": {
            "banking77_test": {
                "url": data.BANKING77_URLS["test"],
                "license": "CC-BY-4.0",
                "sha256": data.sha256(b77_paths["test"]),
            },
            "tickets": {
                "repo": data.TICKETS_REPO,
                "file": data.TICKETS_FILE,
                "license": "CC-BY-NC-4.0",
                "sha256": data.sha256(tickets_path),
            },
        },
        "tickets_cleaning": {
            "english_rows_kept": len(tickets),
            "dropped": dropped,
        },
        "splits": {},
    }

    for name, df in b77_splits.items():
        data.write_jsonl(df, data.SPLITS_DIR / f"banking77_{name}.jsonl")
        manifest["splits"][f"banking77_{name}"] = {
            "rows": len(df),
            "intents": int(df["intent"].nunique()),
        }

    for name, df in ticket_splits.items():
        data.write_jsonl(df, data.SPLITS_DIR / f"tickets_{name}.jsonl")
        entry = {
            "rows": len(df),
            "queue": label_counts(df, "queue"),
            "priority": label_counts(df, "priority"),
            "type": label_counts(df, "type"),
        }
        if "in_gold" in df:
            entry["gold_rows"] = int(df["in_gold"].sum())
        manifest["splits"][f"tickets_{name}"] = entry

    out = data.SPLITS_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
