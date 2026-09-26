"""Build the static dashboard: dashboard/template.html + results/*.json -> docs/index.html.

Usage:
    uv run python scripts/report.py --split test
    uv run python scripts/build_dashboard.py --split test

The output is one self-contained HTML file with the data inlined: no server,
no network requests. docs/ is where GitHub Pages serves it from.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/beese54/jev-ticket-triage"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "holdout"])
    args = ap.parse_args()

    results = ROOT / "results"
    data = {
        "metrics": json.loads(
            (results / f"metrics_{args.split}.json").read_text(encoding="utf-8")
        ),
        "explorer": json.loads(
            (results / f"explorer_{args.split}.json").read_text(encoding="utf-8")
        ),
        "determinism": json.loads(
            (results / "determinism.json").read_text(encoding="utf-8")
        )
        if (results / "determinism.json").exists()
        else {},
        "repo": REPO_URL,
    }
    # "</" inside a <script> would end it early; escape it in the embedded JSON.
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    template = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" in template, "template placeholder missing"
    html = template.replace("/*__DATA__*/null", blob)

    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
