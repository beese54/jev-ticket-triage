"""Load, clean and split the two evaluation datasets.

Raw files are downloaded to data/raw/ (gitignored) and verified by sha256.
Frozen splits are written to data/splits/ (committed) so every run, and every
reader of the repo, evaluates exactly the same tickets.
"""

import hashlib
import json
import re
from pathlib import Path

import httpx
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"

SEED = 20260926

BANKING77_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}
TICKETS_REPO = "Tobi-Bueck/customer-support-tickets"
TICKETS_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"

# Per-class counts drawn from the Banking77 test split (40 per class available).
B77_PER_CLASS = {"dev": 2, "test": 10, "holdout": 1}

# Tobi-Bueck split sizes; test uses proportional allocation with a per-queue floor
# so small queues still get enough tickets for per-class metrics and the gold subset.
TICKETS_DEV = 100
TICKETS_HOLDOUT = 50
TICKETS_TEST = 600
TICKETS_TEST_FLOOR = 20
GOLD_PER_QUEUE = 15

MIN_BODY_CHARS = 20
_GERMAN_WORDS = re.compile(
    r"\b(?:ich|und|nicht|bitte|wir|unsere|sehr geehrte)\b", re.IGNORECASE
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_banking77() -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in BANKING77_URLS.items():
        path = RAW_DIR / f"banking77_{split}.csv"
        if not path.exists():
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        paths[split] = path
    return paths


def download_tickets() -> Path:
    return Path(
        hf_hub_download(
            TICKETS_REPO, TICKETS_FILE, repo_type="dataset", local_dir=RAW_DIR
        )
    )


def clean_text(text: object) -> str:
    """Normalize the escaping artefacts found in the synthetic tickets."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\\n", "\n")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_banking77(test_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv).rename(columns={"category": "intent"})
    df["id"] = [f"b77-{i:04d}" for i in range(len(df))]
    df["text"] = df["text"].map(clean_text)
    return df[["id", "text", "intent"]]


def ticket_id(subject: str, body: str) -> str:
    return "tb-" + hashlib.sha1(f"{subject}\n{body}".encode()).hexdigest()[:10]


def load_tickets(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """English tickets, cleaned. Returns the frame plus counts of what was dropped."""
    raw = pd.read_csv(csv_path, encoding="utf-8")
    df = raw[raw["language"] == "en"].copy()
    dropped = {"non_english_label": int(len(raw) - len(df))}

    df["subject"] = df["subject"].map(clean_text)
    df["body"] = df["body"].map(clean_text)

    short = df["body"].str.len() < MIN_BODY_CHARS
    dropped["body_under_20_chars"] = int(short.sum())
    df = df[~short]

    german = df["body"].str.contains(_GERMAN_WORDS)
    dropped["german_text_labelled_en"] = int(german.sum())
    df = df[~german]

    dupes = df.duplicated(subset=["subject", "body"])
    dropped["duplicates"] = int(dupes.sum())
    df = df[~dupes]

    df["id"] = [ticket_id(s, b) for s, b in zip(df["subject"], df["body"])]
    assert df["id"].is_unique, "ticket id collision"
    cols = ["id", "subject", "body", "queue", "priority", "type", "version"]
    return df[cols].reset_index(drop=True), dropped


def split_banking77(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Balanced per-intent draw so every one of the 77 intents is equally represented."""
    total = sum(B77_PER_CLASS.values())
    picked = df.groupby("intent", group_keys=False).sample(n=total, random_state=SEED)
    splits: dict[str, list] = {k: [] for k in B77_PER_CLASS}
    for _, group in picked.groupby("intent"):
        start = 0
        for name, n in B77_PER_CLASS.items():
            splits[name].append(group.iloc[start : start + n])
            start += n
    return {
        k: pd.concat(v).sort_values("id").reset_index(drop=True)
        for k, v in splits.items()
    }


def _allocate(counts: pd.Series, total: int, floor: int) -> dict[str, int]:
    """Proportional allocation with a per-class floor (largest-remainder rounding)."""
    alloc = {q: floor for q in counts.index}
    remaining = total - floor * len(counts)
    share = counts / counts.sum() * remaining
    extra = share.astype(int)
    leftover = remaining - int(extra.sum())
    order = (share - extra).sort_values(ascending=False).index[:leftover]
    for q in counts.index:
        alloc[q] += int(extra[q]) + (1 if q in order else 0)
    return alloc


def _stratified(pool: pd.DataFrame, n: int, floor: int = 0) -> pd.DataFrame:
    alloc = _allocate(pool["queue"].value_counts(), n, floor)
    return pd.concat(
        pool[pool["queue"] == q].sample(n=k, random_state=SEED)
        for q, k in alloc.items()
    )


def split_tickets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rest = df
    holdout = _stratified(rest, TICKETS_HOLDOUT)
    rest = rest.drop(holdout.index)
    dev = _stratified(rest, TICKETS_DEV)
    rest = rest.drop(dev.index)
    test = _stratified(rest, TICKETS_TEST, TICKETS_TEST_FLOOR)
    gold_ids = set(
        test.groupby("queue", group_keys=False).sample(
            n=GOLD_PER_QUEUE, random_state=SEED
        )["id"]
    )
    test = test.assign(in_gold=test["id"].isin(gold_ids))

    return {
        name: part.sort_values("id").reset_index(drop=True)
        for name, part in {"dev": dev, "test": test, "holdout": holdout}.items()
    }


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_split(dataset: str, split: str) -> pd.DataFrame:
    """Load a frozen split, e.g. read_split("tickets", "test")."""
    return pd.read_json(SPLITS_DIR / f"{dataset}_{split}.jsonl", lines=True)
