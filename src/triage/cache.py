"""Content-addressed cache of raw API calls.

One file per (system, dataset, ticket, request): cache/<system>/<dataset>/<id>.<key>.json.gz
(compact JSON, gzipped: logprobs make raw records ~30 KB each).
The key hashes the exact request payload + model, so changing a prompt or label
description creates a new entry and older runs stay reproducible.
"""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"


def request_key(model: str, payload: dict, repeat: int = 0) -> str:
    """Repeat > 0 marks a deliberate re-run of the same request (determinism checks);
    it is left out of the hash for repeat 0 so the primary run keeps its key."""
    body = {"model": model, "payload": payload}
    if repeat:
        body["repeat"] = repeat
    canonical = json.dumps(body, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def path_for(system: str, dataset: str, item_id: str, key: str) -> Path:
    return CACHE_DIR / system / dataset / f"{item_id}.{key}.json.gz"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _read(path)


def _read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    # mtime=0 keeps the gzip bytes deterministic, so identical records give identical files.
    tmp.write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
    tmp.replace(path)


def iter_records(system: str | None = None):
    pattern = f"{system}/*/*.json.gz" if system else "*/*/*.json.gz"
    for path in sorted(CACHE_DIR.glob(pattern)):
        yield _read(path)


def total_spend(system_prefix: str) -> float:
    """USD spent so far by systems whose name starts with the prefix."""
    return sum(
        r.get("cost_usd") or 0.0
        for r in iter_records()
        if r.get("system", "").startswith(system_prefix)
    )
