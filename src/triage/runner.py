"""Run a system over a split: cached calls are reused, new calls are saved."""

import asyncio
import time
from dataclasses import dataclass, field

from triage import cache, config
from triage.backends import Backend
from triage.tasks import Task


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunStats:
    cached: int = 0
    called: int = 0
    failed: int = 0
    spent_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


def make_record(
    backend: Backend, task: Task, row: dict, key: str, payload: dict, result: dict
) -> dict:
    cost = config.cost_usd(
        backend.model, result["input_tokens"], result["output_tokens"]
    )
    return {
        "system": backend.system,
        "model": backend.model,
        "dataset": task.dataset,
        "id": row["id"],
        "key": key,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latency_ms": round(result["latency_ms"], 1),
        "attempts": result["attempts"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": cost,
        "request": payload,
        "response": result["response"],
    }


async def run(
    backend: Backend,
    task: Task,
    rows: list[dict],
    concurrency: int = 4,
    budget_usd: float | None = None,
) -> tuple[list[dict], RunStats]:
    """Returns (records in row order, stats). Records for failed calls are omitted."""
    stats = RunStats()
    spent_before = cache.total_spend(backend.system) if budget_usd is not None else 0.0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(row: dict) -> dict | None:
        payload = backend.build_payload(task, row)
        key = cache.request_key(backend.model, payload)
        path = cache.path_for(backend.system, task.dataset, row["id"], key)
        if (hit := cache.load(path)) is not None:
            stats.cached += 1
            return hit
        async with sem:
            if budget_usd is not None and spent_before + stats.spent_usd >= budget_usd:
                raise BudgetExceeded(
                    f"{backend.system}: ${spent_before + stats.spent_usd:.4f} spent, "
                    f"budget ${budget_usd:.2f}"
                )
            try:
                result = await backend.call(payload)
            # Any failure is recorded (not cached) so a rerun retries it.
            except Exception as exc:  # noqa: BLE001
                stats.failed += 1
                stats.errors.append(f"{row['id']}: {type(exc).__name__}: {exc}"[:300])
                return None
        record = make_record(backend, task, row, key, payload, result)
        cache.save(path, record)
        async with lock:
            stats.called += 1
            stats.spent_usd += record["cost_usd"] or 0.0
        return record

    results = await asyncio.gather(*(one(r) for r in rows))
    return [r for r in results if r is not None], stats
