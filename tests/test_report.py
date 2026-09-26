import json

import pytest

from triage import cache, data, report
from triage.backends import TogetherBackend
from triage.runner import make_record
from triage.tasks import TICKETS


def fake_response(labels: dict) -> dict:
    content = json.dumps(labels)
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {
                    "tokens": list(content),
                    "token_logprobs": [0.0] * len(content),
                },
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 40},
    }


def test_report_rebuilds_metrics_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    backend = TogetherBackend("together-small", "Qwen/Qwen3.5-9B")
    rows = data.read_split("tickets", "dev").to_dict(orient="records")
    signals = {
        n: False
        for n in (
            "security_or_data_breach",
            "service_outage",
            "customer_frustrated",
            "asks_for_refund",
        )
    }
    for i, row in enumerate(rows):
        gold = TICKETS.gold(row)
        # queue always right; type right on even rows only
        answer = dict(gold, **signals)
        if i % 2:
            answer["type"] = "Change" if gold["type"] != "Change" else "Request"
        payload = backend.build_payload(TICKETS, row)
        key = cache.request_key(backend.model, payload)
        result = {
            "response": fake_response(answer),
            "latency_ms": 100.0 + i,
            "input_tokens": 500,
            "output_tokens": 40,
            "attempts": 1,
        }
        cache.save(
            cache.path_for(backend.system, "tickets", row["id"], key),
            make_record(backend, TICKETS, row, key, payload, result),
        )

    out = report.build("dev")
    runs = [r for r in out["runs"] if r["dataset"] == "tickets"]
    assert [r["system"] for r in runs] == ["together-small"]
    run = runs[0]
    assert run["n"] == len(rows) and run["repeats"] == [0]
    assert run["fields"]["queue"]["accuracy"] == 1.0
    assert run["fields"]["type"]["accuracy"] == pytest.approx(0.5)
    assert run["fields"]["queue"]["coverage_at_95"] == 1.0
    assert len(run["fields"]["type"]["ranked"]) == len(rows)
    # cost: 500 in x $0.17/M + 40 out x $0.25/M, per 1,000 tickets
    assert run["ops"]["cost_per_1k_usd"] == pytest.approx(
        (500 * 0.17 + 40 * 0.25) / 1e6 * 1000
    )
    assert out["gold_status"] == "pending (P2)"
    assert "jev" in out["systems"] and not out["systems"]["jev"]["datasets"]
