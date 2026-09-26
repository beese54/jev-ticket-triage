import asyncio
import json
import math

import httpx
import httpx2
import pytest

from triage import cache, config, logprobs
from triage.backends import JevBackend, TogetherBackend
from triage.runner import BudgetExceeded, run
from triage.tasks import BANKING77, TICKETS

TICKET = {
    "id": "tb-1",
    "subject": "Charged twice",
    "body": "Refund the duplicate charge.",
    "queue": "Billing and Payments",
    "priority": "high",
    "type": "Request",
}


# --- logprobs ----------------------------------------------------------------------


def test_normalize_both_shapes_and_null_logprobs():
    native = {
        "tokens": ["{", "x", "<|eot_id|>"],
        "token_logprobs": [0.0, math.log(0.5), None],
        "top_logprobs": [{"{": 0.0}, {"x": math.log(0.5), "y": None}, None],
    }
    openai = {
        "content": [
            {"token": "{", "logprob": 0.0, "top_logprobs": []},
            {
                "token": "x",
                "logprob": math.log(0.5),
                "top_logprobs": [{"token": "x", "logprob": math.log(0.5)}],
            },
        ]
    }
    for shape in (native, openai):
        toks = logprobs.normalize(shape)
        assert [t["token"] for t in toks] == ["{", "x"]
        assert toks[1]["p"] == pytest.approx(0.5)


def test_field_probability_multiplies_value_tokens_only():
    content = '{"queue": "Billing and Payments", "refund": true}'
    pieces = [
        ('{"', 1),
        ("queue", 1),
        ('":', 1),
        (' "', 1),
        ("Billing", 0.8),
        (" and", 0.9),
        (" Payments", 1.0),
        ('",', 1),
        (' "', 1),
        ("refund", 1),
        ('":', 1),
        (" true", 0.6),
        ("}", 1),
        ("<|eot_id|>", 1),
    ]
    tokens = [{"token": t, "p": p, "top": {}} for t, p in pieces]
    assert "".join(t for t, _ in pieces[:-1]) == content
    probs = logprobs.field_probabilities(
        content, tokens, ["queue", "refund", "missing"]
    )
    assert probs["queue"] == pytest.approx(0.8 * 0.9)
    assert probs["refund"] == pytest.approx(0.6)
    assert probs["missing"] is None


def test_field_probability_none_when_tokens_do_not_match_content():
    tokens = [{"token": "nope", "p": 1.0, "top": {}}]
    assert logprobs.field_probabilities('{"a": "b"}', tokens, ["a"]) == {"a": None}


# --- Jev backend (SDK over a mock transport) ---------------------------------------

JEV_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "queue": {
            "type": "choice",
            "choice": "Billing and Payments",
            "confidence": 0.8,
            "probabilities": {"Billing and Payments": 0.9, "Customer Service": 0.1},
        },
        "priority": {
            "type": "score",
            "score": 1.6,
            "confidence": 0.4,
            "legend": {"0": "l", "1": "m", "2": "h"},
            "probabilities": {"0": 0.0, "1": 0.4, "2": 0.6},
        },
        "type": {
            "type": "choice",
            "choice": "Request",
            "confidence": 0.9,
            "probabilities": {"Request": 0.95, "Incident": 0.05},
        },
        "asks_for_refund": {"type": "noul", "noul": 0.97},
    },
    "usage": {"input_tokens": 400, "output_tokens": 30},
}


def test_jev_payload_call_and_parse():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=JEV_RESPONSE)

    route = config.JevRoute("k", None, "jev-latest", "typesafe")
    backend = JevBackend(route, transport=httpx2.MockTransport(handler))
    payload = backend.build_payload(TICKETS, TICKET)
    assert payload["questions"]["priority"]["type"] == "score"
    assert len(payload["questions"]["priority"]["criteria"]) == 3
    assert payload["questions"]["customer_frustrated"]["type"] == "noul"

    result = asyncio.run(backend.call(payload))
    assert seen["body"]["questions"] == payload["questions"]
    assert result["input_tokens"] == 400

    pred = backend.parse(TICKETS, result["response"])
    assert pred["queue"]["label"] == "Billing and Payments"
    assert pred["queue"]["p"] == pytest.approx(0.9)
    # score -> argmax level mapped back to our label name, not the expected value
    assert pred["priority"]["label"] == "high"
    assert pred["priority"]["expected_level"] == 1.6
    assert pred["asks_for_refund"]["label"] is True
    # a question missing from the response parses to None instead of raising
    assert pred["service_outage"]["label"] is None


# --- Together backend (httpx mock) --------------------------------------------------


def together_response(content: str) -> dict:
    # One token per character keeps alignment trivial; each has p=1 except none.
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {
                    "tokens": list(content),
                    "token_logprobs": [0.0] * len(content),
                    "top_logprobs": None,
                },
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 40},
    }


def test_together_payload_has_same_labels_as_jev():
    backend = TogetherBackend("together-small", "Qwen/Qwen3.5-9B")
    payload = backend.build_payload(
        BANKING77, {"id": "b77-1", "text": "card?", "intent": "card_arrival"}
    )
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["intent"]["enum"] == list(BANKING77.fields[0].options)
    assert "card_arrival: " in payload["messages"][0]["content"]
    assert payload["messages"][1]["content"] == "card?"


def test_together_call_retries_then_parses(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))
    content = json.dumps(
        {
            "queue": "Billing and Payments",
            "priority": "high",
            "type": "NotAType",
            "security_or_data_breach": False,
            "service_outage": False,
            "customer_frustrated": True,
            "asks_for_refund": True,
        }
    )
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=together_response(content))

    backend = TogetherBackend(
        "together-small", "Qwen/Qwen3.5-9B", transport=httpx.MockTransport(handler)
    )
    result = asyncio.run(backend.call(backend.build_payload(TICKETS, TICKET)))
    assert result["attempts"] == 2
    pred = backend.parse(TICKETS, result["response"])
    assert pred["queue"] == {
        "label": "Billing and Payments",
        "p": pytest.approx(1.0),
        "probs": None,
        "confidence": None,
    }
    assert pred["type"]["label"] is None  # invalid label -> None (scored as wrong)
    assert pred["customer_frustrated"]["label"] is True


def test_together_parse_handles_unparseable_output():
    backend = TogetherBackend("together-small", "Qwen/Qwen3.5-9B")
    pred = backend.parse(
        TICKETS, {"choices": [{"message": {"content": "sorry, I can't"}}]}
    )
    assert all(v["label"] is None for v in pred.values())


# --- runner: cache + budget ---------------------------------------------------------


class FakeBackend:
    system = "together-fake"
    model = "Qwen/Qwen3.5-9B"

    def __init__(self):
        self.calls = 0

    def build_payload(self, task, row):
        return {"state": task.state(row)}

    async def call(self, payload):
        self.calls += 1
        return {
            "response": {"ok": True},
            "latency_ms": 5.0,
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "attempts": 1,
        }


def test_runner_caches_and_enforces_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    rows = [dict(TICKET, id=f"tb-{i}") for i in range(3)]
    backend = FakeBackend()

    records, stats = asyncio.run(
        run(backend, TICKETS, rows[:2], concurrency=1, budget_usd=10)
    )
    assert stats.called == 2 and backend.calls == 2
    assert stats.spent_usd == pytest.approx(0.34)  # 2 x 1M input tokens x $0.17/M

    records, stats = asyncio.run(
        run(backend, TICKETS, rows[:2], concurrency=1, budget_usd=10)
    )
    assert stats.cached == 2 and backend.calls == 2  # served from cache, no new calls
    assert [r["id"] for r in records] == ["tb-0", "tb-1"]

    with pytest.raises(BudgetExceeded):
        asyncio.run(run(backend, TICKETS, rows, concurrency=1, budget_usd=0.30))


def test_repeat_changes_key_but_repeat_zero_keeps_legacy_key():
    payload = {"messages": ["x"]}
    legacy = cache.request_key("m", payload)
    assert cache.request_key("m", payload, 0) == legacy
    assert (
        len(
            {
                legacy,
                cache.request_key("m", payload, 1),
                cache.request_key("m", payload, 2),
            }
        )
        == 3
    )
