"""P0 smoke test: verify both APIs expose what the eval depends on.

Usage:
    uv run python scripts/p0_smoke.py --list-models   # Together chat models + prices
    uv run python scripts/p0_smoke.py                 # one call to Jev + each Together model

Writes results/p0_smoke.json as evidence for the P0 checklist.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

ROOT = Path(__file__).resolve().parent.parent
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

TICKET = (
    "Subject: Charged twice and now locked out\n\n"
    "Hi, my card was charged twice for this month's subscription and since this "
    "morning I can't log in to the portal at all. I need the duplicate refunded "
    "and access restored today, our team can't work."
)
QUEUES = {
    "Billing and Payments": "Billing issues and payment processing",
    "Technical Support": "Technical issues and support requests",
    "Customer Service": "Customer inquiries and service requests",
}
PRIORITIES = ["low", "medium", "high"]


def check_jev() -> dict:
    client = TypeSafeClient()
    t0 = time.perf_counter()
    resp = client.system_one(
        state=TICKET,
        questions={
            "queue": Choice(
                instructions="Which team should handle this ticket?", criteria=QUEUES
            ),
            "priority": Score(
                instructions="How urgent is this ticket?",
                criteria=[
                    "low: no business impact, can wait",
                    "medium: some impact, workaround exists",
                    "high: blocking work or money at stake, needs same-day action",
                ],
            ),
            "refund": Noul(instructions="The customer asks for a refund."),
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    out = {
        "model": resp.model,
        "latency_ms": round(latency_ms, 1),
        "usage": resp.usage.model_dump(),
        "answers": {k: v.model_dump() for k, v in resp.answers.items()},
    }
    print(json.dumps(out, indent=2))
    return out


def together_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}


def normalize_logprobs(lp: dict | None) -> list[dict]:
    """Together returns two logprob shapes depending on the model:
    native {tokens, token_logprobs, top_logprobs: [{tok: lp}]} (e.g. Llama) or
    OpenAI-style {content: [{token, logprob, top_logprobs: [{token, logprob}]}]} (e.g. Qwen).
    Normalize to [{token, p, top: {token: p}}]."""
    if not lp:
        return []
    if "content" in lp:
        return [
            {
                "token": t["token"],
                "p": math.exp(t["logprob"]),
                "top": {
                    a["token"]: math.exp(a["logprob"])
                    for a in t.get("top_logprobs") or []
                },
            }
            for t in lp["content"] or []
        ]
    tops = lp.get("top_logprobs") or [{}] * len(lp.get("tokens", []))
    return [
        {
            "token": tok,
            "p": math.exp(l),
            "top": {k: math.exp(v) for k, v in (top or {}).items() if v is not None},
        }
        for tok, l, top in zip(lp["tokens"], lp["token_logprobs"], tops)
        # special tokens (e.g. <|eot_id|>) can carry a null logprob
        if l is not None
    ]


def list_together_models() -> list[dict]:
    # Together's /models returns a bare JSON list, which the OpenAI client can't page.
    resp = httpx.get(
        f"{TOGETHER_BASE_URL}/models",
        headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for m in resp.json():
        if m.get("type") != "chat":
            continue
        pricing = m.get("pricing") or {}
        rows.append(
            {
                "id": m["id"],
                "context": m.get("context_length"),
                "input_per_M": pricing.get("input"),
                "output_per_M": pricing.get("output"),
            }
        )
    rows.sort(key=lambda r: (r["input_per_M"] or 0, r["id"]))
    for r in rows:
        print(
            f"{r['id']:<70} ctx={r['context']!s:<8} in=${r['input_per_M']}/M out=${r['output_per_M']}/M"
        )
    return rows


def check_together(model: str) -> dict:
    prompt = (
        "Classify the support ticket. Reply with JSON only: "
        '{"queue": <one of ' + json.dumps(list(QUEUES)) + ">, "
        '"priority": <one of ' + json.dumps(PRIORITIES) + ">}\n\n" + TICKET
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 100,
        "response_format": {"type": "json_object"},
        "logprobs": 5,
        # Hybrid-thinking models (Qwen3.5) would otherwise spend max_tokens on reasoning.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{TOGETHER_BASE_URL}/chat/completions",
        headers=together_headers(),
        json=body,
        timeout=90,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    tokens = normalize_logprobs(choice.get("logprobs"))
    out = {
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "usage": data.get("usage"),
        "content": content,
        "parsed_ok": parsed is not None,
        "logprobs_available": bool(tokens),
        "top_logprobs_available": any(len(t["top"]) > 1 for t in tokens),
        "tokens": [
            {
                "token": t["token"],
                "p": round(t["p"], 4),
                "top": {k: round(v, 4) for k, v in t["top"].items()},
            }
            for t in tokens
        ],
    }
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        list_together_models()
        return

    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if os.environ.get("TYPESAFE_API_KEY"):
        print("== Jev ==")
        report["jev"] = check_jev()
    else:
        print("skip Jev: TYPESAFE_API_KEY not set")
    for key in ("TOGETHER_SMALL_MODEL", "TOGETHER_LARGE_MODEL"):
        model = os.environ.get(key)
        if not model:
            print(f"skip {key}: not set")
            continue
        print(f"== Together {key} ==")
        report[key] = check_together(model)

    out_path = ROOT / "results" / "p0_smoke.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
