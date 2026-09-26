"""The systems under test. Each backend builds a request payload from a Task,
calls its API, and parses the raw response into the shared prediction format:

    {field: {"label": str | bool | None, "p": float | None,
             "probs": {label: p} | None, "confidence": float | None}}

`p` is the probability the system gives its chosen label. That is the common
signal used for coverage and calibration curves. `confidence` is Jev's native
concentration-based confidence (None for LLMs).
"""

import asyncio
import os
import random
import time
from typing import Protocol

import httpx
from typesafe_sdk import AsyncTypeSafeClient

from triage import config, logprobs
from triage.tasks import Field, Task

Prediction = dict[str, dict]


class Backend(Protocol):
    system: str
    model: str

    def build_payload(self, task: Task, row: dict) -> dict: ...
    async def call(self, payload: dict) -> dict:
        """Returns {response, latency_ms, input_tokens, output_tokens, attempts}."""
        ...

    def parse(self, task: Task, response: dict) -> Prediction: ...


# --- Jev ---------------------------------------------------------------------------


def jev_question(field: Field) -> dict:
    if field.kind == "choice":
        return {
            "type": "choice",
            "instructions": field.instruction,
            "criteria": dict(field.options),
        }
    if field.kind == "score":
        return {
            "type": "score",
            "instructions": field.instruction,
            "criteria": list(field.options.values()),
        }
    return {"type": "noul", "instructions": field.instruction}


class JevBackend:
    system = "jev"

    def __init__(self, route: config.JevRoute, transport=None):
        self.route = route
        self.model = route.model
        self._transport = transport

    def build_payload(self, task: Task, row: dict) -> dict:
        return {
            "state": task.state(row),
            "questions": {f.name: jev_question(f) for f in task.fields},
        }

    async def call(self, payload: dict) -> dict:
        kwargs = {"api_key": self.route.api_key, "model": self.route.model}
        if self.route.base_url:
            kwargs["base_url"] = self.route.base_url
        if self._transport:
            kwargs["transport"] = self._transport
        async with AsyncTypeSafeClient(**kwargs) as client:
            t0 = time.perf_counter()
            resp = await client.system_one(payload["state"], payload["questions"])
            latency_ms = (time.perf_counter() - t0) * 1000
        raw = resp.model_dump(mode="json")
        usage = raw.get("usage") or {}
        return {
            "response": raw,
            "latency_ms": latency_ms,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "attempts": 1,  # SDK retries internally; latency includes them
        }

    def parse(self, task: Task, response: dict) -> Prediction:
        answers = response.get("answers") or {}
        out: Prediction = {}
        for f in task.fields:
            a = answers.get(f.name)
            if a is None:
                out[f.name] = {
                    "label": None,
                    "p": None,
                    "probs": None,
                    "confidence": None,
                }
            elif f.kind == "choice":
                probs = a["probabilities"]
                out[f.name] = {
                    "label": a["choice"],
                    "p": probs.get(a["choice"]),
                    "probs": probs,
                    "confidence": a.get("confidence"),
                }
            elif f.kind == "score":
                # Score returns an expected value (e.g. 1.43); the label is the most
                # likely level, mapped back from the level index to our label name.
                names = list(f.options)
                probs = {names[int(i)]: p for i, p in a["probabilities"].items()}
                label = max(probs, key=probs.get)
                out[f.name] = {
                    "label": label,
                    "p": probs[label],
                    "probs": probs,
                    "confidence": a.get("confidence"),
                    "expected_level": a.get("score"),
                }
            else:
                yes = a["noul"]
                out[f.name] = {
                    "label": yes >= 0.5,
                    "p": max(yes, 1 - yes),
                    "probs": {True: yes, False: 1 - yes},
                    "confidence": None,
                }
        return out


# --- Together LLMs -----------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a customer support triage classifier. Read the customer's message and "
    "answer every question below. Choose only from the listed options. "
    "Respond with JSON only."
)
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}


def llm_instructions(task: Task) -> str:
    parts = [SYSTEM_PROMPT]
    for f in task.fields:
        if f.kind == "signal":
            parts.append(f"\n## {f.name} (true or false)\n{f.instruction}")
            continue
        options = "\n".join(f"- {label}: {desc}" for label, desc in f.options.items())
        parts.append(f"\n## {f.name}\n{f.instruction}\nOptions:\n{options}")
    return "\n".join(parts)


def llm_schema(task: Task) -> dict:
    props = {
        f.name: {"type": "boolean"}
        if f.kind == "signal"
        else {"type": "string", "enum": list(f.options)}
        for f in task.fields
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


class TogetherBackend:
    def __init__(
        self, system: str, model: str, transport: httpx.AsyncBaseTransport | None = None
    ):
        self.system = system
        self.model = model
        self._transport = transport

    def build_payload(self, task: Task, row: dict) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": llm_instructions(task)},
                {"role": "user", "content": task.state(row)},
            ],
            "temperature": 0,
            "max_tokens": 300,
            "logprobs": 5,
            # Hybrid-thinking models (Qwen3.5) otherwise spend max_tokens on reasoning.
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": task.dataset,
                    "schema": llm_schema(task),
                    "strict": True,
                },
            },
        }

    async def call(self, payload: dict, max_attempts: int = 5) -> dict:
        headers = {"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"}
        async with httpx.AsyncClient(timeout=120, transport=self._transport) as client:
            for attempt in range(1, max_attempts + 1):
                t0 = time.perf_counter()
                resp = await client.post(
                    f"{config.TOGETHER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code in RETRY_STATUSES and attempt < max_attempts:
                    await asyncio.sleep(min(30, 2**attempt) + random.random())
                    continue
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                return {
                    "response": data,
                    # Latency of the successful attempt only; retries are counted separately.
                    "latency_ms": latency_ms,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "attempts": attempt,
                }
        raise RuntimeError("unreachable")

    def parse(self, task: Task, response: dict) -> Prediction:
        choice = (response.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        parsed = logprobs.parse_json(content) or {}
        tokens = logprobs.normalize(choice.get("logprobs"))
        field_p = logprobs.field_probabilities(
            content or "", tokens, [f.name for f in task.fields]
        )
        out: Prediction = {}
        for f in task.fields:
            value = parsed.get(f.name)
            valid = (
                isinstance(value, bool)
                if f.kind == "signal"
                else isinstance(value, str) and value in f.options
            )
            out[f.name] = {
                "label": value if valid else None,
                "p": field_p.get(f.name) if valid else None,
                "probs": None,
                "confidence": None,
            }
        return out


def get_backend(system: str) -> Backend:
    if system == "jev":
        route = config.jev_route()
        if route is None:
            raise SystemExit(
                "No Jev access: set TYPESAFE_API_KEY or OPENROUTER_API_KEY in .env"
            )
        return JevBackend(route)
    models = config.together_models()
    if system in models:
        return TogetherBackend(system, models[system])
    raise SystemExit(f"unknown system {system!r}; choose jev or {', '.join(models)}")


SYSTEMS = ("jev", "together-small", "together-large")
