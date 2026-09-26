"""Turn Together token logprobs into per-field label probabilities."""

import json
import math
import re


def normalize(lp: dict | None) -> list[dict]:
    """Together returns two logprob shapes depending on the model:
    native {tokens, token_logprobs, top_logprobs: [{tok: lp}]} (e.g. Llama) or
    OpenAI-style {content: [{token, logprob, top_logprobs: [{token, logprob}]}]} (e.g. Qwen).
    Normalize to [{token, p, top: {token: p}}]; tokens with a null logprob
    (special tokens such as <|eot_id|>) are dropped."""
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
                    if a.get("logprob") is not None
                },
            }
            for t in lp["content"] or []
            if t.get("logprob") is not None
        ]
    tops = lp.get("top_logprobs") or [{}] * len(lp.get("tokens", []))
    return [
        {
            "token": tok,
            "p": math.exp(l),
            "top": {k: math.exp(v) for k, v in (top or {}).items() if v is not None},
        }
        for tok, l, top in zip(lp["tokens"], lp["token_logprobs"], tops)
        if l is not None
    ]


def _align(content: str, tokens: list[dict]) -> list[tuple[int, int, float]] | None:
    """Char span of each token within content, skipping special tokens that are
    not part of the text. None if the tokens don't reproduce the content."""
    spans, pos = [], 0
    for t in tokens:
        text = t["token"]
        if content.startswith(text, pos):
            spans.append((pos, pos + len(text), t["p"]))
            pos += len(text)
        elif text.startswith("<|") or text == "":
            continue
        else:
            return None
    return spans if pos == len(content) else None


def value_span(content: str, field: str) -> tuple[int, int] | None:
    """Char span of a field's value: the text inside the quotes, or a bare true/false."""
    m = re.search(rf'"{re.escape(field)}"\s*:\s*(?:"([^"]*)"|(true|false))', content)
    if not m:
        return None
    group = 1 if m.group(1) is not None else 2
    return m.span(group)


def field_probabilities(
    content: str, tokens: list[dict], fields: list[str]
) -> dict[str, float | None]:
    """P(generated value) per field = product of the probabilities of the tokens
    that overlap the value's characters. None when it can't be located."""
    spans = _align(content, tokens)
    out: dict[str, float | None] = {}
    for field in fields:
        span = value_span(content, field) if spans is not None else None
        if span is None:
            out[field] = None
            continue
        start, end = span
        p = 1.0
        for t_start, t_end, t_p in spans:
            if t_start < end and t_end > start:
                p *= t_p
        out[field] = p
    return out


def parse_json(content: str | None) -> dict | None:
    try:
        parsed = json.loads(content or "")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
