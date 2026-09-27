"""Systems under test, prices and runtime settings."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


@dataclass(frozen=True)
class Price:
    """USD per million tokens, with the date it was read from the provider."""

    input_per_m: float
    output_per_m: float
    as_of: str
    source: str


# Together prices from GET /v1/models; Jev from docs.typesafe.ai/models
# (input price only is published; output is recorded as 0 until confirmed).
PRICES: dict[str, Price] = {
    "Qwen/Qwen3.5-9B": Price(0.17, 0.25, "2026-09-26", "together /v1/models"),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": Price(
        1.04, 1.04, "2026-09-26", "together /v1/models"
    ),
    "jev-latest": Price(0.042, 0.0, "2026-09-26", "docs.typesafe.ai/models"),
    # Same model via OpenRouter; confirmed by the account's usage delta for one call
    # (764 input tokens -> $0.000032088 = $0.042/M input, output not billed).
    "~typesafe/jev-latest": Price(0.042, 0.0, "2026-09-27", "openrouter credits delta"),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1e6


def together_models() -> dict[str, str]:
    """System name -> Together model id."""
    return {
        "together-small": os.environ.get("TOGETHER_SMALL_MODEL", "Qwen/Qwen3.5-9B"),
        "together-large": os.environ.get(
            "TOGETHER_LARGE_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        ),
    }


def together_budget_usd() -> float:
    return float(os.environ.get("TOGETHER_BUDGET_USD", "10"))


@dataclass(frozen=True)
class JevRoute:
    api_key: str
    base_url: str | None
    model: str
    route: str  # "typesafe" or "openrouter"


def jev_route() -> JevRoute | None:
    """Direct TypeSafe key if present, otherwise OpenRouter; None if neither."""
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return JevRoute(key, None, "jev-latest", "typesafe")
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return JevRoute(key, OPENROUTER_BASE_URL, "~typesafe/jev-latest", "openrouter")
    return None
