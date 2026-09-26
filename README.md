# Jev vs LLMs: customer support ticket triage

Does a **System One** model ([TypeSafe Jev](https://docs.typesafe.ai/introduction)) triage support
tickets as well as general-purpose LLMs, at a fraction of the cost and latency? And can its
confidence score decide which tickets are safe to automate?

This repo is a reproducible, head-to-head evaluation:

| System | Role |
|---|---|
| TypeSafe `jev-latest` | Choice / Score / Noul questions, one request per ticket |
| `Qwen/Qwen3.5-9B` (Together.ai) | small LLM baseline, JSON output + logprob confidence |
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Together.ai) | large LLM baseline, same prompt |

**Status:** work in progress. See [`tasks/todo.md`](tasks/todo.md) for the plan and progress.

## Datasets
- [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC-BY-4.0): clean-label intent benchmark.
- [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
  (CC-BY-NC-4.0, synthetic): multi-field triage (queue, priority, type). The original labels are noisy,
  so results are also reported against a hand-adjudicated gold subset.

## Setup
```sh
uv sync
cp .env.example .env                    # add TOGETHER_API_KEY (and TYPESAFE_API_KEY when available)
git config core.hooksPath scripts/hooks # pre-commit hook that blocks committing secrets
uv run python scripts/p0_smoke.py       # one test call per API -> results/p0_smoke.json
```

## License
Code: MIT. Datasets keep their own licenses (Tobi-Bueck is non-commercial).
