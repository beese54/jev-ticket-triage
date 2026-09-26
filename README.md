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

**Status:** work in progress. The Together.ai baselines are done. Jev results are pending
TypeSafe API access, and hand-adjudicated ticket labels are pending.
Plan and progress: [`tasks/todo.md`](tasks/todo.md). Every tuning decision: [`tasks/eval_ledger.md`](tasks/eval_ledger.md).

## Datasets
- [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) (CC-BY-4.0): clean-label intent benchmark.
- [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
  (CC-BY-NC-4.0, synthetic): multi-field triage (queue, priority, type). The original labels are noisy,
  so results are also reported against a hand-adjudicated gold subset.

Frozen splits live in [`data/splits/`](data/README.md).

## How the comparison is kept fair
- Every system gets the same ticket text and the same label names and descriptions, all from one file:
  [`src/triage/labels.py`](src/triage/labels.py).
- Jev asks typed questions (Choice / Score / Noul), all in one request per ticket. The LLMs run at
  temperature 0 and must answer in a JSON format that only allows the listed labels. An invalid answer counts as wrong.
- Confidence is compared on a common signal: the probability each system gives its chosen label
  (for the LLMs, taken from token logprobs). Jev's own confidence score is reported as well.
- Tuning happens on dev only, one change at a time, and every change and its result is logged. Test is
  run once per frozen configuration. LLM runs are repeated 3 times where results vary between runs.

## Layout
| Path | What |
|---|---|
| `src/triage/` | labels, tasks, backends (Jev SDK / Together HTTP), runner, cache, metrics, report |
| `cache/` | **every raw API response** (gzipped JSON), keyed by a hash of the exact request |
| `results/` | metrics and explorer JSON, regenerated from `cache/` |
| `dashboard/`, `docs/` | dashboard template and the built page (GitHub Pages) |
| `analysis/` | label-semantics, confusion and determinism analyses |

## Reproduce
```sh
uv sync
cp .env.example .env                    # add TOGETHER_API_KEY (and TYPESAFE_API_KEY when available)
git config core.hooksPath scripts/hooks # pre-commit hook that blocks committing secrets

# rebuild the numbers and the dashboard from the committed cache: no API keys needed
uv run python scripts/report.py --split test
uv run python scripts/build_dashboard.py --split test

# or re-run a system (cached calls are reused, new ones are saved)
uv run python scripts/run.py --system together-small --dataset banking77 --split test
uv run python scripts/run.py --system jev --dataset tickets --split test
uv run pytest
```

## License
Code: MIT. Datasets keep their own licenses (Tobi-Bueck is non-commercial).
