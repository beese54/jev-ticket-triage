# Ticket Triage: Jev vs Together.ai LLMs — Plan

Status: **PLANNING — awaiting approval before any code** (2026-09-26)

## 1. The claim we are testing

> For structured triage decisions, a System One model (Jev) matches or approaches
> general-purpose LLM accuracy at a fraction of the cost and latency — and its
> confidence score lets you safely automate the easy tickets and route the rest to humans.

Every number in the LinkedIn post must come from a cached, reproducible run. If the
data says Jev loses on accuracy, the post says that.

## 2. Decisions made

| Decision | Choice |
|---|---|
| Dataset | **Hybrid**: Banking77 = clean-label accuracy benchmark; Tobi-Bueck (EN) = multi-field triage demo scored against a hand-adjudicated gold subset |
| Baselines | **Small + large** Together.ai models (e.g. ~8B and ~70B class; exact IDs picked from live catalog at build time) |
| Demo | **Static HTML dashboard** reading cached JSON only (zero API calls at view time) |

## 3. Dataset findings (from profiling, 2026-09-26)

**Tobi-Bueck/customer-support-tickets** (CC-BY-NC-4.0, synthetic, LLM-generated)
- The HF viewer merges 3 CSVs with different schemas. Use only
  `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (28,587 rows; 16,338 EN).
- Labels: queue (10), priority (low/medium/high), type (Incident/Request/Problem/Change).
- **Labels are noisy**: in an 18-row eyeball sample, ~⅓ of queue labels looked wrong
  (e.g. "security incident report" → Returns and Exchanges; "marketing strategy
  question" → Returns and Exchanges; "data breach" → medium priority).
  → Scoring against raw labels mostly measures noise. Hence the adjudicated gold subset.
- 8,399 bodies overlap with the older `4-20k` file (irrelevant if we only use `aa_`).
- ~8% missing subjects; handle as empty string.

**PolyAI/banking77** (CC-BY-4.0): real banking customer queries, 77 intents, 3,080-row
test split, clean labels, widely used benchmark. Stresses Jev's Choice with many options
(limit is 255).

## 4. System design

### 4.1 Questions per ticket

**Banking77** — 1 question
- `intent`: Choice over 77 labels (each with a one-line description)

**Tobi-Bueck triage** — 1 Jev request, fanned out:
- `queue`: Choice (10 queues, with descriptions from the dataset card)
- `priority`: Score (low → medium → high, with descriptive anchors)
- `type`: Choice (Incident / Request / Problem / Change, ITIL definitions)
- Extra Nouls for the demo (no gold labels, shown as signals only):
  `security_or_data_breach`, `service_outage`, `customer_frustrated`, `asks_for_refund`

### 4.2 Fairness rules (both systems get identical information)
- Same label names + descriptions, same ticket text (subject + body).
- LLMs: temperature 0, JSON-schema / JSON-mode output, one call per ticket for all fields.
- LLM confidence: derived from **logprobs** of the chosen label so confidence-gating
  is compared apples-to-apples (fallback: verbalized confidence, clearly labeled).
- Invalid/unparseable LLM output counts as wrong (and is reported separately).
- Latency measured the same way (wall clock per ticket, same machine, sequential +
  a batched/concurrent run reported separately).
- Cost from the token usage each API returns × published prices (recorded with date).

### 4.3 Metrics
| Metric | Why |
|---|---|
| Accuracy, macro-F1 (per field) | headline quality; macro-F1 because queues are imbalanced |
| Priority: exact + off-by-one | priority is ordinal |
| **Coverage vs accuracy curve** | "automate X% of tickets at Y% accuracy" — the key confidence story |
| Calibration (ECE, reliability plot) | is confidence trustworthy? |
| Latency p50 / p95 | speed claim |
| Cost per 1,000 tickets | cost claim |
| Parse-failure rate (LLMs) | structural reliability claim |

### 4.4 Eval discipline (Pattern R)
- Split per dataset: **dev** (prompt/description tuning, ~100) · **test** (reported,
  touched once per config) · **holdout** (~50, opened only at the end).
- Prompt/description changes logged one variable at a time in `tasks/eval_ledger.md`.
- Determinism check: 3 repeat runs on 50 tickets per system before trusting single runs.

### 4.5 Gold adjudication (Tobi-Bueck)
- 150 stratified EN test tickets. Claude pre-labels with a rubric; **user reviews every
  disagreement with the original label**. Neither Jev nor the tested Together models are
  used for labeling (avoids bias).
- Report both: accuracy vs adjudicated gold (headline) and vs raw labels (transparency),
  plus the measured raw-label noise rate (a result in itself).

### 4.6 Caching
- `cache/<system>/<dataset>/<ticket_id>.json` = {request, response, latency_ms,
  usage, timestamp, model_id, prompt_version}. The key includes a hash of the request,
  so a changed prompt → new entry, and old runs stay reproducible.
- The runner skips cached entries (resumable, no double spend).
- `results/*.json` = aggregated metrics the dashboard consumes.

### 4.7 Project layout (proposed)
```
jev_first_project/
  .env.example            TYPESAFE_API_KEY, TOGETHER_API_KEY
  pyproject.toml          uv-managed; typesafe-sdk, together/openai, pandas, datasets
  src/triage/
    data.py               load + clean + split both datasets
    labels.py             label sets + descriptions (single source of truth)
    jev_runner.py         builds Jev questions, calls API, writes cache
    llm_runner.py         same task via Together, JSON output + logprobs
    metrics.py            accuracy, F1, coverage curve, ECE, cost, latency
    report.py             cache → results/*.json
  data/splits/            frozen dev/test/holdout ids + adjudicated gold
  cache/                  raw API responses (committed; they ARE the evidence)
  results/                aggregated metrics
  demo/index.html         static dashboard (reads results + sample cache)
  tasks/todo.md, tasks/eval_ledger.md
```

## 5. Phases & checklist

- [x] **P0 Setup** (Together side) — German is out of scope (user decision 2026-09-26; EN only).
  - [x] git init, uv project (py3.12, typesafe-sdk 0.7.1, httpx), .gitignore, .env.example
  - [x] public GitHub repo; pre-commit hook blocks `.env` + any .env key value (tested both)
  - [x] Together models chosen (2026-09-26, serverless on this account, top-5 logprobs):
    - small `Qwen/Qwen3.5-9B` — $0.17 in / $0.25 out per M; needs
      `chat_template_kwargs.enable_thinking=false` or it burns max_tokens on reasoning
    - large `meta-llama/Llama-3.3-70B-Instruct-Turbo` — $1.04 in / $1.04 out per M
    - rejected: most catalog models are dedicated-endpoint only; Qwen3.7-Max / 3.8-Flash
      are streaming-only; DeepSeek-V4-Flash returns no top-k alternatives
  - [x] smoke run → `results/p0_smoke.json`: both parse JSON; label tokens carry real
        distributions (Qwen queue: Billing .86 / Technical .12 / Customer .03)
  - Gotchas: Together `/models` is a bare list (OpenAI client can't parse) → raw httpx;
    logprobs come in two shapes (Llama native vs Qwen OpenAI-style) → `normalize_logprobs`;
    special tokens may have null logprobs.
  - [ ] **BLOCKED: Jev smoke test** — TypeSafe signups restricted. Options: wait, or
        route via OpenRouter (`~typesafe/jev-latest`, same SDK, different base_url/key).
  - Note: Jev Score returns an expected value (e.g. 1.43) → use argmax of
    `probabilities` for the priority label.
- [x] **P1 Data** — `src/triage/data.py`, `scripts/p1_make_splits.py`, `tests/test_data.py` (8 pass).
  - Banking77 (official test split, balanced per intent): dev 154 · test 770 · holdout 77.
  - Tickets (EN, 16,306 after cleaning): dev 100 · test 600 (≥20/queue) · holdout 50;
    gold = 150 test rows (15/queue), marked `in_gold`.
  - Cleaning: literal `\n` / `<br>` → newlines; dropped 31 bodies <20 chars, 1 German-in-EN.
  - Splits are deterministic (seed 20260926; rebuild test asserts identical ids);
    manifest records source sha256 + label counts. Details: `data/README.md`.
  - Size: Banking77 ≈13 tokens/text, tickets ≈100 → prompts dominated by label
    descriptions; est. ≈$2 per full Together pass (both models, all splits).
- [ ] **P2 Gold** — adjudicate 150 Tobi-Bueck tickets (user review step).
- [ ] **P3 Runners** — Jev + Together runners with caching; smoke test on 10 dev tickets.
- [ ] **P4 Dev tuning** — label descriptions/prompt iterated on dev only; ledger rows.
- [ ] **P5 Test runs** — full test for Jev + 2 LLMs, both datasets; determinism check.
- [ ] **P6 Metrics** — report.py → results; holdout sanity check.
- [ ] **P7 Dashboard** — static HTML: headline cards (accuracy / cost / latency),
      coverage-vs-accuracy chart with confidence slider, ticket explorer
      (side-by-side answers + probabilities), methodology + caveats section.
- [ ] **P8 LinkedIn** — post draft (Pattern V), 1–2 charts, link to dashboard/repo.

## 6. Size & cost estimate (to confirm in P0)
- Banking77: ~1,000 stratified test + dev/holdout. Tobi-Bueck: ~600 EN test (150 gold).
- Jev: $0.042 / M input tokens → effectively cents for the whole project.
- Together: ~1,600 tickets × 3 runs-worth × ~1.5k tokens ≈ 7M tokens → roughly single-digit
  USD even for a 70B-class model. Hard cap to set in P0.

## 7. Risks
- **Label noise** (Tobi-Bueck) → mitigated by adjudicated gold; noise rate reported.
- **Jev jaggedness** — literal reading of instructions; weak at numbers/dates (not
  needed here). Label descriptions matter a lot → tuned on dev only.
- **Language** — Jev is best in English; German excluded from headline results.
- **Licensing** — Tobi-Bueck is CC-BY-NC: fine for a non-commercial post/demo with
  attribution; do not reuse commercially. Banking77 CC-BY-4.0.
- **Rate limits** — Jev dynamic limits (1,200 rpm); Together limits per tier → backoff.
- **Fair-comparison criticism** — addressed by §4.2 rules + publishing prompts & cache.

## 8. Definition of done
- Both datasets run end-to-end for 3 systems, 100% of test tickets cached.
- `results/` regenerates identically from `cache/` with one command.
- Dashboard loads offline with no API keys and every headline number traces to a results file.
- Methodology + caveats stated on the dashboard and in the post.
