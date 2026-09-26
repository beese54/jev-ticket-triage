# Eval ledger

Pattern R discipline: one variable per iteration, a written prediction **before** the change,
scored on dev only. Test and holdout are not used here.

## Phase 0: what good looks like
- **Gradeable outputs:** exact-match label per field (queue, priority, type, Banking77 intent),
  macro-F1, priority off-by-one, `p` calibration (ECE), coverage vs accuracy.
- **Failure modes:** invalid or unparseable label (counted wrong); labels that follow a generic
  definition instead of the dataset's convention; label noise (queue/priority) mistaken for
  model error.
- **Budget:** Together total ≤ $10 (enforced by the cache). Latency is reported, not optimized.

## Fairness rule for tuning (decided 2026-09-26)
Label descriptions are shared by all systems. Until Jev can run, descriptions may change
**only from dataset evidence** (pool tickets outside every split, or the dataset card), never
from Together model outputs. LLM dev runs are used to *measure* a change, not to design it.

## Ceiling reference (analysis/label_semantics.py, no LLMs)
TF-IDF + logistic regression trained on 15,556 pool tickets (outside every split), scored on dev (n=100):

| Field | Supervised | Majority class |
|---|---|---|
| queue | 63% | 29% |
| priority | 61% | 38% |
| type | 93% | 46% |

So type is highly learnable (its labels are consistent). Queue and priority depend only
weakly on the text, which fits the label noise found in P1. A supervised model trained on
15k examples is a rough upper reference for zero-shot systems, not a hard ceiling.

## Iterations: tickets, dev (n=100), accuracy vs original labels

| # | Change | Qwen3.5-9B q / p / t | Llama-3.3-70B q / p / t | Spend | Verdict |
|---|---|---|---|---|---|
| 0 | baseline: ITIL-style type descriptions | 39 / 40 / 56 | 35 / 42 / 65 | $0.059 | — |
| 1 | type descriptions rewritten to follow the dataset's convention | 42 / 36 / 55 | 35 / 42 / 59 | $0.071 | **REVERT** |

### Iteration 1: type descriptions follow the dataset's convention
**Evidence** (pool only): Change tickets use enhance/improve/update wording 93% of the time
and include the word "request" 66% of the time. Request tickets ask for information
("could you provide…", 67%) and never describe anything broken. Incident tickets report an
event (outage, breach, crash, access attempt). Problem tickets describe something ongoing
or recurring ("facing", "despite", "due to", "frequent").

**Iteration 0 errors (type):** Change→Request 7/7 (small) and 5/7 (large); broken-thing→Request
13 (small) and 3 (large); Incident↔Problem 24 (small) and 27 (large).

**Prediction** (written before the change):
- Change→Request mostly moves to Change (Change row ≥5/7 correct for both).
- Incident/Problem→Request drops sharply for the small model (13 → ≤4).
- Incident↔Problem improves only modestly; even the supervised model confuses these.
- Risk: some Request tickets containing "update" words drift to Change (Request row may fall below 21/21).
- Queue and priority: no intended effect. The shared prompt changed, so a move of up to ±5 pts is noise.

**Result:** the prediction was mostly wrong, so this iteration is **reverted** (type 56→55 small, 65→59 large).
- Change→Request was barely fixed: 3/7 (small) and 2/7 (large) correct. The new Request wording
  ("nothing is reported as broken") fits Change tickets just as well.
- Incident→Problem got much worse: 18→30 (small) and 16→29 (large). Incident tickets in this dataset
  also describe troubleshooting and suspected causes, so the new Problem wording pulled them in.
- Request stayed at 21/21. Queue and priority moved within noise (±4 pts).

**Lesson:** word patterns that separate classes for a supervised model (trained on 15k examples)
don't translate into definitions a model can apply without examples. The type labels follow the
dataset generator's style more than any definition a reader could apply. This is a finding to report,
not something to keep tuning. Further rewrites would overfit 100 dev tickets.

**Decision:** freeze the iteration-0 descriptions (confirmed restored byte-for-byte: 100/100 cache
hits, identical scores). Iteration-1 responses stay in the cache as evidence. The type gap
(supervised 93% vs zero-shot 56–65%) is reported as a result about label conventions, and it applies
equally to Jev.
