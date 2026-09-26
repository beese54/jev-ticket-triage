# Data

`splits/` holds the frozen evaluation sets. Every system is scored on exactly these tickets.
Rebuild them with `uv run python scripts/p1_make_splits.py`. The output is deterministic, and
`manifest.json` records the source file hashes, what cleaning dropped, and label counts.
Raw downloads go to `raw/`, which is gitignored.

| File | Rows | Use |
|---|---|---|
| `banking77_dev.jsonl` | 154 (2 × 77 intents) | label descriptions/prompt tuning |
| `banking77_test.jsonl` | 770 (10 × 77) | reported results |
| `banking77_holdout.jsonl` | 77 (1 × 77) | final overfitting check, not looked at until then |
| `tickets_dev.jsonl` | 100 | prompt tuning |
| `tickets_test.jsonl` | 600 (≥20 per queue; `in_gold` marks 150 = 15 per queue) | reported results |
| `tickets_holdout.jsonl` | 50 | final overfitting check |

## Sources and licenses

- **Banking77**: Casanueva et al., 2020, *Efficient Intent Detection with Dual Sentence
  Encoders* ([paper](https://arxiv.org/abs/2003.04807),
  [data](https://github.com/PolyAI-LDN/task-specific-datasets)). CC-BY-4.0.
  Drawn from the official test split.
- **Customer Support Tickets**: Tobias Bueck,
  [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
  DOI 10.57967/hf/6184. **CC-BY-NC-4.0**, so the subsets here are also non-commercial only.
  Synthetic data. Only the English rows of `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
  are used.

## Cleaning applied to the tickets
- Literal `\n` sequences and `<br>` tags are converted to real newlines, and repeated whitespace is collapsed.
- Bodies under 20 characters are dropped (e.g. "Our system"): there is too little text to triage.
- One ticket labelled `en` that is actually German is dropped.
- Placeholders such as `<name>` are left as-is.
- Ticket ids are `tb-` plus the first 10 hex characters of sha1(subject + body), so they stay stable across rebuilds.

## Known label noise
The original ticket labels are noisy: in a spot check, about 1 in 3 queue labels looked wrong.
Results are therefore reported against a hand-adjudicated gold subset (the `in_gold` rows), and
the raw-label numbers are reported next to them for transparency.
