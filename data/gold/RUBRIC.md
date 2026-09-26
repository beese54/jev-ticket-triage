# Gold-label rubric (P2)

The 150 `in_gold` test tickets (15 per queue) get hand-checked labels. The original dataset
labels are noisy (see `tasks/eval_ledger.md`), so headline ticket results are scored against these.

## Process
1. **Blind first pass (Claude):** labels each ticket from its text alone. It does not see the
   original labels or any system's predictions. Saved to `claude_labels.jsonl` with a one-line reason.
2. **Human review (repo owner):** every ticket where the first pass and the original label differ.
   The reviewer picks the original label, Claude's label, or another label. Agreements are accepted as-is.
3. The result is `data/splits/tickets_gold_labels.jsonl`, which `report.py` picks up automatically.

Neither Jev nor the Together models are used at any step, so the gold labels can't favour a system.

## Label definitions
These are the same label names and descriptions every system gets (`src/triage/labels.py`),
plus tie-break rules so that labelling is consistent.

### queue: which team should resolve it
| Queue | Use when |
|---|---|
| Technical Support | A technical fault or how-to with the vendor's software or service: bugs, errors, integrations, configuration |
| Product Support | Questions or issues about a specific product's features, compatibility, usage or performance |
| IT Support | The customer's own internal IT: their infrastructure, devices, networks, accounts, security tooling |
| Customer Service | Account or service matters, complaints, general service requests that aren't technical or billing |
| Billing and Payments | Invoices, charges, payment methods, refunds of charges, subscription billing |
| Returns and Exchanges | Sending a product back, exchanges, warranty replacement of a physical item |
| Sales and Pre-Sales | Buying, pricing, quotes, product information before purchase, upgrades to a paid plan |
| Human Resources | Employee matters: hiring, payroll, benefits, leave, HR systems as an HR process |
| Service Outages and Maintenance | A service is down or degraded for many users, or questions about planned maintenance |
| General Inquiry | Information requests that fit no team above, e.g. partnership, marketing strategy, general advice |

Tie-breaks: the team that must **act** wins over the topic mentioned. A security breach in the
customer's own systems → IT Support. A breach or outage in the vendor's service → Technical Support
or Service Outages (Outages if it is described as widespread or down). Marketing or strategy
advice with no product problem → General Inquiry.

### priority
| Level | Use when |
|---|---|
| high | Business-critical: security breach or data exposure, outage or blocked work, money at risk, explicit urgency with real impact |
| medium | Real impact but work continues, a workaround exists, or the issue is intermittent |
| low | Information request, advice, feature idea or minor issue with no stated impact |

### type (ITIL)
| Type | Use when |
|---|---|
| Incident | Something that worked is broken or disrupted now |
| Problem | A recurring or underlying issue whose root cause needs investigation (frequent, repeated, persistent despite fixes) |
| Request | A request for information, help, access or a standard service; nothing is broken |
| Change | A request to modify, add, enhance or integrate a feature, system or configuration |
