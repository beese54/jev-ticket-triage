"""P4 (system-neutral): how does the dataset use its labels, and how learnable are they?

Uses only the English tickets that are in NO split (the "pool"), plus the dev split for
scoring. Test and holdout are never read. No LLM or Jev output is used.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from triage import data

tickets, _ = data.load_tickets(data.RAW_DIR / data.TICKETS_FILE)
used = set().union(
    *(set(data.read_split("tickets", s)["id"]) for s in ("dev", "test", "holdout"))
)
pool = tickets[~tickets["id"].isin(used)].reset_index(drop=True)
dev = data.read_split("tickets", "dev")
text = lambda df: (df["subject"].fillna("") + " \n " + df["body"]).str.lower()
print(f"pool {len(pool)} tickets (disjoint from dev/test/holdout), dev {len(dev)}\n")

vec = TfidfVectorizer(
    ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True
)
Xp, Xd = vec.fit_transform(text(pool)), vec.transform(text(dev))
vocab = np.array(vec.get_feature_names_out())

print("== Learnability: TF-IDF + logistic regression trained on pool, scored on dev ==")
for field in ("queue", "priority", "type"):
    clf = LogisticRegression(max_iter=2000, C=4).fit(Xp, pool[field])
    acc = (clf.predict(Xd) == dev[field]).mean()
    majority = (dev[field] == pool[field].mode()[0]).mean()
    print(f"{field:<9} supervised {acc:.0%} | majority-class {majority:.0%}")
    if field == "type":
        type_clf = clf
print()

print("== Most indicative words per type (logistic regression weights) ==")
for i, cls in enumerate(type_clf.classes_):
    top = vocab[np.argsort(type_clf.coef_[i])[-15:][::-1]]
    print(f"{cls:<9}", ", ".join(top))
print()

print("== type x priority in pool (row %) ==")
print((pd.crosstab(pool["type"], pool["priority"], normalize="index") * 100).round(0))
print()

rng = np.random.default_rng(0)
for cls in ("Incident", "Problem", "Request", "Change"):
    print(f"== 5 random pool examples labelled {cls} ==")
    for _, r in pool[pool["type"] == cls].sample(5, random_state=1).iterrows():
        body = r["body"].replace("\n", " ")[:170]
        print(f"  [{r['queue']} | {r['priority']}] {r['subject'][:60]} :: {body}")
    print()

print("== Supervised type confusion on dev (rows = gold) ==")
print(
    pd.crosstab(dev["type"], type_clf.predict(Xd), rownames=["gold"], colnames=["pred"])
)
print()

prio_clf = LogisticRegression(max_iter=2000, C=4).fit(Xp, pool["priority"])
print("== Most indicative words per priority ==")
for i, cls in enumerate(prio_clf.classes_):
    top = vocab[np.argsort(prio_clf.coef_[i])[-15:][::-1]]
    print(f"{cls:<7}", ", ".join(top))
print()

print("== How often key phrases appear, by type (% of pool tickets) ==")
phrases = {
    "incident": r"\bincident",
    "problem": r"\bproblem",
    "despite": r"\bdespite",
    "could you / please provide": r"could you|please provide|provide (?:more )?(?:details|information)",
    "enhance/improve/update": r"\benhanc|\bimprov|\bupdate\b|\bupgrade",
    "outage/breach/crash": r"outage|breach|crash",
    "request (word)": r"\brequest",
}
t = text(pool)
print(
    pd.DataFrame(
        {
            k: t.str.contains(v).groupby(pool["type"]).mean().mul(100).round(0)
            for k, v in phrases.items()
        }
    )
)
