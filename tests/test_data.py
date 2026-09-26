import pandas as pd
import pytest

from triage import data


def test_clean_text_normalizes_escapes():
    assert data.clean_text("Hi,\\n\\nthanks<br>bye<BR/>x") == "Hi,\n\nthanks\nbye\nx"
    assert data.clean_text("a   b\n\n\n\nc") == "a b\n\nc"
    assert data.clean_text(float("nan")) == ""


def test_allocate_respects_total_and_floor():
    counts = pd.Series({"big": 900, "mid": 90, "tiny": 10})
    alloc = data._allocate(counts, total=100, floor=20)
    assert sum(alloc.values()) == 100
    assert min(alloc.values()) >= 20
    assert alloc["big"] > alloc["mid"] >= alloc["tiny"]


@pytest.fixture(scope="module")
def splits():
    return {
        (ds, sp): data.read_split(ds, sp)
        for ds in ("banking77", "tickets")
        for sp in ("dev", "test", "holdout")
    }


@pytest.mark.parametrize("dataset", ["banking77", "tickets"])
def test_splits_are_disjoint(splits, dataset):
    ids = [set(splits[(dataset, sp)]["id"]) for sp in ("dev", "test", "holdout")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_banking77_is_balanced(splits):
    for split, per_class in data.B77_PER_CLASS.items():
        counts = splits[("banking77", split)]["intent"].value_counts()
        assert len(counts) == 77
        assert (counts == per_class).all()


def test_ticket_split_sizes_and_gold(splits):
    assert len(splits[("tickets", "dev")]) == data.TICKETS_DEV
    assert len(splits[("tickets", "holdout")]) == data.TICKETS_HOLDOUT
    test = splits[("tickets", "test")]
    assert len(test) == data.TICKETS_TEST
    assert test["queue"].value_counts().min() >= data.TICKETS_TEST_FLOOR
    gold = test[test["in_gold"]]
    assert (gold["queue"].value_counts() == data.GOLD_PER_QUEUE).all()
    assert gold["queue"].nunique() == 10


def test_ticket_text_is_clean(splits):
    for sp in ("dev", "test", "holdout"):
        body = splits[("tickets", sp)]["body"]
        assert not body.str.contains("\\n", regex=False).any()
        assert not body.str.contains("<br", case=False, regex=False).any()
        assert (body.str.len() >= data.MIN_BODY_CHARS).all()


@pytest.mark.skipif(
    not (data.RAW_DIR / data.TICKETS_FILE).exists(), reason="raw data not downloaded"
)
def test_splits_rebuild_identically(splits):
    tickets, _ = data.load_tickets(data.RAW_DIR / data.TICKETS_FILE)
    rebuilt = data.split_tickets(tickets)
    for sp, df in rebuilt.items():
        assert list(df["id"]) == list(splits[("tickets", sp)]["id"])
    b77 = data.split_banking77(data.load_banking77(data.RAW_DIR / "banking77_test.csv"))
    for sp, df in b77.items():
        assert list(df["id"]) == list(splits[("banking77", sp)]["id"])
