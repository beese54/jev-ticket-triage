import pytest

from triage import metrics


def test_accuracy_and_invalid_count_as_wrong():
    assert metrics.accuracy(["a", "b", "c", "a"], ["a", "b", None, "b"]) == 0.5
    assert metrics.invalid_rate(["a", None, None, "b"]) == 0.5


def test_macro_f1_weights_classes_equally():
    gold = ["a"] * 8 + ["b"] * 2
    pred = ["a"] * 10  # majority guess: great on a, zero on b
    f1_a = 2 * 8 / (2 * 8 + 2)
    assert metrics.macro_f1(gold, pred) == pytest.approx((f1_a + 0.0) / 2)
    assert metrics.accuracy(gold, pred) == 0.8


def test_within_one_is_ordinal():
    order = ["low", "medium", "high"]
    gold = ["low", "low", "high", "medium"]
    pred = ["medium", "high", "high", None]
    assert metrics.within_one(gold, pred, order) == 0.5


def test_coverage_ranks_by_confidence_and_none_last():
    correct = [True, False, True, True]
    score = [0.9, 0.95, None, 0.5]
    curve = metrics.coverage_curve(correct, score, points=4)
    # ranked: 0.95(F), 0.9(T), 0.5(T), None(T)
    assert [c["accuracy"] for c in curve] == [0.0, 0.5, pytest.approx(2 / 3), 0.75]
    assert metrics.accuracy_at_coverage(correct, score, 0.5) == 0.5


def test_coverage_at_accuracy_finds_largest_share():
    correct = [True, True, True, False, True, False]
    score = [0.99, 0.98, 0.97, 0.96, 0.5, 0.4]
    # prefixes: 1/1, 2/2, 3/3, 3/4=.75, 4/5=.8, 4/6=.67
    assert metrics.coverage_at_accuracy(correct, score, 0.8) == pytest.approx(5 / 6)
    assert metrics.coverage_at_accuracy(correct, score, 1.0) == 0.5
    assert metrics.coverage_at_accuracy([False], [0.9], 0.5) == 0.0


def test_calibration_perfect_and_overconfident():
    assert metrics.calibration([True, False], [1.0, 0.0])["ece"] == 0.0
    over = metrics.calibration([True, False, True, False], [0.95] * 4)
    assert over["ece"] == pytest.approx(0.45)
    assert over["bins"] == [{"lo": 0.9, "hi": 1.0, "n": 4, "conf": 0.95, "acc": 0.5}]


def test_percentile_and_ops_summary():
    assert metrics.percentile([1, 2, 3, 4], 50) == 2.5
    recs = [
        {
            "latency_ms": 100,
            "input_tokens": 10,
            "output_tokens": 2,
            "cost_usd": 0.001,
            "attempts": 1,
        },
        {
            "latency_ms": 300,
            "input_tokens": 30,
            "output_tokens": 4,
            "cost_usd": 0.003,
            "attempts": 2,
        },
    ]
    ops = metrics.ops_summary(recs)
    assert ops["latency_ms_p50"] == 200
    assert ops["cost_per_1k_usd"] == pytest.approx(2.0)
    assert ops["retried_calls"] == 1


def test_ranked_pairs_matches_coverage_order():
    correct = [True, False, True, True]
    score = [0.9, 0.95, None, 0.5]
    assert metrics.ranked_pairs(correct, score) == [
        [0.95, 0],
        [0.9, 1],
        [0.5, 1],
        [None, 1],
    ]
