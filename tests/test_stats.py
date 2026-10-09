import pytest

from agent_eval.stats import (
    calibration_metrics,
    cohen_kappa,
    confusion_counts,
    mcnemar_exact,
    paired_bootstrap,
)


class TestConfusionCounts:
    def test_known_values(self):
        c = confusion_counts(
            [True, False, True, False, True],
            [True, False, False, True, True],
        )
        assert c == {"tp": 2, "tn": 1, "fp": 1, "fn": 1}

    def test_empty(self):
        assert confusion_counts([], []) == {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


class TestCohenKappa:
    def test_perfect_agreement_is_one(self):
        y = [True, False, True, True, False]
        assert cohen_kappa(y, y) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        assert cohen_kappa([], []) == 0.0

    def test_known_value_85_percent_agreement(self):
        # n=20, 17/20 agree (85%). Cross-checked against
        # sklearn.metrics.cohen_kappa_score == 0.6808510638297873.
        y_true = [True] * 13 + [False] * 7
        y_pred = [False, False] + [True] * 11 + [True] + [False] * 6
        assert cohen_kappa(y_true, y_pred) == pytest.approx(0.6808510638297873)

    def test_golden_set_stub_disagreement(self):
        # 6/8 agreement with balanced marginals -> kappa 0.5
        y_true = [True, False, False, True, True, False, True, False]
        y_pred = [True, False, True, False, True, False, True, False]
        assert cohen_kappa(y_true, y_pred) == pytest.approx(0.5)

    def test_single_label_perfect_agreement_is_one(self):
        # pe == 1.0 branch: everyone agrees and there is only one label
        assert cohen_kappa([True, True, True], [True, True, True]) == 1.0

    def test_complete_disagreement_is_negative(self):
        y_true = [True, True, False, False]
        y_pred = [False, False, True, True]
        assert cohen_kappa(y_true, y_pred) < 0


class TestCalibrationMetrics:
    def test_known_values(self):
        # tp=3 tn=3 fp=1 fn=1 (stub judge on the 8-item golden set)
        y_true = [True, False, False, True, True, False, True, False]
        y_pred = [True, False, True, False, True, False, True, False]
        m = calibration_metrics(y_true, y_pred)
        assert m["n"] == 8
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["kappa"] == pytest.approx(0.5)
        assert m["tpr"] == pytest.approx(0.75)
        assert m["fpr"] == pytest.approx(0.25)
        assert m["fnr"] == pytest.approx(0.25)
        assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 3, 1, 1)
        for key in ("accuracy", "kappa", "tpr", "fpr", "fnr"):
            assert -1.0 <= m[key] <= 1.0

    def test_empty_is_zero(self):
        m = calibration_metrics([], [])
        assert m["accuracy"] == 0.0
        assert m["n"] == 0.0


class TestMcNemarExact:
    def test_no_flips_is_one(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_one_sided_six_flips(self):
        # two-sided exact: 2 * C(6,0)/2^6 = 1/32
        assert mcnemar_exact(6, 0) == pytest.approx(0.03125)

    def test_single_flip_not_significant(self):
        assert mcnemar_exact(1, 0) == pytest.approx(1.0)

    def test_asymmetric_nine_vs_one(self):
        # two-sided exact: 2 * (C(10,0)+C(10,1)) / 2^10 = 22/1024
        assert mcnemar_exact(9, 1) == pytest.approx(22 / 1024)

    def test_symmetric(self):
        assert mcnemar_exact(9, 1) == mcnemar_exact(1, 9)

    def test_balanced_flips_not_significant(self):
        assert mcnemar_exact(5, 5) == pytest.approx(1.0)


class TestPairedBootstrap:
    def test_reproducible_with_fixed_seed(self):
        a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        b = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        assert paired_bootstrap(a, b, seed=42) == paired_bootstrap(a, b, seed=42)

    def test_mean_difference(self):
        a = [1.0, 1.0, 1.0]
        b = [0.0, 1.0, 0.0]
        mean, lo, hi = paired_bootstrap(a, b, seed=0)
        assert mean == pytest.approx(1 / 3 - 1.0)
        assert lo <= mean <= hi

    def test_constant_regression_has_degenerate_ci(self):
        a = [1.0] * 8
        b = [0.0] * 8
        mean, lo, hi = paired_bootstrap(a, b, seed=0)
        assert (mean, lo, hi) == (-1.0, -1.0, -1.0)

    def test_ci_contains_true_difference(self):
        a = [0.9, 0.8, 1.0, 0.7, 0.95, 0.85, 1.0, 0.6]
        b = [v - 0.2 for v in a]
        mean, lo, hi = paired_bootstrap(a, b, seed=7)
        assert mean == pytest.approx(-0.2)
        assert lo <= -0.2 <= hi

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap([], [])
