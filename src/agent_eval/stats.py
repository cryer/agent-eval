from __future__ import annotations

import math
import random


def confusion_counts(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def cohen_kappa(y_true: list[bool], y_pred: list[bool]) -> float:
    """Agreement between judge and human labels, corrected for chance."""
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    true_pos = sum(y_true) / n
    pred_pos = sum(y_pred) / n
    pe = true_pos * pred_pos + (1 - true_pos) * (1 - pred_pos)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def calibration_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    """How well a judge reproduces human labels on a golden set."""
    c = confusion_counts(y_true, y_pred)
    n = len(y_true)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    return {
        "n": float(n),
        "accuracy": (tp + tn) / n if n else 0.0,
        "kappa": cohen_kappa(y_true, y_pred),
        "tpr": tp / (tp + fn) if tp + fn else 0.0,  # recall on "pass"
        "fpr": fp / (fp + tn) if fp + tn else 0.0,  # judge passes what humans fail
        "fnr": fn / (fn + tp) if fn + tp else 0.0,  # judge fails what humans pass
        **{k: float(v) for k, v in c.items()},
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = cases that passed in run A but failed in run B (regression flips)
    c = cases that failed in run A but passed in run B (improvement flips)
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(
    values_a: list[float],
    values_b: list[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over per-case pass rates.

    Returns (mean difference B-A, 95% CI low, 95% CI high).
    """
    if len(values_a) != len(values_b) or not values_a:
        raise ValueError("paired_bootstrap needs two non-empty equal-length series")
    rng = random.Random(seed)
    n = len(values_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(values_b[i] - values_a[i] for i in idx) / n)
    diffs.sort()
    mean = sum(values_b) / n - sum(values_a) / n
    return mean, diffs[int(0.025 * n_boot)], diffs[min(n_boot - 1, int(0.975 * n_boot))]
