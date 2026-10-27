from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .judge import Judge, JudgeVerdict
from .manifest import dataset_hash
from .stats import calibration_metrics
from .storage import Storage


@dataclass
class GoldenItem:
    id: str
    input: str
    output: str
    rubric: str
    label: bool  # human label: does the output satisfy the rubric?
    note: str = ""


@dataclass
class CalibrationItem:
    item: GoldenItem
    verdict: JudgeVerdict

    @property
    def agree(self) -> bool:
        return self.verdict.passed == self.item.label


@dataclass
class CalibrationReport:
    items: list[CalibrationItem]
    metrics: dict[str, float]


def load_golden(path: str | Path) -> list[GoldenItem]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"golden set must be a YAML list, got {type(raw).__name__}")
    items = []
    for entry in raw:
        label = entry["label"]
        if label not in ("pass", "fail"):
            raise ValueError(f"golden item {entry.get('id')!r}: label must be 'pass' or 'fail'")
        items.append(
            GoldenItem(
                id=entry["id"],
                input=entry.get("input", ""),
                output=entry["output"],
                rubric=entry["rubric"],
                label=label == "pass",
                note=entry.get("note", ""),
            )
        )
    return items


def run_calibration(items: list[GoldenItem], judge: Judge) -> CalibrationReport:
    results = [
        CalibrationItem(item, judge.judge(item.input, item.output, item.rubric))
        for item in items
    ]
    metrics = calibration_metrics(
        [r.item.label for r in results],
        [r.verdict.passed for r in results],
    )
    return CalibrationReport(items=results, metrics=metrics)


def evaluate_admission(
    metrics: dict[str, float],
    *,
    min_samples: int = 50,
    min_per_class: int = 10,
    min_kappa: float = 0.6,
    max_fpr: float = 0.05,
) -> dict:
    if min_samples < 1 or min_per_class < 1:
        raise ValueError("calibration sample thresholds must be at least 1")
    if not -1 <= min_kappa <= 1:
        raise ValueError("min_kappa must be between -1 and 1")
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be between 0 and 1")
    human_pass = int(metrics["tp"] + metrics["fn"])
    human_fail = int(metrics["tn"] + metrics["fp"])
    reasons = []
    if metrics["n"] < min_samples:
        reasons.append(f"sample count {int(metrics['n'])} is below required {min_samples}")
    if min(human_pass, human_fail) < min_per_class:
        reasons.append(
            f"human pass/fail counts {human_pass}/{human_fail} require at least"
            f" {min_per_class} per class"
        )
    if metrics["kappa"] < min_kappa:
        reasons.append(f"kappa {metrics['kappa']:.4f} is below required {min_kappa:.4f}")
    if metrics["fpr"] > max_fpr:
        reasons.append(f"FPR {metrics['fpr']:.4f} exceeds allowed {max_fpr:.4f}")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "policy": {
            "min_samples": min_samples,
            "min_per_class": min_per_class,
            "min_kappa": min_kappa,
            "max_fpr": max_fpr,
        },
    }


def calibration_details(report: CalibrationReport) -> list[dict]:
    return [
        {
            "id": r.item.id,
            "label": r.item.label,
            "judge_pass": r.verdict.passed,
            "score": r.verdict.score,
            "rationale": r.verdict.rationale,
            "agree": r.agree,
        }
        for r in report.items
    ]


def store_calibration(
    storage: Storage,
    report: CalibrationReport,
    judge_name: str,
    golden_path: str,
    *,
    judge_config: dict | None = None,
    admission: dict | None = None,
) -> int:
    return storage.record_calibration(
        judge=judge_name,
        golden_path=golden_path,
        golden_hash=dataset_hash(golden_path),
        judge_config=judge_config or {},
        admission=admission or {},
        metrics=report.metrics,
        details=calibration_details(report),
    )
