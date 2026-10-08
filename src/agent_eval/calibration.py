from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .judge import Judge, JudgeVerdict
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


def store_calibration(
    storage: Storage, report: CalibrationReport, judge_name: str, golden_path: str
) -> int:
    return storage.record_calibration(
        judge=judge_name,
        golden_path=golden_path,
        metrics=report.metrics,
        details=[
            {
                "id": r.item.id,
                "label": r.item.label,
                "judge_pass": r.verdict.passed,
                "score": r.verdict.score,
                "rationale": r.verdict.rationale,
                "agree": r.agree,
            }
            for r in report.items
        ],
    )
