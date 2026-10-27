from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .runner import RunSummary


def write_json(path: str | Path, data: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summary_report(summary: RunSummary, manifest: dict, source: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": summary.run_id,
        "source": source,
        "manifest": manifest,
        "summary": {
            "total_cases": summary.total,
            "passed_cases": summary.passed,
            "mean_pass_rate": summary.mean_pass_rate,
            "samples_per_case": summary.samples_per_case,
            "error_samples": summary.error_count,
        },
        "cases": [
            {
                "id": report.case.id,
                "passed": report.passed,
                "pass_rate": report.pass_rate,
                "duration_ms": report.duration_ms,
                "cost_usd": report.cost_usd,
                "samples": [
                    {
                        "sample_idx": sample.sample_idx,
                        "status": sample.status,
                        "passed": sample.passed,
                        "error": sample.error,
                        "output": sample.output,
                        "trace_id": sample.trace_id,
                        "duration_ms": sample.duration_ms,
                        "cost_usd": sample.cost_usd,
                        "assertions": [
                            {
                                "name": result.name,
                                "passed": result.passed,
                                "detail": result.detail,
                            }
                            for result in sample.results
                        ],
                    }
                    for sample in report.samples
                ],
            }
            for report in summary.reports
        ],
    }


def write_junit(path: str | Path, report: dict) -> None:
    cases = report["cases"]
    samples = [sample for case in cases for sample in case["samples"]]
    failures = sum(
        sample["status"] == "completed" and not sample["passed"] for sample in samples
    )
    errors = sum(sample["status"] == "error" for sample in samples)
    suite = ET.Element(
        "testsuite",
        {
            "name": f"agent-eval:{report['run_id']}:{report['source']}",
            "tests": str(len(samples)),
            "failures": str(failures),
            "errors": str(errors),
            "time": f"{sum(s['duration_ms'] for s in samples) / 1000:.6f}",
        },
    )
    for case in cases:
        for sample in case["samples"]:
            testcase = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": case["id"],
                    "name": f"sample-{sample['sample_idx']}",
                    "time": f"{sample['duration_ms'] / 1000:.6f}",
                },
            )
            if sample["status"] == "error":
                element = ET.SubElement(testcase, "error", {"message": sample["error"]})
                element.text = sample["error"]
            elif not sample["passed"]:
                failed = [a for a in sample["assertions"] if not a["passed"]]
                element = ET.SubElement(
                    testcase,
                    "failure",
                    {"message": "; ".join(a["name"] for a in failed)},
                )
                element.text = "\n".join(
                    f"{a['name']}: {a['detail']}" for a in failed
                )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(destination, encoding="utf-8", xml_declaration=True)
