import json
import sqlite3
import sys
import types

import pytest
import yaml

from agent_eval.calibration import evaluate_admission
from agent_eval.cli import main
from agent_eval.compare import analyze_run_results
from agent_eval.manifest import judge_config
from agent_eval.stats import classify_delta
from agent_eval.storage import Storage


def result_rows(values, samples=1):
    return [
        {"case_id": f"c{index}", "passed": value}
        for index, case_values in enumerate(values)
        for value in (case_values if samples > 1 else [case_values])
    ]


class TestStatisticalDecision:
    def test_non_inferiority_boundaries(self):
        assert classify_delta(-0.05, -0.05, -0.05, 0.05) == "no_meaningful_regression"
        assert classify_delta(-0.1, -0.2, -0.05, 0.05) == "inconclusive"
        assert classify_delta(-0.1, -0.2, -0.0501, 0.05) == "significant_regression"
        assert classify_delta(-0.1, -0.2, 0.0, 0.05) == "inconclusive"
        assert classify_delta(0.1, 0.0501, 0.2, 0.05) == "significant_improvement"

    def test_underpowered_regression_is_inconclusive(self):
        analysis = analyze_run_results(result_rows([1] * 8), result_rows([0] * 8))
        assert analysis["verdict"] == "inconclusive"
        assert analysis["return_code"] == 3
        assert "only 8 common cases" in analysis["gate_reasons"][0]

    def test_multi_sample_rates_are_clustered_by_case(self):
        rows_a = result_rows([[1, 1], [1, 0]], samples=2)
        rows_b = result_rows([[1, 0], [0, 0]], samples=2)
        analysis = analyze_run_results(rows_a, rows_b, min_cases=2)
        assert [case["pass_rate_a"] for case in analysis["cases"]] == [1.0, 0.5]
        assert [case["pass_rate_b"] for case in analysis["cases"]] == [0.5, 0.0]
        assert analysis["metrics"]["mean_pass_rate_delta"] == -0.5
        assert analysis["metrics"]["inference"] == "paired_case_bootstrap"
        assert analysis["metrics"]["mcnemar_p_value"] is None

    def test_single_sample_mcnemar_is_diagnostic(self):
        analysis = analyze_run_results(result_rows([1, 1]), result_rows([0, 1]), min_cases=2)
        assert analysis["metrics"]["mcnemar_p_value"] == 1.0
        assert analysis["metrics"]["mcnemar_regression_flips"] == 1

    def test_custom_confidence_and_seed_are_reproducible(self):
        kwargs = dict(min_cases=2, alpha=0.1, n_boot=200, seed=42)
        a = result_rows([1, 0])
        b = result_rows([0, 1])
        first = analyze_run_results(a, b, **kwargs)
        assert first == analyze_run_results(a, b, **kwargs)
        assert first["metrics"]["confidence_level"] == pytest.approx(0.9)
        assert first["metrics"]["bootstrap_samples"] == 200


class TestCalibrationAdmission:
    def metrics(self, **overrides):
        values = {
            "n": 50,
            "tp": 20,
            "fn": 5,
            "tn": 24,
            "fp": 1,
            "kappa": 0.8,
            "fpr": 1 / 25,
        }
        values.update(overrides)
        return values

    def test_admits_metrics_at_threshold(self):
        admission = evaluate_admission(self.metrics())
        assert admission["accepted"] is True
        assert admission["reasons"] == []

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"n": 49}, "sample count"),
            ({"tp": 9, "fn": 0}, "per class"),
            ({"kappa": 0.59}, "kappa"),
            ({"fpr": 0.051}, "FPR"),
        ],
    )
    def test_rejects_each_failed_threshold(self, overrides, message):
        admission = evaluate_admission(self.metrics(**overrides))
        assert admission["accepted"] is False
        assert message in " ".join(admission["reasons"])

    def test_missing_negative_class_does_not_pass_via_zero_fpr(self):
        admission = evaluate_admission(
            self.metrics(n=50, tp=50, fn=0, tn=0, fp=0, fpr=0),
            min_per_class=10,
        )
        assert admission["accepted"] is False
        assert "50/0" in admission["reasons"][0]

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="max_fpr"):
            evaluate_admission(self.metrics(), max_fpr=1.1)


class TestCalibrationStorage:
    def test_latest_matching_calibration(self, storage):
        matching = judge_config("stub", "ignored")
        other = judge_config("litellm", "k3")
        for config, accepted in ((other, True), (matching, True), (matching, False)):
            storage.record_calibration(
                "judge",
                "golden.yaml",
                {},
                [],
                judge_config=config,
                admission={"accepted": accepted},
            )
        latest = storage.latest_calibration(matching)
        assert latest["admission"]["accepted"] is False
        assert latest["id"] == 3

    def test_legacy_calibration_schema_is_migrated(self, tmp_path):
        path = tmp_path / "legacy.db"
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE calibrations (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " judge TEXT NOT NULL, golden_path TEXT NOT NULL, metrics TEXT NOT NULL,"
            " details TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO calibrations (judge, golden_path, metrics, details, created_at)"
            " VALUES ('stub', 'old.yaml', '{}', '[]', '2025-01-01')"
        )
        connection.commit()
        connection.close()
        migrated = Storage(str(path))
        try:
            row = migrated.list_calibrations()[0]
            assert row["golden_hash"] == ""
            assert row["judge_config"] == {}
            assert row["admission"] == {}
        finally:
            migrated.close()


@pytest.fixture
def gate_agent(monkeypatch):
    module = types.ModuleType("decision_gate_agent")
    module.agent = lambda text: "hello"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def judged_case(tmp_path):
    path = tmp_path / "judged.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "j1",
                    "input": "question",
                    "assertions": [{"judge": {"rubric": 'must contain "hello"'}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def run_pair(db, cases):
    for run_id in ("A", "B"):
        assert main(
            [
                "run",
                "--cases",
                cases,
                "--agent",
                "decision_gate_agent:agent",
                "--judge-backend",
                "stub",
                "--db",
                db,
                "--run-id",
                run_id,
            ]
        ) == 0


def compare_with_calibration(db):
    return main(
        ["compare", "A", "B", "--db", db, "--min-cases", "1", "--require-calibration"]
    )


class TestCalibrationCliGate:
    def test_missing_then_admitted_calibration(self, tmp_path, db_path, gate_agent):
        run_pair(db_path, judged_case(tmp_path))
        assert compare_with_calibration(db_path) == 1
        report_path = tmp_path / "calibration.json"
        assert main(
            [
                "calibrate",
                "--golden",
                "examples/golden.yaml",
                "--judge-backend",
                "stub",
                "--db",
                db_path,
                "--min-samples",
                "8",
                "--min-per-class",
                "4",
                "--min-kappa",
                "0",
                "--max-fpr",
                "1",
                "--json-out",
                str(report_path),
            ]
        ) == 0
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["admission"]["accepted"] is True
        assert len(report["golden_hash"]) == 64
        assert report["judge"]["prompt_sha256"] == judge_config("stub", None)["prompt_sha256"]
        assert compare_with_calibration(db_path) == 0

    def test_latest_rejected_calibration_blocks(self, tmp_path, db_path, gate_agent):
        run_pair(db_path, judged_case(tmp_path))
        base = [
            "calibrate",
            "--golden",
            "examples/golden.yaml",
            "--judge-backend",
            "stub",
            "--db",
            db_path,
        ]
        assert main(
            base
            + ["--min-samples", "8", "--min-per-class", "4", "--min-kappa", "0", "--max-fpr", "1"]
        ) == 0
        assert compare_with_calibration(db_path) == 0
        assert main(base) == 4
        assert compare_with_calibration(db_path) == 1

    def test_rule_only_run_does_not_require_calibration(self, tmp_path, db_path, gate_agent):
        path = tmp_path / "rules.yaml"
        path.write_text(
            yaml.safe_dump(
                [{"id": "r1", "input": "q", "assertions": [{"output_contains": "hello"}]}]
            ),
            encoding="utf-8",
        )
        for run_id in ("A", "B"):
            assert main(
                [
                    "run",
                    "--cases",
                    str(path),
                    "--agent",
                    "decision_gate_agent:agent",
                    "--db",
                    db_path,
                    "--run-id",
                    run_id,
                ]
            ) == 0
        assert compare_with_calibration(db_path) == 0

    @pytest.mark.parametrize(
        "args",
        [
            ["compare", "A", "B", "--min-cases", "0"],
            ["compare", "A", "B", "--min-delta", "-0.1"],
            ["compare", "A", "B", "--alpha", "1"],
            ["calibrate", "--golden", "x", "--judge-backend", "stub", "--max-fpr", "2"],
        ],
    )
    def test_invalid_gate_parameters_are_rejected(self, args):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2
