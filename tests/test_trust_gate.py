import json
import sys
import types
import xml.etree.ElementTree as ET

import pytest
import yaml

from agent_eval.cli import main
from agent_eval.storage import Storage


@pytest.fixture
def trust_agents(monkeypatch):
    mod = types.ModuleType("trust_gate_agents")

    def good(text):
        return "hello world"

    def bad(text):
        return "wrong"

    def exploding(text):
        raise RuntimeError("agent exploded")

    def chat(messages):
        return "safe reply"

    mod.good = good
    mod.bad = bad
    mod.exploding = exploding
    mod.chat = chat
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod


def write_cases(tmp_path, n=2, name="cases.yaml"):
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "id": f"case{i}",
                    "input": f"question {i}",
                    "assertions": [{"output_contains": "hello"}],
                }
                for i in range(n)
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def run_cases(db, cases, run_id, agent="good", samples=1, extra=()):
    return main(
        [
            "run",
            "--cases",
            cases,
            "--agent",
            f"trust_gate_agents:{agent}",
            "--db",
            db,
            "--run-id",
            run_id,
            "--samples",
            str(samples),
            *extra,
        ]
    )


def test_agent_error_returns_three_and_writes_error_reports(
    tmp_path, db_path, trust_agents, capsys
):
    cases = write_cases(tmp_path)
    json_out = tmp_path / "reports" / "run.json"
    junit_out = tmp_path / "reports" / "run.xml"
    rc = run_cases(
        db_path,
        cases,
        "broken",
        agent="exploding",
        extra=("--json-out", str(json_out), "--junit-out", str(junit_out)),
    )
    assert rc == 3
    assert "ERROR" in capsys.readouterr().out
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["execution_status"] == "error"
    assert report["summary"]["error_samples"] == 2
    assert report["cases"][0]["samples"][0]["error"] == "RuntimeError: agent exploded"
    suite = ET.parse(junit_out).getroot()
    assert suite.attrib["errors"] == "2"
    assert suite.attrib["failures"] == "0"


def test_assertion_failure_is_not_an_execution_error(tmp_path, db_path, trust_agents):
    cases = write_cases(tmp_path, n=1)
    junit_out = tmp_path / "failure.xml"
    assert run_cases(
        db_path, cases, "failed", agent="bad", extra=("--junit-out", str(junit_out))
    ) == 1
    suite = ET.parse(junit_out).getroot()
    assert suite.attrib["failures"] == "1"
    assert suite.attrib["errors"] == "0"
    storage = Storage(db_path)
    try:
        row = storage.get_run_results("failed")[0]
        assert row["status"] == "completed"
        execution = storage.get_run_executions("failed")[0]
        assert execution["status"] == "completed"
    finally:
        storage.close()


def test_compare_rejects_different_sample_counts(tmp_path, db_path, trust_agents, capsys):
    cases = write_cases(tmp_path)
    assert run_cases(db_path, cases, "A") == 0
    assert run_cases(db_path, cases, "B", samples=2) == 0
    assert main(["compare", "A", "B", "--db", db_path]) == 1
    assert "samples differs" in capsys.readouterr().out


def test_compare_rejects_changed_dataset(tmp_path, db_path, trust_agents, capsys):
    cases = write_cases(tmp_path)
    assert run_cases(db_path, cases, "A") == 0
    data = yaml.safe_load(open(cases, encoding="utf-8"))
    data[0]["input"] = "changed input"
    with open(cases, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    assert run_cases(db_path, cases, "B") == 0
    assert main(["compare", "A", "B", "--db", db_path]) == 1
    assert "source_hash differs" in capsys.readouterr().out


def test_compare_rejects_different_judge_config(tmp_path, db_path, trust_agents, capsys):
    cases = write_cases(tmp_path)
    assert run_cases(db_path, cases, "A") == 0
    assert run_cases(db_path, cases, "B", extra=("--judge-backend", "stub")) == 0
    assert main(["compare", "A", "B", "--db", db_path]) == 1
    assert "judge differs" in capsys.readouterr().out


def test_compare_rejects_execution_errors(tmp_path, db_path, trust_agents, capsys):
    cases = write_cases(tmp_path)
    assert run_cases(db_path, cases, "A") == 0
    assert run_cases(db_path, cases, "B", agent="exploding") == 3
    assert main(["compare", "A", "B", "--db", db_path]) == 1
    out = capsys.readouterr().out
    assert "execution status is error" in out
    assert "no statistical comparison" in out


def test_compare_rejects_legacy_runs_without_manifest(db_path, capsys):
    storage = Storage(db_path)
    try:
        for run_id in ("A", "B"):
            storage.record_case_result(
                run_id, "c1", True, "ok", [], "0" * 32, 1.0, 0.0
            )
    finally:
        storage.close()
    assert main(["compare", "A", "B", "--db", db_path]) == 1
    assert "no execution manifest" in capsys.readouterr().out


def test_compare_rejects_missing_scenario_source(tmp_path, db_path, trust_agents, capsys):
    cases = write_cases(tmp_path)
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "scenario1",
                    "persona": {"style": "polite", "hidden_goal": "help", "script": ["hi"]},
                    "assertions": [{"output_contains": "safe reply"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert run_cases(db_path, cases, "A") == 0
    assert main(
        [
            "simulate",
            "--scenarios",
            str(scenarios),
            "--agent",
            "trust_gate_agents:chat",
            "--db",
            db_path,
            "--run-id",
            "A",
        ]
    ) == 0
    assert run_cases(db_path, cases, "B") == 0
    assert main(["compare", "A", "B", "--db", db_path]) == 1
    assert "execution sources differ" in capsys.readouterr().out


def test_valid_compare_writes_json(tmp_path, db_path, trust_agents):
    cases = write_cases(tmp_path)
    assert run_cases(db_path, cases, "A") == 0
    assert run_cases(db_path, cases, "B") == 0
    output = tmp_path / "reports" / "compare.json"
    assert main([
        "compare", "A", "B", "--db", db_path, "--min-cases", "2",
        "--json-out", str(output),
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["verdict"] == "no_meaningful_regression"
    assert report["metrics"]["common_cases"] == 2
    assert len(report["executions"]["A"]) == 1


def test_simulation_agent_error_isolated_and_returned_as_three(
    tmp_path, db_path, trust_agents
):
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            [
                {
                    "id": f"scenario{i}",
                    "persona": {"style": "polite", "hidden_goal": "help", "script": ["hi"]},
                    "assertions": [],
                }
                for i in range(2)
            ]
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "simulate",
            "--scenarios",
            str(scenarios),
            "--agent",
            "trust_gate_agents:exploding",
            "--db",
            db_path,
            "--run-id",
            "sim-error",
            "--concurrency",
            "2",
        ]
    )
    assert rc == 3
    storage = Storage(db_path)
    try:
        rows = storage.get_run_results("sim-error")
        assert len(rows) == 2
        assert all(row["status"] == "error" for row in rows)
        assert storage.get_run_executions("sim-error")[0]["status"] == "error"
    finally:
        storage.close()


def test_run_records_observed_agent_model_and_prompt_version(tmp_path, db_path):
    output = tmp_path / "observed.json"
    rc = main(
        [
            "run",
            "--cases",
            "examples/cases.yaml",
            "--agent",
            "examples.toy_agent:agent",
            "--judge-backend",
            "stub",
            "--db",
            db_path,
            "--run-id",
            "observed",
            "--json-out",
            str(output),
        ]
    )
    assert rc == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["manifest"]["agent_models"] == ["gpt-4o-mini"]
    assert len(report["manifest"]["agent_prompt_versions"]) == 1
    storage = Storage(db_path)
    try:
        execution = storage.get_run_executions("observed")[0]
        assert execution["metadata"]["agent_models"] == ["gpt-4o-mini"]
    finally:
        storage.close()
