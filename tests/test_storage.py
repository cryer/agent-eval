import sqlite3

import pytest

from agent_eval.storage import Storage


class TestSpans:
    def test_insert_and_get_by_trace_id(self, storage, fake_span):
        storage.insert_span(fake_span(trace_id=0xAA, span_id=1, name="root"))
        storage.insert_span(
            fake_span(
                trace_id=0xAA,
                span_id=2,
                parent_span_id=1,
                name="child",
                start_time=500,
                attributes={"tool.name": "lookup", "n": 3},
            )
        )
        storage.insert_span(fake_span(trace_id=0xBB, span_id=3, name="other trace"))

        spans = storage.get_spans(f"{0xAA:032x}")
        assert len(spans) == 2
        # ordered by start_time: child (500) before root (1000)
        assert [s["name"] for s in spans] == ["child", "root"]
        child = spans[0]
        assert child["attributes"] == {"tool.name": "lookup", "n": 3}
        assert child["parent_span_id"] == f"{1:016x}"
        assert spans[1]["parent_span_id"] is None
        assert child["status"] == "OK"

    def test_get_spans_unknown_trace_is_empty(self, storage):
        assert storage.get_spans("0" * 32) == []

    def test_insert_or_replace_same_span(self, storage, fake_span):
        storage.insert_span(fake_span(trace_id=1, span_id=1, name="v1"))
        storage.insert_span(fake_span(trace_id=1, span_id=1, name="v2"))
        spans = storage.get_spans(f"{1:032x}")
        assert len(spans) == 1
        assert spans[0]["name"] == "v2"


def _record(storage, run_id, case_id, passed=True, sample_idx=0, output="out"):
    storage.record_case_result(
        run_id=run_id,
        case_id=case_id,
        passed=passed,
        output=output,
        assertions=[{"name": "a", "passed": passed, "detail": ""}],
        trace_id=f"{0xF00:032x}",
        duration_ms=1.5,
        cost_usd=0.001,
        sample_idx=sample_idx,
    )


class TestCaseResults:
    def test_record_and_read_back(self, storage):
        _record(storage, "r1", "c1", passed=False, sample_idx=1, output="hello")
        rows = storage.get_run_results("r1")
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == "r1"
        assert row["case_id"] == "c1"
        assert row["sample_idx"] == 1
        assert row["passed"] == 0
        assert row["output"] == "hello"
        assert row["assertions"] == [{"name": "a", "passed": False, "detail": ""}]
        assert row["duration_ms"] == pytest.approx(1.5)
        assert row["cost_usd"] == pytest.approx(0.001)

    def test_results_ordered_by_case_and_sample(self, storage):
        _record(storage, "r1", "c2", sample_idx=0)
        _record(storage, "r1", "c1", sample_idx=1)
        _record(storage, "r1", "c1", sample_idx=0)
        rows = storage.get_run_results("r1")
        assert [(r["case_id"], r["sample_idx"]) for r in rows] == [
            ("c1", 0),
            ("c1", 1),
            ("c2", 0),
        ]

    def test_get_run_results_scoped_to_run(self, storage):
        _record(storage, "r1", "c1")
        _record(storage, "r2", "c1")
        assert len(storage.get_run_results("r1")) == 1
        assert len(storage.get_run_results("r2")) == 1

    def test_delete_only_listed_cases(self, storage):
        _record(storage, "r1", "c1")
        _record(storage, "r1", "c2")
        _record(storage, "r1", "c3")
        storage.delete_run_results("r1", case_ids=["c1", "c3"])
        remaining = storage.get_run_results("r1")
        assert [r["case_id"] for r in remaining] == ["c2"]

    def test_delete_does_not_touch_other_runs(self, storage):
        _record(storage, "r1", "c1")
        _record(storage, "r2", "c1")
        storage.delete_run_results("r1", case_ids=["c1"])
        assert storage.get_run_results("r1") == []
        assert len(storage.get_run_results("r2")) == 1

    def test_delete_without_case_ids_clears_whole_run(self, storage):
        _record(storage, "r1", "c1")
        _record(storage, "r1", "c2")
        storage.delete_run_results("r1")
        assert storage.get_run_results("r1") == []

    def test_idempotent_rerun_does_not_accumulate_rows(self, storage):
        # CI re-run semantics: same run-id + same cases -> same row count
        for _ in range(3):
            storage.delete_run_results("r1", case_ids=["c1", "c2"])
            for case_id in ("c1", "c2"):
                for sample_idx in (0, 1):
                    _record(storage, "r1", case_id, sample_idx=sample_idx)
        assert len(storage.get_run_results("r1")) == 4

    def test_mixed_sources_under_one_run_id(self, storage):
        # static cases + scenarios share a run-id; deleting only the
        # scenario's ids must keep the static results (and vice versa)
        _record(storage, "mix", "static_case")
        _record(storage, "mix", "scenario_1")
        storage.delete_run_results("mix", case_ids=["scenario_1"])
        _record(storage, "mix", "scenario_1", passed=False)
        rows = storage.get_run_results("mix")
        assert [(r["case_id"], r["passed"]) for r in rows] == [
            ("scenario_1", 0),
            ("static_case", 1),
        ]


class TestRuns:
    def test_record_and_get(self, storage):
        storage.record_run("r1", agent="m:a", cases_path="cases.yaml", total=4, passed=3)
        run = storage.get_run("r1")
        assert run["agent"] == "m:a"
        assert run["total"] == 4
        assert run["passed"] == 3
        assert run["created_at"]

    def test_get_missing_run_is_none(self, storage):
        assert storage.get_run("nope") is None

    def test_record_run_replaces_and_merges_totals(self, storage):
        # the CLI merge pattern: second command under the same run-id adds
        # its totals onto the existing row (INSERT OR REPLACE)
        storage.record_run("r1", agent="m:a", cases_path="a.yaml", total=2, passed=2)
        existing = storage.get_run("r1")
        storage.record_run(
            "r1",
            agent="m:a",
            cases_path=f"{existing['cases_path']} + b.yaml",
            total=existing["total"] + 1,
            passed=existing["passed"] + 0,
        )
        run = storage.get_run("r1")
        assert run["total"] == 3
        assert run["passed"] == 2
        assert run["cases_path"] == "a.yaml + b.yaml"

    def test_list_runs(self, storage):
        storage.record_run("r1", agent="m:a", cases_path="c", total=1, passed=1)
        storage.record_run("r2", agent="m:a", cases_path="c", total=2, passed=1)
        runs = storage.list_runs()
        assert {r["run_id"] for r in runs} == {"r1", "r2"}


class TestCalibrations:
    def test_record_and_list(self, storage):
        metrics = {"accuracy": 0.75, "kappa": 0.5}
        details = [{"id": "g1", "agree": True}]
        cal_id = storage.record_calibration("stub", "golden.yaml", metrics, details)
        assert isinstance(cal_id, int)
        rows = storage.list_calibrations()
        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == cal_id
        assert row["judge"] == "stub"
        assert row["golden_path"] == "golden.yaml"
        assert row["metrics"] == metrics
        assert row["details"] == details

    def test_ids_increase(self, storage):
        a = storage.record_calibration("stub", "g", {}, [])
        b = storage.record_calibration("stub", "g", {}, [])
        assert b > a


class TestMigration:
    def test_old_schema_without_sample_idx_is_migrated(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE case_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                case_id     TEXT NOT NULL,
                passed      INTEGER NOT NULL,
                output      TEXT NOT NULL,
                assertions  TEXT NOT NULL,
                trace_id    TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                cost_usd    REAL NOT NULL,
                created_at  TEXT NOT NULL
            );
            """
        )
        conn.close()

        storage = Storage(db_path)
        try:
            cols = {
                r["name"]
                for r in storage._conn.execute("PRAGMA table_info(case_results)")
            }
            assert "sample_idx" in cols
            _record(storage, "r1", "c1")  # works with default sample_idx=0
            assert storage.get_run_results("r1")[0]["sample_idx"] == 0
        finally:
            storage.close()
