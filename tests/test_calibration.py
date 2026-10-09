from pathlib import Path

import pytest
import yaml

from agent_eval.calibration import (
    load_golden,
    run_calibration,
    store_calibration,
)
from agent_eval.judge import Judge, StubBackend

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "examples" / "golden.yaml"


class TestLoadGolden:
    def test_loads_examples_golden_set(self):
        items = load_golden(GOLDEN_PATH)
        assert len(items) == 8
        by_id = {i.id: i for i in items}
        assert by_id["g1_good_refund"].label is True
        assert by_id["g2_no_policy"].label is False
        assert by_id["g2_no_policy"].note  # note is optional but parsed
        assert by_id["g1_good_refund"].note == ""

    def test_invalid_label_raises(self, tmp_path):
        p = tmp_path / "g.yaml"
        p.write_text(
            yaml.safe_dump([{"id": "x", "output": "o", "rubric": "r", "label": "maybe"}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="label must be 'pass' or 'fail'"):
            load_golden(p)

    def test_non_list_raises(self, tmp_path):
        p = tmp_path / "g.yaml"
        p.write_text("not: a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML list"):
            load_golden(p)


class TestRunCalibration:
    def test_stub_on_examples_golden_set(self):
        items = load_golden(GOLDEN_PATH)
        report = run_calibration(items, Judge(StubBackend()))
        m = report.metrics
        # stub judges by keyword presence: it is fooled by g3 (keyword present
        # but wrong policy -> false positive) and g4 (paraphrase -> false
        # negative). Hand-computed: tp=3 tn=3 fp=1 fn=1.
        assert m["n"] == 8
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["kappa"] == pytest.approx(0.5)
        assert m["fpr"] == pytest.approx(0.25)
        assert m["fnr"] == pytest.approx(0.25)
        assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 3, 1, 1)
        for key in ("accuracy", "kappa", "tpr", "fpr", "fnr"):
            assert 0.0 <= m[key] <= 1.0

        disagreed = {r.item.id for r in report.items if not r.agree}
        assert disagreed == {"g3_evasive_keyword", "g4_paraphrase"}

    def test_perfect_judge_scores_one(self):
        items = load_golden(GOLDEN_PATH)
        # keep only the items the keyword stub gets right
        easy = [i for i in items if i.id not in {"g3_evasive_keyword", "g4_paraphrase"}]
        report = run_calibration(easy, Judge(StubBackend()))
        assert report.metrics["accuracy"] == pytest.approx(1.0)
        assert report.metrics["kappa"] == pytest.approx(1.0)
        assert all(r.agree for r in report.items)


class TestStoreCalibration:
    def test_results_landed_in_db(self, storage):
        items = load_golden(GOLDEN_PATH)
        report = run_calibration(items, Judge(StubBackend()))
        cal_id = store_calibration(storage, report, "stub", str(GOLDEN_PATH))
        rows = storage.list_calibrations()
        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == cal_id
        assert row["judge"] == "stub"
        assert row["metrics"]["kappa"] == pytest.approx(0.5)
        assert len(row["details"]) == 8
        g3 = next(d for d in row["details"] if d["id"] == "g3_evasive_keyword")
        assert g3["agree"] is False
        assert g3["label"] is False
        assert g3["judge_pass"] is True
