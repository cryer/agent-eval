"""Distill: rebuild regression cases from recorded traces."""

import pytest
import yaml

from agent_eval.cli import main
from agent_eval.distill import distill_cases, parse_since, render_cases_yaml
from agent_eval.runner import Case, load_cases, run_cases

T_OLD = parse_since("2024-01-01")
T_NEW = parse_since("2025-06-01")


def seed_trace(
    storage,
    fake_span,
    trace_id,
    case_id,
    input_text,
    output="some answer",
    tools=(),
    start_time=T_NEW,
    prompt_version=None,
    with_input_attr=True,
):
    """Insert one case trace: root case.<id> span + llm/tool child spans."""
    root_attrs = {"case.id": case_id, "case.sample_idx": 0, "case.output": output}
    if with_input_attr:
        root_attrs["case.input"] = input_text
    storage.insert_span(
        fake_span(
            trace_id=trace_id,
            span_id=0x1,
            name=f"case.{case_id}",
            start_time=start_time,
            end_time=start_time + 3000,
            attributes=root_attrs,
        )
    )
    llm_attrs = {}
    if prompt_version:
        llm_attrs["prompt.version"] = prompt_version
    storage.insert_span(
        fake_span(
            trace_id=trace_id,
            span_id=0x2,
            parent_span_id=0x1,
            name="llm.chat",
            start_time=start_time + 100,
            end_time=start_time + 2000,
            attributes=llm_attrs,
        )
    )
    for i, tool in enumerate(tools):
        storage.insert_span(
            fake_span(
                trace_id=trace_id,
                span_id=0x10 + i,
                parent_span_id=0x2,
                name=f"tool.{tool}",
                start_time=start_time + 200 + i,
                end_time=start_time + 300 + i,
                attributes={"tool.name": tool},
            )
        )


class TestParseSince:
    def test_date_only_is_utc_midnight(self):
        from datetime import datetime, timezone

        expect = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp() * 1e9)
        assert parse_since("2025-06-01") == expect

    def test_aware_datetime_kept(self):
        assert parse_since("2025-06-01T12:00:00+00:00") == parse_since("2025-06-01T12:00:00")


class TestDistillCases:
    def test_rebuilds_case_with_tool_assertions(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "refund", "能退款吗", tools=["lookup_order"])
        cases, scanned = distill_cases(storage)
        assert scanned == 1
        assert len(cases) == 1
        c = cases[0]
        assert c.id == "refund"
        assert c.input == "能退款吗"
        assert c.tools == ["lookup_order"]
        assert c.to_yaml_entry()["assertions"] == [{"tool_called": "lookup_order"}]

    def test_samples_of_same_case_dedupe(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "refund", "能退款吗", start_time=T_NEW)
        seed_trace(storage, fake_span, 0xA2, "refund", "能退款吗", start_time=T_NEW + 1)
        cases, scanned = distill_cases(storage)
        assert scanned == 2
        assert len(cases) == 1

    def test_same_case_id_distinct_inputs_get_suffix(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "refund", "能退款吗", start_time=T_NEW)
        seed_trace(storage, fake_span, 0xA2, "refund", "运费谁承担", start_time=T_NEW + 1)
        cases, _ = distill_cases(storage)
        assert {c.id for c in cases} == {"refund", "refund_v2"}

    def test_duplicate_tool_calls_dedupe_keep_order(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "multi", "q", tools=["a", "b", "a"])
        cases, _ = distill_cases(storage)
        assert cases[0].tools == ["a", "b"]

    def test_since_filter(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "old", "旧问题", start_time=T_OLD)
        seed_trace(storage, fake_span, 0xA2, "new", "新问题", start_time=T_NEW)
        cases, scanned = distill_cases(storage, since_ns=parse_since("2025-01-01"))
        assert scanned == 1
        assert [c.id for c in cases] == ["new"]

    def test_prompt_version_filter(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "v1case", "q1", prompt_version="aaa111")
        seed_trace(storage, fake_span, 0xA2, "v2case", "q2", prompt_version="bbb222")
        cases, _ = distill_cases(storage, prompt_version="bbb222")
        assert [c.id for c in cases] == ["v2case"]

    def test_limit(self, storage, fake_span):
        for i in range(5):
            seed_trace(storage, fake_span, 0xB0 + i, f"case{i}", f"q{i}")
        cases, scanned = distill_cases(storage, limit=2)
        assert len(cases) == 2
        assert scanned == 2

    def test_skips_scenario_roots_and_legacy_traces(self, storage, fake_span):
        storage.insert_span(
            fake_span(
                trace_id=0xC1,
                span_id=0x1,
                name="scenario.attack",
                start_time=T_NEW,
                end_time=T_NEW + 100,
                attributes={"scenario.id": "attack"},
            )
        )
        seed_trace(storage, fake_span, 0xC2, "legacy", "ignored", with_input_attr=False)
        cases, scanned = distill_cases(storage)
        assert cases == []
        assert scanned == 0  # legacy root is skipped before scanning

    def test_newest_first(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "older", "q1", start_time=T_OLD)
        seed_trace(storage, fake_span, 0xA2, "newer", "q2", start_time=T_NEW)
        cases, _ = distill_cases(storage)
        assert [c.id for c in cases] == ["newer", "older"]


class TestRender:
    def test_yaml_roundtrip_through_load_cases(self, storage, fake_span, tmp_path):
        seed_trace(storage, fake_span, 0xA1, "refund", "能退款吗", tools=["lookup_order"])
        seed_trace(storage, fake_span, 0xA2, "greeting", "你好")
        cases, _ = distill_cases(storage)
        out = tmp_path / "distilled.yaml"
        out.write_text(render_cases_yaml(cases), encoding="utf-8")

        loaded = load_cases(out)
        assert [c.id for c in loaded] == ["refund", "greeting"]
        assert loaded[0].input == "能退款吗"
        assert [a.name for a in loaded[0].assertions] == ["tool_called('lookup_order')"]
        assert loaded[1].assertions == []

        # the derived tool_called assertion actually passes against the source trace
        spans = storage.get_spans(f"{0xA1:032x}")
        from agent_eval.assertions import CaseContext

        ctx = CaseContext(case_id="refund", input="能退款吗", output="", turns=1, spans=spans)
        assert loaded[0].assertions[0].check(ctx).passed

    def test_header_comments_present(self, storage, fake_span):
        seed_trace(storage, fake_span, 0xA1, "refund", "能退款吗")
        cases, _ = distill_cases(storage)
        text = render_cases_yaml(cases)
        assert "agent-eval distill" in text
        assert f"# trace {0xA1:032x}" in text
        assert yaml.safe_load(text)  # comments must not break parsing


class TestCli:
    def test_writes_file_and_exits_zero(self, storage, fake_span, db_path, tmp_path, capsys):
        seed_trace(storage, fake_span, 0xA1, "refund", "能退款吗", tools=["lookup_order"])
        storage.close()
        out = tmp_path / "out.yaml"
        rc = main(["distill", "--db", db_path, "--out", str(out)])
        assert rc == 0
        assert [c["id"] for c in yaml.safe_load(out.read_text(encoding="utf-8"))] == ["refund"]
        assert "distilled 1 cases" in capsys.readouterr().out

    def test_empty_db_exits_one(self, db_path, tmp_path, capsys):
        out = tmp_path / "out.yaml"
        rc = main(["distill", "--db", db_path, "--out", str(out)])
        assert rc == 1
        assert not out.exists()
        assert "no distillable traces" in capsys.readouterr().out

    def test_since_and_prompt_version_flags(self, storage, fake_span, db_path, tmp_path):
        seed_trace(storage, fake_span, 0xA1, "old", "q1", start_time=T_OLD, prompt_version="aaa")
        seed_trace(storage, fake_span, 0xA2, "new", "q2", start_time=T_NEW, prompt_version="bbb")
        storage.close()
        out = tmp_path / "out.yaml"
        rc = main(
            [
                "distill", "--db", db_path, "--out", str(out),
                "--since", "2025-01-01", "--prompt-version", "bbb",
            ]
        )
        assert rc == 0
        assert [c["id"] for c in yaml.safe_load(out.read_text(encoding="utf-8"))] == ["new"]


class TestEndToEnd:
    def test_run_then_distill_recovers_case(self, storage, tracer, tmp_path):
        """The runner records case.input on the root span, so a fresh run is
        immediately distillable — the full observe -> distill loop."""

        def fake_agent(text: str) -> dict:
            with tracer.start_as_current_span("tool.lookup_order") as s:
                s.set_attribute("tool.name", "lookup_order")
            return {"output": "7 天内可退", "turns": 1}

        cases = [Case(id="refund", input="能退款吗", assertions=[])]
        run_cases(fake_agent, cases, storage, tracer, run_id="r1", samples=2)

        distilled, scanned = distill_cases(storage)
        assert scanned == 2  # two samples, one logical case
        assert len(distilled) == 1
        assert distilled[0].input == "能退款吗"
        assert distilled[0].tools == ["lookup_order"]
