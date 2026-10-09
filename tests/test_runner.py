import pytest
import yaml
from opentelemetry import trace

from agent_eval.assertions import MaxTurns, OutputContains, ToolCalled, build_assertions
from agent_eval.judge import Judge, StubBackend
from agent_eval.runner import Case, _normalize_agent_result, load_cases, run_cases


class TestNormalizeAgentResult:
    def test_plain_string(self):
        assert _normalize_agent_result("hello") == ("hello", 1)

    def test_dict_with_output_and_turns(self):
        assert _normalize_agent_result({"output": "hi", "turns": 3}) == ("hi", 3)

    def test_dict_defaults_turns_to_one(self):
        assert _normalize_agent_result({"output": "hi"}) == ("hi", 1)

    def test_dict_output_is_stringified(self):
        assert _normalize_agent_result({"output": 42}) == ("42", 1)

    @pytest.mark.parametrize("bad", [42, None, ["x"], {"no_output": 1}, ("x",)])
    def test_rejects_other_types(self, bad):
        with pytest.raises(TypeError, match="must return a string or a dict"):
            _normalize_agent_result(bad)


class TestLoadCases:
    def test_loads_rule_and_judge_assertions(self, tmp_path):
        p = tmp_path / "cases.yaml"
        p.write_text(
            yaml.safe_dump(
                [
                    {
                        "id": "c1",
                        "input": "hi",
                        "assertions": [
                            {"output_contains": "hello"},
                            {"judge": {"rubric": 'needs "hello"', "threshold": 0.5}},
                        ],
                    },
                    {"id": "c2", "input": "no assertions"},
                ]
            ),
            encoding="utf-8",
        )
        cases = load_cases(p, judge=Judge(StubBackend()))
        assert [c.id for c in cases] == ["c1", "c2"]
        assert len(cases[0].assertions) == 2
        assert cases[1].assertions == []

    def test_non_list_file_raises(self, tmp_path):
        p = tmp_path / "cases.yaml"
        p.write_text("not: a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a YAML list"):
            load_cases(p)


def _case(case_id, *assertions):
    return Case(id=case_id, input="ignored input", assertions=list(assertions))


class TestRunCases:
    def test_all_pass(self, storage, tracer):
        cases = [_case("c1", OutputContains("hello"))]
        summary = run_cases(lambda _: "hello world", cases, storage, tracer)
        assert summary.ok is True
        assert summary.passed == 1
        assert summary.total == 1
        assert summary.mean_pass_rate == pytest.approx(1.0)
        assert len(summary.run_id) == 8  # auto-generated hex id

    def test_explicit_run_id(self, storage, tracer):
        summary = run_cases(
            lambda _: "x", [_case("c1")], storage, tracer, run_id="my-run"
        )
        assert summary.run_id == "my-run"
        assert len(storage.get_run_results("my-run")) == 1

    def test_pass_k_semantics_one_bad_sample_fails_case(self, storage, tracer):
        calls = {"n": 0}

        def flaky_agent(_):
            calls["n"] += 1
            # second sample of every case fails
            return "good output" if calls["n"] % 2 == 1 else "bad"

        cases = [
            _case("flaky", OutputContains("good")),
            _case("stable", OutputContains("put")),
        ]
        summary = run_cases(flaky_agent, cases, storage, tracer, samples=2)

        flaky = next(r for r in summary.reports if r.case.id == "flaky")
        stable = next(r for r in summary.reports if r.case.id == "stable")
        # agent call order: flaky s0 (good), flaky s1 (bad), stable s0 (good), stable s1 (bad)...
        # -> flaky pass_rate 0.5, stable pass_rate 0.5
        assert flaky.passed is False
        assert flaky.pass_rate == pytest.approx(0.5)
        assert stable.pass_rate == pytest.approx(0.5)
        assert summary.ok is False
        assert summary.passed == 0
        assert summary.mean_pass_rate == pytest.approx(0.5)

    def test_samples_stored_per_sample(self, storage, tracer):
        cases = [_case("c1", OutputContains("x"))]
        run_cases(lambda _: "x", cases, storage, tracer, run_id="r1", samples=3)
        rows = storage.get_run_results("r1")
        assert len(rows) == 3
        assert [r["sample_idx"] for r in rows] == [0, 1, 2]
        assert all(r["passed"] == 1 for r in rows)
        # each sample has its own trace with spans exported
        assert len({r["trace_id"] for r in rows}) == 3
        for r in rows:
            assert storage.get_spans(r["trace_id"])

    def test_failed_assertion_detail_stored(self, storage, tracer):
        cases = [_case("c1", OutputContains("missing"))]
        run_cases(lambda _: "actual", cases, storage, tracer, run_id="r1")
        row = storage.get_run_results("r1")[0]
        assert row["passed"] == 0
        assert row["assertions"][0]["passed"] is False
        assert "substring not found" in row["assertions"][0]["detail"]

    def test_dict_result_turns_feed_max_turns(self, storage, tracer):
        agent = lambda _: {"output": "x", "turns": 5}
        ok = run_cases(agent, [_case("c1", MaxTurns(5))], storage, tracer)
        assert ok.ok is True
        bad = run_cases(agent, [_case("c2", MaxTurns(4))], storage, tracer)
        assert bad.ok is False

    def test_agent_spans_reach_tool_called_assertion(self, storage, tracer):
        # agent emits a tool span through the global provider; it must land in
        # the same trace so ToolCalled can see it
        agent_tracer = trace.get_tracer("test-agent")

        def agent_with_tool(_):
            with agent_tracer.start_as_current_span("tool.lookup") as s:
                s.set_attribute("tool.name", "lookup_order")
            return "done"

        cases = [_case("c1", ToolCalled("lookup_order"))]
        summary = run_cases(agent_with_tool, cases, storage, tracer)
        assert summary.ok is True

        missing = run_cases(
            agent_with_tool, [_case("c2", ToolCalled("refund"))], storage, tracer
        )
        assert missing.ok is False

    def test_samples_below_one_raises(self, storage, tracer):
        with pytest.raises(ValueError, match="samples must be >= 1"):
            run_cases(lambda _: "x", [_case("c1")], storage, tracer, samples=0)

    def test_summary_counts_with_mixed_outcomes(self, storage, tracer):
        cases = [
            _case("ok1", OutputContains("yes")),
            _case("ok2", OutputContains("yes")),
            _case("bad", OutputContains("no")),
        ]
        summary = run_cases(lambda _: "yes indeed", cases, storage, tracer)
        assert summary.total == 3
        assert summary.passed == 2
        assert summary.ok is False
        assert summary.mean_pass_rate == pytest.approx(2 / 3)
