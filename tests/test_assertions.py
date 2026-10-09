import pytest

from agent_eval.assertions import (
    CaseContext,
    MaxCostUsd,
    MaxLatencyMs,
    MaxTurns,
    OutputContains,
    OutputMatches,
    OutputNotContains,
    ToolCalled,
    build_assertions,
)
from agent_eval.judge import Judge, JudgeAssertion, StubBackend


def ctx(**overrides) -> CaseContext:
    base = dict(
        case_id="c1",
        input="in",
        output="the quick brown fox 123",
        turns=1,
        spans=[],
        cost_usd=0.0,
        duration_ms=0.0,
    )
    base.update(overrides)
    return CaseContext(**base)


class TestOutputContains:
    def test_pass(self):
        assert OutputContains("brown").check(ctx()).passed is True

    def test_fail(self):
        r = OutputContains("purple").check(ctx())
        assert r.passed is False
        assert "substring not found" in r.detail


class TestOutputNotContains:
    def test_pass(self):
        assert OutputNotContains("purple").check(ctx()).passed is True

    def test_fail(self):
        r = OutputNotContains("fox").check(ctx())
        assert r.passed is False
        assert "forbidden substring" in r.detail


class TestOutputMatches:
    def test_pass(self):
        assert OutputMatches(r"\d+").check(ctx()).passed is True

    def test_fail(self):
        r = OutputMatches(r"^hello$").check(ctx())
        assert r.passed is False
        assert "regex did not match" in r.detail


class TestToolCalled:
    def _spans(self, *tool_names):
        return [{"attributes": {"tool.name": t}} for t in tool_names]

    def test_pass(self):
        c = ctx(spans=self._spans("lookup_order", "refund"))
        assert ToolCalled("refund").check(c).passed is True

    def test_fail_when_tool_absent(self):
        c = ctx(spans=self._spans("lookup_order"))
        r = ToolCalled("refund").check(c)
        assert r.passed is False
        assert "no span with tool.name='refund'" in r.detail

    def test_fail_on_empty_spans(self):
        assert ToolCalled("refund").check(ctx(spans=[])).passed is False

    def test_spans_without_tool_attributes_ignored(self):
        c = ctx(spans=[{"attributes": {"other": "x"}}])
        assert ToolCalled("refund").check(c).passed is False


class TestMaxTurns:
    def test_pass_at_limit(self):
        assert MaxTurns(3).check(ctx(turns=3)).passed is True

    def test_fail_over_limit(self):
        r = MaxTurns(3).check(ctx(turns=4))
        assert r.passed is False
        assert "4 turns > 3" in r.detail


class TestMaxCostUsd:
    def test_pass(self):
        assert MaxCostUsd(0.01).check(ctx(cost_usd=0.005)).passed is True

    def test_fail(self):
        r = MaxCostUsd(0.01).check(ctx(cost_usd=0.02))
        assert r.passed is False
        assert ">" in r.detail


class TestMaxLatencyMs:
    def test_pass(self):
        assert MaxLatencyMs(100).check(ctx(duration_ms=50.0)).passed is True

    def test_fail(self):
        r = MaxLatencyMs(100).check(ctx(duration_ms=150.0))
        assert r.passed is False
        assert "150.0ms > 100.0ms" in r.detail


class TestBuildAssertions:
    def test_builds_all_rule_kinds(self):
        specs = [
            {"output_contains": "x"},
            {"output_not_contains": "y"},
            {"output_matches": ".*"},
            {"tool_called": "t"},
            {"max_turns": 2},
            {"max_cost_usd": 0.1},
            {"max_latency_ms": 10},
        ]
        built = build_assertions(specs)
        assert [type(a) for a in built] == [
            OutputContains,
            OutputNotContains,
            OutputMatches,
            ToolCalled,
            MaxTurns,
            MaxCostUsd,
            MaxLatencyMs,
        ]

    def test_none_spec_list_is_empty(self):
        assert build_assertions(None) == []

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown assertion kind"):
            build_assertions([{"wat": 1}])

    def test_multi_key_mapping_raises(self):
        with pytest.raises(ValueError, match="single-key mapping"):
            build_assertions([{"output_contains": "x", "max_turns": 2}])

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="single-key mapping"):
            build_assertions(["output_contains"])

    def test_judge_assertion_without_judge_raises(self):
        with pytest.raises(ValueError, match="no judge backend"):
            build_assertions([{"judge": {"rubric": "r"}}], judge=None)

    def test_judge_assertion_with_judge(self):
        judge = Judge(StubBackend())
        built = build_assertions([{"judge": {"rubric": "r", "threshold": 0.5}}], judge=judge)
        assert len(built) == 1
        assert isinstance(built[0], JudgeAssertion)
