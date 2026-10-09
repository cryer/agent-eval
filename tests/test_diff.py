from agent_eval.diff import diff_runs, tool_sequence


def _record(storage, run_id, case_id, passed, output, trace_id, sample_idx=0):
    storage.record_case_result(
        run_id=run_id,
        case_id=case_id,
        passed=passed,
        output=output,
        assertions=[],
        trace_id=trace_id,
        duration_ms=1.0,
        cost_usd=0.0,
        sample_idx=sample_idx,
    )


def _trace_with_tools(storage, fake_span, trace_int, tools):
    for i, tool in enumerate(tools, start=1):
        storage.insert_span(
            fake_span(
                trace_id=trace_int,
                span_id=i,
                name=f"tool.{tool}",
                start_time=i,
                attributes={"tool.name": tool},
            )
        )


class TestToolSequence:
    def test_extracts_only_tool_spans_in_order(self):
        spans = [
            {"attributes": {"tool.name": "a"}},
            {"attributes": {"other": 1}},
            {"attributes": {"tool.name": "b"}},
        ]
        assert tool_sequence(spans) == ["a", "b"]

    def test_empty(self):
        assert tool_sequence([]) == []


class TestDiffRuns:
    def _setup(self, storage, fake_span):
        _trace_with_tools(storage, fake_span, 0xA1, ["lookup", "reply"])
        _trace_with_tools(storage, fake_span, 0xB1, ["lookup", "reply"])
        _trace_with_tools(storage, fake_span, 0xB2, ["reply"])  # fewer tools
        _trace_with_tools(storage, fake_span, 0xA3, ["lookup"])
        _trace_with_tools(storage, fake_span, 0xB3, ["lookup"])

        tid = lambda t: f"{t:032x}"
        # c_same: identical trajectory
        _record(storage, "A", "c_same", True, "same output", tid(0xA1))
        _record(storage, "B", "c_same", True, "same output", tid(0xB1))
        # c_changed: different tools, different output, flipped pass
        _record(storage, "A", "c_changed", True, "old answer", tid(0xA1))
        _record(storage, "B", "c_changed", False, "new answer", tid(0xB2))
        # c_rate: same trajectory, pass rate changed via samples
        _record(storage, "A", "c_rate", True, "x", tid(0xA3), sample_idx=0)
        _record(storage, "A", "c_rate", True, "x", tid(0xA3), sample_idx=1)
        _record(storage, "B", "c_rate", True, "x", tid(0xB3), sample_idx=0)
        _record(storage, "B", "c_rate", False, "x", tid(0xB3), sample_idx=1)

    def test_per_case_diff(self, storage, fake_span):
        self._setup(storage, fake_span)
        diffs = {d.case_id: d for d in diff_runs(storage, "A", "B")}
        assert set(diffs) == {"c_same", "c_changed", "c_rate"}

        same = diffs["c_same"]
        assert same.changed is False
        assert same.tool_diff == []
        assert same.output_diff == []

        changed = diffs["c_changed"]
        assert changed.changed is True
        assert changed.tools_a == ["lookup", "reply"]
        assert changed.tools_b == ["reply"]
        assert any(line.startswith("-lookup") for line in changed.tool_diff)
        assert any("-old answer" in line for line in changed.output_diff)
        assert any("+new answer" in line for line in changed.output_diff)
        assert changed.pass_rate_a == 1.0
        assert changed.pass_rate_b == 0.0

        rate = diffs["c_rate"]
        assert rate.changed is True  # pass rate flipped even with same trajectory
        assert rate.pass_rate_a == 1.0
        assert rate.pass_rate_b == 0.5
        assert rate.tool_diff == []
        assert rate.output_diff == []

    def test_no_common_cases_returns_empty(self, storage, fake_span):
        tid = f"{0xA1:032x}"
        _record(storage, "A", "only_a", True, "x", tid)
        assert diff_runs(storage, "A", "B") == []
