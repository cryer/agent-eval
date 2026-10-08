from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from .storage import Storage


def tool_sequence(spans: list[dict]) -> list[str]:
    """Ordered tool call sequence of a trace — the 'how', not just the 'what'."""
    return [s["attributes"]["tool.name"] for s in spans if "tool.name" in s["attributes"]]


@dataclass
class CaseDiff:
    case_id: str
    pass_rate_a: float
    pass_rate_b: float
    tools_a: list[str] = field(default_factory=list)
    tools_b: list[str] = field(default_factory=list)
    tool_diff: list[str] = field(default_factory=list)
    output_diff: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.tool_diff or self.output_diff or self.pass_rate_a != self.pass_rate_b)


def _first_sample(rows: list[dict]) -> dict[str, dict]:
    """case_id -> first sample's result row."""
    out: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: (r["case_id"], r["sample_idx"])):
        out.setdefault(r["case_id"], r)
    return out


def _pass_rates(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for r in rows:
        grouped.setdefault(r["case_id"], []).append(bool(r["passed"]))
    return {cid: sum(v) / len(v) for cid, v in grouped.items()}


def diff_runs(storage: Storage, run_a: str, run_b: str) -> list[CaseDiff]:
    """Per-case trajectory diff between two runs (first sample of each case)."""
    rows_a = storage.get_run_results(run_a)
    rows_b = storage.get_run_results(run_b)
    first_a = _first_sample(rows_a)
    first_b = _first_sample(rows_b)
    rates_a = _pass_rates(rows_a)
    rates_b = _pass_rates(rows_b)

    diffs: list[CaseDiff] = []
    for case_id in sorted(set(first_a) & set(first_b)):
        ra, rb = first_a[case_id], first_b[case_id]
        spans_a = storage.get_spans(ra["trace_id"])
        spans_b = storage.get_spans(rb["trace_id"])
        tools_a = tool_sequence(spans_a)
        tools_b = tool_sequence(spans_b)

        tool_diff = []
        if tools_a != tools_b:
            tool_diff = list(
                difflib.unified_diff(tools_a, tools_b, fromfile=run_a, tofile=run_b, lineterm="")
            )
        output_diff = []
        if ra["output"] != rb["output"]:
            output_diff = list(
                difflib.unified_diff(
                    ra["output"].splitlines(),
                    rb["output"].splitlines(),
                    fromfile=run_a,
                    tofile=run_b,
                    lineterm="",
                )
            )
        diffs.append(
            CaseDiff(
                case_id=case_id,
                pass_rate_a=rates_a[case_id],
                pass_rate_b=rates_b[case_id],
                tools_a=tools_a,
                tools_b=tools_b,
                tool_diff=tool_diff,
                output_diff=output_diff,
            )
        )
    return diffs
