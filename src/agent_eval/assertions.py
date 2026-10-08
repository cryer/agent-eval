from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CaseContext:
    """Everything an assertion is allowed to look at."""

    case_id: str
    input: str
    output: str
    turns: int
    spans: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_ms: float = 0.0


@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""


class Assertion(Protocol):
    name: str

    def check(self, ctx: CaseContext) -> AssertionResult: ...


class OutputContains:
    def __init__(self, text: str):
        self.text = text
        self.name = f"output_contains({text!r})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        passed = self.text in ctx.output
        return AssertionResult(self.name, passed, "" if passed else "substring not found in output")


class OutputNotContains:
    def __init__(self, text: str):
        self.text = text
        self.name = f"output_not_contains({text!r})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        passed = self.text not in ctx.output
        return AssertionResult(self.name, passed, "" if passed else "forbidden substring found in output")


class OutputMatches:
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.name = f"output_matches({pattern!r})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        passed = re.search(self.pattern, ctx.output) is not None
        return AssertionResult(self.name, passed, "" if passed else "regex did not match output")


class ToolCalled:
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.name = f"tool_called({tool_name!r})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        called = [s for s in ctx.spans if s["attributes"].get("tool.name") == self.tool_name]
        return AssertionResult(
            self.name,
            bool(called),
            "" if called else f"no span with tool.name={self.tool_name!r} in trace",
        )


class MaxTurns:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.name = f"max_turns({self.limit})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        passed = ctx.turns <= self.limit
        return AssertionResult(self.name, passed, "" if passed else f"{ctx.turns} turns > {self.limit}")


class MaxCostUsd:
    def __init__(self, limit: float):
        self.limit = float(limit)
        self.name = f"max_cost_usd({self.limit})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        passed = ctx.cost_usd <= self.limit
        return AssertionResult(
            self.name, passed, "" if passed else f"${ctx.cost_usd:.6f} > ${self.limit:.6f}"
        )


class MaxLatencyMs:
    def __init__(self, limit: float):
        self.limit = float(limit)
        self.name = f"max_latency_ms({self.limit})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        passed = ctx.duration_ms <= self.limit
        return AssertionResult(
            self.name, passed, "" if passed else f"{ctx.duration_ms:.1f}ms > {self.limit:.1f}ms"
        )


_REGISTRY = {
    "output_contains": OutputContains,
    "output_not_contains": OutputNotContains,
    "output_matches": OutputMatches,
    "tool_called": ToolCalled,
    "max_turns": MaxTurns,
    "max_cost_usd": MaxCostUsd,
    "max_latency_ms": MaxLatencyMs,
}


def build_assertions(spec_list: list[dict[str, Any]]) -> list[Assertion]:
    """Build assertion objects from a YAML/JSON spec like `- tool_called: lookup_order`."""
    assertions: list[Assertion] = []
    for spec in spec_list or []:
        if not isinstance(spec, dict) or len(spec) != 1:
            raise ValueError(f"assertion spec must be a single-key mapping, got: {spec!r}")
        (kind, value), = spec.items()
        cls = _REGISTRY.get(kind)
        if cls is None:
            raise ValueError(f"unknown assertion kind {kind!r}; available: {sorted(_REGISTRY)}")
        assertions.append(cls(value))
    return assertions
