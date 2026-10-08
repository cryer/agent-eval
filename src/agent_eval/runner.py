from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .assertions import Assertion, AssertionResult, CaseContext, build_assertions
from .storage import Storage


@dataclass
class Case:
    id: str
    input: str
    assertions: list[Assertion]


@dataclass
class SampleReport:
    sample_idx: int
    passed: bool
    output: str
    results: list[AssertionResult]
    trace_id: str
    duration_ms: float
    cost_usd: float


@dataclass
class CaseReport:
    case: Case
    samples: list[SampleReport] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.passed) / len(self.samples)

    @property
    def passed(self) -> bool:
        """pass^k: a case counts as passed only if every sample passed."""
        return bool(self.samples) and all(s.passed for s in self.samples)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.samples)

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.samples)


@dataclass
class RunSummary:
    run_id: str
    reports: list[CaseReport]
    samples_per_case: int = 1

    @property
    def total(self) -> int:
        return len(self.reports)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.reports if r.passed)

    @property
    def ok(self) -> bool:
        return self.passed == self.total

    @property
    def mean_pass_rate(self) -> float:
        if not self.reports:
            return 0.0
        return sum(r.pass_rate for r in self.reports) / len(self.reports)


def load_cases(path: str | Path, judge=None) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"cases file must contain a YAML list, got {type(raw).__name__}")
    cases = []
    for entry in raw:
        cases.append(
            Case(
                id=entry["id"],
                input=entry["input"],
                assertions=build_assertions(entry.get("assertions", []), judge=judge),
            )
        )
    return cases


def _normalize_agent_result(result: Any) -> tuple[str, int]:
    """Agent may return a plain string or a mapping with 'output' and optional 'turns'."""
    if isinstance(result, str):
        return result, 1
    if isinstance(result, dict) and "output" in result:
        return str(result["output"]), int(result.get("turns", 1))
    raise TypeError(
        "agent callable must return a string or a dict with an 'output' key, "
        f"got {type(result).__name__}"
    )


def _run_sample(
    agent_fn: Callable[[str], Any],
    case: Case,
    storage: Storage,
    tracer,
    run_id: str,
    sample_idx: int,
) -> SampleReport:
    t0 = time.perf_counter()
    with tracer.start_as_current_span(f"case.{case.id}") as span:
        span.set_attribute("case.id", case.id)
        span.set_attribute("case.sample_idx", sample_idx)
        output, turns = _normalize_agent_result(agent_fn(case.input))
        span.set_attribute("case.output", output)
        trace_id = f"{span.get_span_context().trace_id:032x}"
    duration_ms = (time.perf_counter() - t0) * 1000

    # root span ended -> all spans of this trace are exported to storage
    spans = storage.get_spans(trace_id)
    cost_usd = sum(float(s["attributes"].get("gen_ai.usage.cost_usd", 0.0)) for s in spans)

    ctx = CaseContext(
        case_id=case.id,
        input=case.input,
        output=output,
        turns=turns,
        spans=spans,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
    )
    results = [a.check(ctx) for a in case.assertions]
    passed = all(r.passed for r in results)

    storage.record_case_result(
        run_id=run_id,
        case_id=case.id,
        sample_idx=sample_idx,
        passed=passed,
        output=output,
        assertions=[{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
        trace_id=trace_id,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
    )
    return SampleReport(
        sample_idx=sample_idx,
        passed=passed,
        output=output,
        results=results,
        trace_id=trace_id,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
    )


def run_cases(
    agent_fn: Callable[[str], Any],
    cases: list[Case],
    storage: Storage,
    tracer,
    run_id: str | None = None,
    samples: int = 1,
) -> RunSummary:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    run_id = run_id or uuid.uuid4().hex[:8]
    reports: list[CaseReport] = []

    for case in cases:
        report = CaseReport(case=case)
        for sample_idx in range(samples):
            report.samples.append(
                _run_sample(agent_fn, case, storage, tracer, run_id, sample_idx)
            )
        reports.append(report)

    return RunSummary(run_id=run_id, reports=reports, samples_per_case=samples)
