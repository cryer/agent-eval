from __future__ import annotations

import inspect
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .assertions import Assertion, AssertionResult, CaseContext, FileDiffMaxFiles, build_assertions
from .storage import Storage


@dataclass
class Case:
    id: str
    input: str
    assertions: list[Assertion]
    # Fixture directory copied to a fresh scratch dir per sample (coding-agent cases).
    env: str | None = None


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
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"cases file must contain a YAML list, got {type(raw).__name__}")
    base = path.resolve().parent
    cases = []
    for entry in raw:
        env = entry.get("env")
        cases.append(
            Case(
                id=entry["id"],
                input=entry["input"],
                assertions=build_assertions(entry.get("assertions", []), judge=judge),
                env=str((base / env).resolve()) if env else None,
            )
        )
    return cases


def _accepts_env(agent_fn: Callable) -> bool:
    """Coding agents declare `fn(task, env_dir)`; text agents declare `fn(task)`."""
    try:
        params = list(inspect.signature(agent_fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _prepare_env(case: Case) -> str | None:
    """Copy the case fixture into a fresh scratch dir; git-init it when a
    file_diff assertion needs a baseline. Caller must clean up the result."""
    if case.env is None:
        return None
    env_dir = tempfile.mkdtemp(prefix=f"agent_eval_{case.id}_")
    shutil.rmtree(env_dir)
    shutil.copytree(case.env, env_dir)
    if any(isinstance(a, FileDiffMaxFiles) for a in case.assertions):
        git = ["git", "-c", "user.email=agent-eval@local", "-c", "user.name=agent-eval"]
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "fixture baseline"]):
            subprocess.run(
                git + args, cwd=env_dir, capture_output=True
            )  # best-effort; the assertion reports a missing repo
        # Python bytecode caches are a side effect of running checks, not an edit
        exclude = Path(env_dir) / ".git" / "info" / "exclude"
        if exclude.exists():
            exclude.write_text(exclude.read_text() + "\n__pycache__/\n", encoding="utf-8")
    return env_dir


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
    agent_fn: Callable,
    case: Case,
    storage: Storage,
    tracer,
    run_id: str,
    sample_idx: int,
    accepts_env: bool = False,
) -> SampleReport:
    env_dir = _prepare_env(case)
    try:
        t0 = time.perf_counter()
        with tracer.start_as_current_span(f"case.{case.id}") as span:
            span.set_attribute("case.id", case.id)
            span.set_attribute("case.sample_idx", sample_idx)
            result = agent_fn(case.input, env_dir) if accepts_env else agent_fn(case.input)
            output, turns = _normalize_agent_result(result)
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
            env_dir=env_dir,
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
    finally:
        if env_dir is not None:
            shutil.rmtree(env_dir, ignore_errors=True)


def run_cases(
    agent_fn: Callable,
    cases: list[Case],
    storage: Storage,
    tracer,
    run_id: str | None = None,
    samples: int = 1,
    concurrency: int = 1,
) -> RunSummary:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    accepts_env = _accepts_env(agent_fn)
    if not accepts_env and any(c.env for c in cases):
        raise TypeError(
            "cases declare an env fixture but the agent callable takes only one argument; "
            "coding agents must accept (task: str, env_dir: str)"
        )
    run_id = run_id or uuid.uuid4().hex[:8]
    reports = [CaseReport(case=c) for c in cases]

    tasks = [(case_idx, sample_idx) for case_idx in range(len(cases)) for sample_idx in range(samples)]
    if concurrency == 1:
        for case_idx, sample_idx in tasks:
            reports[case_idx].samples.append(
                _run_sample(
                    agent_fn, cases[case_idx], storage, tracer, run_id, sample_idx, accepts_env
                )
            )
    else:
        # IO-bound workload (LLM calls) -> threads are enough. Storage and the
        # OTel exporter are lock-guarded; span context is per-thread, so each
        # worker gets its own trace root.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _run_sample,
                    agent_fn,
                    cases[case_idx],
                    storage,
                    tracer,
                    run_id,
                    sample_idx,
                    accepts_env,
                ): (case_idx, sample_idx)
                for case_idx, sample_idx in tasks
            }
            for fut in as_completed(futures):
                case_idx, _ = futures[fut]
                reports[case_idx].samples.append(fut.result())
        # as_completed order is nondeterministic; reports must follow case/sample order
        for report in reports:
            report.samples.sort(key=lambda s: s.sample_idx)

    return RunSummary(run_id=run_id, reports=reports, samples_per_case=samples)
