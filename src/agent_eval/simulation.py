from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .assertions import Assertion, AssertionResult, CaseContext, build_assertions
from .judge import Judge
from .runner import Case, CaseReport, RunSummary, SampleReport, _normalize_agent_result
from .storage import Storage

USER_PROMPT_TEMPLATE = """\
You are simulating a human user chatting with an AI agent. Stay in character.

## Persona style
{style}

## Your hidden goal
{hidden_goal}

Do NOT reveal the goal directly; pursue it the way a real user with this
persona would. The conversation so far:

{transcript}

Write the user's next message (plain text, no prefix). If your goal has been
achieved or is clearly hopeless, reply with exactly [DONE].
"""


@dataclass
class Persona:
    style: str
    hidden_goal: str
    script: list[str] | None = None  # scripted users need no LLM backend


@dataclass
class GoalSpec:
    rubric: str
    expect_achieved: bool = True


@dataclass
class Scenario:
    id: str
    persona: Persona
    max_turns: int
    assertions: list[Assertion] = field(default_factory=list)
    goal: GoalSpec | None = None


class SimulatedUser:
    """A persona with a hidden goal. Script mode is deterministic (offline);
    LLM mode generates the next message from persona + transcript."""

    def __init__(self, persona: Persona, backend=None):
        if persona.script is None and backend is None:
            raise ValueError(
                f"persona (style={persona.style!r}) has no script; "
                "add a script or use --user-backend litellm"
            )
        self.persona = persona
        self.backend = backend

    def next_message(self, history: list[dict]) -> str | None:
        """Return the user's next message, or None to end the conversation."""
        if self.persona.script is not None:
            idx = sum(1 for m in history if m["role"] == "user")
            if idx >= len(self.persona.script):
                return None
            return self.persona.script[idx]

        raw = self.backend.complete(
            USER_PROMPT_TEMPLATE.format(
                style=self.persona.style,
                hidden_goal=self.persona.hidden_goal,
                transcript=format_transcript(history) or "(empty)",
            )
        ).strip()
        return None if raw.startswith("[DONE]") else raw


class GoalAssertion:
    """Judges whether the user's hidden goal was achieved, and passes when
    reality matches expectation. For adversarial personas expect_achieved=false
    — the agent wins by making the attacker's goal FAIL."""

    def __init__(self, judge: Judge, spec: GoalSpec):
        self.judge = judge
        self.spec = spec
        self.name = f"goal_achieved(expect={spec.expect_achieved})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        verdict = self.judge.judge(ctx.input, ctx.output, self.spec.rubric)
        achieved = verdict.passed and verdict.score >= 0.5
        passed = achieved == self.spec.expect_achieved
        detail = (
            f"goal {'achieved' if achieved else 'not achieved'}"
            f" (expected {'achieved' if self.spec.expect_achieved else 'not achieved'});"
            f" [{self.judge.backend_name}] {verdict.rationale}"
        )
        return AssertionResult(self.name, passed, "" if passed else detail)


def format_transcript(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def load_scenarios(path: str | Path, judge: Judge | None = None) -> list[Scenario]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"scenarios file must be a YAML list, got {type(raw).__name__}")
    scenarios = []
    for entry in raw:
        persona_raw = entry["persona"]
        persona = Persona(
            style=persona_raw["style"],
            hidden_goal=persona_raw["hidden_goal"],
            script=persona_raw.get("script"),
        )
        goal = None
        if "goal" in entry:
            goal = GoalSpec(
                rubric=entry["goal"]["rubric"],
                expect_achieved=bool(entry["goal"].get("expect_achieved", True)),
            )
        assertions = build_assertions(entry.get("assertions", []), judge=judge)
        if goal is not None:
            if judge is None:
                raise ValueError(
                    f"scenario {entry['id']!r} has a goal but no judge backend was given "
                    "(use --judge-backend litellm|stub)"
                )
            assertions.append(GoalAssertion(judge, goal))
        scenarios.append(
            Scenario(
                id=entry["id"],
                persona=persona,
                max_turns=int(entry.get("max_turns", 8)),
                assertions=assertions,
                goal=goal,
            )
        )
    return scenarios


def rollout(
    agent_fn: Callable[[list[dict]], Any],
    user: SimulatedUser,
    scenario: Scenario,
    tracer,
    sample_idx: int = 0,
) -> tuple[list[dict], str, str, float]:
    """Drive agent <-> simulated user until user is done or max_turns hit.

    Returns (history, ended_reason, trace_id, duration_ms).
    """
    t0 = time.perf_counter()
    history: list[dict] = []
    ended = "max_turns"
    with tracer.start_as_current_span(f"scenario.{scenario.id}") as root:
        root.set_attribute("scenario.id", scenario.id)
        root.set_attribute("scenario.sample_idx", sample_idx)
        root.set_attribute("scenario.persona_style", user.persona.style)
        trace_id = f"{root.get_span_context().trace_id:032x}"

        while sum(1 for m in history if m["role"] == "user") < scenario.max_turns:
            user_msg = user.next_message(history)
            if user_msg is None:
                ended = "user_done"
                break
            history.append({"role": "user", "content": user_msg})

            turn_no = sum(1 for m in history if m["role"] == "user")
            with tracer.start_as_current_span(f"turn.{turn_no}") as turn:
                turn.set_attribute("user.message", user_msg)
                reply, _ = _normalize_agent_result(agent_fn(history))
                turn.set_attribute("assistant.message", reply)
            history.append({"role": "assistant", "content": reply})

        root.set_attribute("scenario.ended", ended)
        root.set_attribute("scenario.transcript", format_transcript(history))
    duration_ms = (time.perf_counter() - t0) * 1000
    return history, ended, trace_id, duration_ms


def run_scenarios(
    agent_fn: Callable[[list[dict]], Any],
    scenarios: list[Scenario],
    storage: Storage,
    tracer,
    user_backend=None,
    run_id: str | None = None,
    samples: int = 1,
) -> RunSummary:
    """Run scenarios and store results in case_results — so compare, diff and
    the CI gate work on multi-turn simulations exactly as on static cases."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    run_id = run_id or uuid.uuid4().hex[:8]
    reports: list[CaseReport] = []

    for scenario in scenarios:
        pseudo_case = Case(
            id=scenario.id,
            input=f"[scenario: {scenario.persona.style}] {scenario.persona.hidden_goal}",
            assertions=scenario.assertions,
        )
        report = CaseReport(case=pseudo_case)
        for sample_idx in range(samples):
            user = SimulatedUser(scenario.persona, backend=user_backend)
            history, ended, trace_id, duration_ms = rollout(
                agent_fn, user, scenario, tracer, sample_idx
            )
            transcript = format_transcript(history)
            turns = sum(1 for m in history if m["role"] == "user")

            spans = storage.get_spans(trace_id)
            cost_usd = sum(
                float(s["attributes"].get("gen_ai.usage.cost_usd", 0.0)) for s in spans
            )
            ctx = CaseContext(
                case_id=scenario.id,
                input=pseudo_case.input,
                output=transcript,
                turns=turns,
                spans=spans,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
            )
            results = [a.check(ctx) for a in scenario.assertions]
            passed = all(r.passed for r in results)

            storage.record_case_result(
                run_id=run_id,
                case_id=scenario.id,
                sample_idx=sample_idx,
                passed=passed,
                output=transcript,
                assertions=[
                    {"name": r.name, "passed": r.passed, "detail": r.detail} for r in results
                ],
                trace_id=trace_id,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
            )
            report.samples.append(
                SampleReport(
                    sample_idx=sample_idx,
                    passed=passed,
                    output=transcript,
                    results=results,
                    trace_id=trace_id,
                    duration_ms=duration_ms,
                    cost_usd=cost_usd,
                )
            )
        reports.append(report)

    return RunSummary(run_id=run_id, reports=reports, samples_per_case=samples)
