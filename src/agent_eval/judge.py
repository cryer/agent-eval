from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from .assertions import AssertionResult, CaseContext
from .retry import call_with_retries

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial evaluator for an LLM agent. Decide whether the agent's
output satisfies the rubric, given the user input.

## Rubric
{rubric}

## User input
{case_input}

## Agent output
{output}

Respond with a single JSON object, no other text:
{{"pass": true|false, "score": <float 0-1>, "rationale": "<one sentence>"}}
"""


@dataclass
class JudgeVerdict:
    passed: bool
    score: float
    rationale: str


class JudgeBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


class LiteLLMBackend:
    """Real model backend via LiteLLM (optional dependency).

    Without extra config it uses the model's default provider credentials.
    With KIMI_API_BASE / KIMI_API_KEY set (or explicit api_base/api_key),
    the model is routed to that OpenAI-compatible endpoint — e.g. Kimi K3.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_base: str | None = None, api_key: str | None = None):
        self.model = model
        self.api_base = api_base or os.environ.get("KIMI_API_BASE")
        self.api_key = api_key or os.environ.get("KIMI_API_KEY")

    def complete(self, prompt: str) -> str:
        try:
            from litellm import completion
        except ImportError as e:
            raise RuntimeError(
                "LiteLLM backend requires `pip install agent-eval[llm]` "
                "(or use --judge-backend stub for offline runs)"
            ) from e
        model = self.model
        kwargs: dict = {}
        if self.api_base:
            # custom OpenAI-compatible endpoint: litellm needs the openai/ prefix
            if "/" not in model:
                model = f"openai/{model}"
            kwargs["api_base"] = self.api_base
            kwargs["api_key"] = self.api_key
        def _call():
            return completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=180,
                **kwargs,
            )

        # rate limits are expected under parallel workers — back off, don't fail
        resp = call_with_retries(_call)
        return resp.choices[0].message.content


class StubBackend:
    """Deterministic offline judge: the output passes iff every double-quoted
    keyword in the rubric appears in the agent output. Exists so the eval loop
    is testable without an API key — and its disagreements with human labels
    demonstrate why judge calibration matters."""

    def complete(self, prompt: str) -> str:
        rubric = _section(prompt, "Rubric")
        output = _section(prompt, "Agent output")
        keywords = re.findall(r'"([^"]+)"', rubric)
        missing = [k for k in keywords if k not in output]
        verdict = {
            "pass": not missing,
            "score": 1.0 if not missing else 0.0,
            "rationale": (
                "all rubric keywords present"
                if not missing
                else f"missing keywords: {', '.join(missing)}"
            ),
        }
        return json.dumps(verdict, ensure_ascii=False)


def _section(prompt: str, title: str) -> str:
    m = re.search(rf"## {re.escape(title)}\n(.*?)(?=\n## |\Z)", prompt, re.S)
    return m.group(1) if m else ""


_BACKENDS = {"litellm": LiteLLMBackend, "stub": StubBackend}


def build_backend(name: str, model: str = "gpt-4o-mini") -> JudgeBackend:
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"unknown judge backend {name!r}; available: {sorted(_BACKENDS)}")
    return cls(model) if name == "litellm" else cls()


class Judge:
    def __init__(self, backend: JudgeBackend, backend_name: str = "unknown"):
        self.backend = backend
        self.backend_name = backend_name

    def judge(self, case_input: str, output: str, rubric: str) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            rubric=rubric, case_input=case_input, output=output
        )
        raw = self.backend.complete(prompt)
        return parse_verdict(raw)


def parse_verdict(raw: str) -> JudgeVerdict:
    """Extract the JSON verdict from a judge response, tolerating prose around it."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return JudgeVerdict(False, 0.0, f"unparseable judge response: {raw[:120]!r}")
    try:
        data = json.loads(m.group(0))
        score = float(data.get("score", 1.0 if data.get("pass") else 0.0))
        return JudgeVerdict(
            passed=bool(data.get("pass", False)),
            score=max(0.0, min(1.0, score)),
            rationale=str(data.get("rationale", "")),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return JudgeVerdict(False, 0.0, f"invalid judge JSON: {e}")


class JudgeAssertion:
    """Assertion backed by an LLM-as-judge. Spec: `- judge: {rubric: ..., threshold: 0.7}`."""

    def __init__(self, judge: Judge, spec: dict):
        self.judge = judge
        self.rubric = spec["rubric"]
        self.threshold = float(spec.get("threshold", 0.5))
        self.name = f"judge(threshold={self.threshold}, rubric={self.rubric[:40]!r})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        verdict = self.judge.judge(ctx.input, ctx.output, self.rubric)
        passed = verdict.passed and verdict.score >= self.threshold
        detail = (
            f"[{self.judge.backend_name}] score={verdict.score:.2f} {verdict.rationale}"
        )
        return AssertionResult(self.name, passed, "" if passed else detail)
