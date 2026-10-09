import json

import pytest

from agent_eval.assertions import CaseContext
from agent_eval.judge import (
    Judge,
    JudgeAssertion,
    JudgeVerdict,
    LiteLLMBackend,
    StubBackend,
    build_backend,
    parse_verdict,
)


class FakeBackend:
    """Offline backend returning a canned raw response."""

    def __init__(self, raw: str):
        self.raw = raw
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.raw


def make_ctx(output: str = "some output") -> CaseContext:
    return CaseContext(case_id="c1", input="user question", output=output, turns=1)


class TestParseVerdict:
    def test_clean_json(self):
        v = parse_verdict('{"pass": true, "score": 0.9, "rationale": "ok"}')
        assert v.passed is True
        assert v.score == pytest.approx(0.9)
        assert v.rationale == "ok"

    def test_prose_around_json_is_tolerated(self):
        raw = 'Sure, here is my verdict:\n{"pass": false, "score": 0.1, "rationale": "bad"}\nHope that helps!'
        v = parse_verdict(raw)
        assert v.passed is False
        assert v.score == pytest.approx(0.1)

    def test_garbage_returns_failed_zero_score(self):
        v = parse_verdict("I cannot decide, no JSON here")
        assert v.passed is False
        assert v.score == 0.0
        assert "unparseable" in v.rationale

    def test_invalid_json_returns_failed(self):
        v = parse_verdict("{not: valid json}")
        assert v.passed is False
        assert v.score == 0.0
        assert "invalid judge JSON" in v.rationale

    def test_score_clamped_above_one(self):
        v = parse_verdict('{"pass": true, "score": 1.7}')
        assert v.score == 1.0

    def test_score_clamped_below_zero(self):
        v = parse_verdict('{"pass": false, "score": -0.5}')
        assert v.score == 0.0

    def test_missing_score_defaults_from_pass(self):
        assert parse_verdict('{"pass": true}').score == 1.0
        assert parse_verdict('{"pass": false}').score == 0.0

    def test_missing_pass_defaults_to_false(self):
        assert parse_verdict('{"score": 0.9}').passed is False


class TestStubBackend:
    def test_passes_when_all_keywords_present(self):
        judge = Judge(StubBackend(), backend_name="stub")
        v = judge.judge(
            "退款政策？",
            "签收 7 天内可退款，您好",
            '回答必须包含 "7 天" 和 "您好"',
        )
        assert v.passed is True
        assert v.score == 1.0
        assert "all rubric keywords present" in v.rationale

    def test_fails_when_a_keyword_is_missing(self):
        judge = Judge(StubBackend(), backend_name="stub")
        v = judge.judge(
            "退款政策？",
            "签收 7 天内可退款",
            '回答必须包含 "7 天" 和 "您好"',
        )
        assert v.passed is False
        assert v.score == 0.0
        assert "missing keywords" in v.rationale
        assert "您好" in v.rationale

    def test_prompt_template_feeds_sections(self):
        backend = StubBackend()
        judge = Judge(backend)
        judge.judge("the input", 'has "kw"', 'needs "kw"')
        # the stub parses ## Rubric / ## Agent output sections; if the
        # template dropped them, no keywords would be found and this would fail
        v = judge.judge("the input", 'has "kw"', 'needs "kw"')
        assert v.passed is True


class TestJudgeAssertion:
    def test_pass_true_below_threshold_fails(self):
        backend = FakeBackend('{"pass": true, "score": 0.3, "rationale": "weak"}')
        judge = Judge(backend, backend_name="fake")
        a = JudgeAssertion(judge, {"rubric": "any", "threshold": 0.7})
        result = a.check(make_ctx())
        assert result.passed is False
        assert "score=0.30" in result.detail

    def test_pass_true_at_threshold_passes(self):
        backend = FakeBackend('{"pass": true, "score": 0.7, "rationale": "ok"}')
        judge = Judge(backend, backend_name="fake")
        a = JudgeAssertion(judge, {"rubric": "any", "threshold": 0.7})
        assert a.check(make_ctx()).passed is True

    def test_default_threshold_is_half(self):
        backend = FakeBackend('{"pass": true, "score": 0.6, "rationale": "ok"}')
        judge = Judge(backend)
        a = JudgeAssertion(judge, {"rubric": "any"})
        assert a.threshold == 0.5
        assert a.check(make_ctx()).passed is True

    def test_judge_fail_always_fails(self):
        backend = FakeBackend('{"pass": false, "score": 0.9, "rationale": "no"}')
        judge = Judge(backend)
        a = JudgeAssertion(judge, {"rubric": "any", "threshold": 0.1})
        assert a.check(make_ctx()).passed is False

    def test_backend_receives_rubric_input_output(self):
        backend = FakeBackend('{"pass": true, "score": 1.0}')
        judge = Judge(backend)
        a = JudgeAssertion(judge, {"rubric": "the rubric"})
        a.check(make_ctx(output="the output"))
        prompt = backend.prompts[0]
        assert "the rubric" in prompt
        assert "user question" in prompt
        assert "the output" in prompt


class TestBuildBackend:
    def test_stub(self):
        assert isinstance(build_backend("stub"), StubBackend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown judge backend"):
            build_backend("wat")

    def test_litellm_constructs_without_network(self):
        b = build_backend("litellm", model="k3")
        assert isinstance(b, LiteLLMBackend)
        assert b.model == "k3"


class TestLiteLLMBackendConstruction:
    def test_no_env_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_BASE", raising=False)
        monkeypatch.delenv("KIMI_API_KEY", raising=False)
        b = LiteLLMBackend()
        assert b.api_base is None
        assert b.api_key is None

    def test_env_vars_are_picked_up(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_BASE", "https://example.invalid/v1")
        monkeypatch.setenv("KIMI_API_KEY", "sk-test")
        b = LiteLLMBackend()
        assert b.api_base == "https://example.invalid/v1"
        assert b.api_key == "sk-test"

    def test_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_BASE", "https://env.invalid")
        b = LiteLLMBackend(api_base="https://explicit.invalid", api_key="sk-x")
        assert b.api_base == "https://explicit.invalid"
        assert b.api_key == "sk-x"

    def test_complete_without_litellm_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("KIMI_API_BASE", raising=False)
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "litellm" or name.startswith("litellm."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(RuntimeError, match=r"agent-eval\[llm\]"):
            LiteLLMBackend().complete("hello")
