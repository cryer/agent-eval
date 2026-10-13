"""Offline tests for the Kimi K3 agent: a scripted fake litellm module stands in
for the real API, so the tool-calling loop and span instrumentation are checked
without network or API key."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from examples import kimi_agent


def _tool_call_resp(name="lookup_order", arguments='{"order_query": "退款"}'):
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def _final_resp(content="根据退款政策，签收 7 天内可无理由退款。"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=30),
    )


@pytest.fixture(autouse=True)
def _fresh_agent_tracer():
    """The module-level tracer in kimi_agent is an OTel ProxyTracer that caches
    the real provider on first use; re-create it per test so spans go to the
    current test's storage, not a previous test's closed DB."""
    from opentelemetry import trace

    kimi_agent.tracer = trace.get_tracer("kimi-agent")


@pytest.fixture
def fake_litellm(monkeypatch):
    """Install a fake litellm module; returns (module, calls) where calls records kwargs."""
    responses: list = []
    calls: list[dict] = []

    mod = ModuleType("litellm")

    def completion(**kwargs):
        calls.append(kwargs)
        if not responses:
            raise AssertionError("fake litellm: unexpected extra completion() call")
        return responses.pop(0)

    mod.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return SimpleNamespace(responses=responses, calls=calls)


def _span_rows(storage):
    rows = storage._conn.execute("SELECT name, attributes FROM spans").fetchall()
    return [(r["name"], json.loads(r["attributes"])) for r in rows]


class TestToolLoop:
    def test_tool_call_then_answer(self, storage, tracer, fake_litellm):
        fake_litellm.responses += [_tool_call_resp(), _final_resp()]
        result = kimi_agent.agent("我要退款")

        assert result["output"].startswith("根据退款政策")
        assert result["turns"] == 2
        assert len(fake_litellm.calls) == 2

        # tool result was fed back to the model as a tool message
        second_messages = fake_litellm.calls[1]["messages"]
        tool_msgs = [m for m in second_messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert json.loads(tool_msgs[0]["content"])["refundable"] is True

    def test_spans_follow_instrumentation_contract(self, storage, tracer, fake_litellm):
        fake_litellm.responses += [_tool_call_resp(), _final_resp()]
        kimi_agent.agent("我要退款")

        rows = _span_rows(storage)
        llm_spans = [attrs for name, attrs in rows if name == "llm.chat"]
        tool_spans = [attrs for name, attrs in rows if name == "tool.lookup_order"]
        assert len(llm_spans) == 2
        assert len(tool_spans) == 1

        assert all(attrs["prompt.version"] for attrs in llm_spans)
        assert llm_spans[0]["gen_ai.usage.prompt_tokens"] == 100
        assert llm_spans[0]["gen_ai.usage.completion_tokens"] == 20
        assert llm_spans[0]["gen_ai.usage.cost_usd"] > 0

        assert tool_spans[0]["tool.name"] == "lookup_order"
        assert "order_query" in tool_spans[0]["tool.input"]
        assert "refundable" in tool_spans[0]["tool.output"]

    def test_direct_answer_without_tool_call(self, storage, tracer, fake_litellm):
        fake_litellm.responses.append(_final_resp("您好，请问有什么可以帮您？"))
        result = kimi_agent.agent("你好")

        assert result["turns"] == 1
        assert result["output"].startswith("您好")
        rows = _span_rows(storage)
        assert [name for name, _ in rows] == ["llm.chat"]

    def test_unknown_tool_error_fed_back(self, storage, tracer, fake_litellm):
        fake_litellm.responses += [_tool_call_resp(name="delete_everything", arguments="{}"), _final_resp()]
        result = kimi_agent.agent("搞点破坏")

        assert result["turns"] == 2
        tool_msgs = [m for m in fake_litellm.calls[1]["messages"] if m["role"] == "tool"]
        assert "unknown tool" in json.loads(tool_msgs[0]["content"])["error"]

    def test_max_turns_fallback(self, storage, tracer, fake_litellm):
        fake_litellm.responses += [_tool_call_resp()] * 10
        result = kimi_agent.agent("我要退款")

        assert result["turns"] == 4
        assert "转人工" in result["output"]


class TestChatContract:
    def test_chat_prepends_system_prompt(self, storage, tracer, fake_litellm):
        fake_litellm.responses.append(_final_resp("reply"))
        out = kimi_agent.chat([{"role": "user", "content": "hi"}])

        assert out == "reply"
        messages = fake_litellm.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == kimi_agent.SYSTEM_PROMPT
        assert messages[1] == {"role": "user", "content": "hi"}


class TestMissingDependency:
    def test_missing_litellm_raises_helpful_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "litellm", None)
        with pytest.raises(RuntimeError, match="agent-eval\\[llm\\]"):
            kimi_agent.agent("hi")
