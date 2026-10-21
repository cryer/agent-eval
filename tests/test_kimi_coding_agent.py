"""Offline tests for the Kimi K3 coding agent: a scripted fake litellm module
drives the function-calling loop against a real tmp workspace, so tool
execution, sandboxing and span instrumentation are checked without network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from examples import kimi_coding_agent as kca


BUGGY_CALC = """def add(a, b):
    return a - b  # BUG: should be a + b


def multiply(a, b):
    return a * b
"""

FIXED_CALC = BUGGY_CALC.replace("return a - b", "return a + b")

CHECK_PY = """from calc import add, multiply

assert add(2, 3) == 5
assert multiply(2, 3) == 6
print("all checks passed")
"""


@pytest.fixture
def env_dir(tmp_path):
    (tmp_path / "calc.py").write_text(BUGGY_CALC, encoding="utf-8")
    (tmp_path / "check.py").write_text(CHECK_PY, encoding="utf-8")
    return str(tmp_path)


def _tool_call_resp(calls):
    """calls: list of (name, arguments-dict) tuples for one assistant turn."""
    tcs = [
        SimpleNamespace(
            id=f"call_{i}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for i, (name, args) in enumerate(calls)
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tcs))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def _final_resp(content="已修复 add() 并通过检查。"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=150, completion_tokens=30),
    )


@pytest.fixture(autouse=True)
def _fresh_agent_tracer():
    """Module-level tracer is an OTel ProxyTracer caching the first provider;
    re-create it per test so spans land in the current test's storage."""
    from opentelemetry import trace

    kca.tracer = trace.get_tracer("kimi-coding-agent")


@pytest.fixture
def fake_litellm(monkeypatch):
    """Install a fake litellm module; returns namespace with responses/calls."""
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


def _tool_msgs(call_kwargs):
    return [m for m in call_kwargs["messages"] if m["role"] == "tool"]


class TestToolLoop:
    def test_fix_bug_end_to_end(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses += [
            _tool_call_resp([("read_file", {"path": "calc.py"})]),
            _tool_call_resp([("write_file", {"path": "calc.py", "content": FIXED_CALC})]),
            _tool_call_resp([("run_command", {"command": "python check.py"})]),
            _final_resp(),
        ]
        result = kca.agent("修复 calc.py 的 bug 并让 check.py 通过", env_dir)

        assert result["turns"] == 4
        # the workspace really got fixed and the checker really passes
        assert (Path(env_dir) / "calc.py").read_text(encoding="utf-8") == FIXED_CALC

        # tool results fed back: the last one is run_command carrying exit_code 0
        run_result = json.loads(_tool_msgs(fake_litellm.calls[3])[-1]["content"])
        assert run_result["exit_code"] == 0
        assert "all checks passed" in run_result["output"]

    def test_spans_follow_instrumentation_contract(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses += [
            _tool_call_resp([("read_file", {"path": "calc.py"})]),
            _final_resp(),
        ]
        kca.agent("看下 calc.py", env_dir)

        rows = _span_rows(storage)
        llm_spans = [attrs for name, attrs in rows if name == "llm.chat"]
        tool_spans = [attrs for name, attrs in rows if name == "tool.read_file"]
        assert len(llm_spans) == 2
        assert len(tool_spans) == 1

        assert all(attrs["prompt.version"] for attrs in llm_spans)
        assert llm_spans[0]["gen_ai.usage.prompt_tokens"] == 100
        assert llm_spans[0]["gen_ai.usage.cost_usd"] > 0

        assert tool_spans[0]["tool.name"] == "read_file"
        assert "calc.py" in tool_spans[0]["tool.input"]
        assert "return a - b" in tool_spans[0]["tool.output"]

    def test_direct_answer_without_tool_call(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses.append(_final_resp("calc.py 提供 add 和 multiply。"))
        result = kca.agent("解释 calc.py，不要修改任何文件", env_dir)

        assert result["turns"] == 1
        assert (Path(env_dir) / "calc.py").read_text(encoding="utf-8") == BUGGY_CALC
        assert [name for name, _ in _span_rows(storage)] == ["llm.chat"]

    def test_unknown_tool_error_fed_back(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses += [
            _tool_call_resp([("delete_everything", {})]),
            _final_resp(),
        ]
        kca.agent("搞点破坏", env_dir)

        err = json.loads(_tool_msgs(fake_litellm.calls[1])[0]["content"])
        assert "unknown tool" in err["error"]

    def test_max_turns_fallback(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses += [
            _tool_call_resp([("list_files", {})]) for _ in range(kca.MAX_TURNS)
        ]
        result = kca.agent("修 bug", env_dir)

        assert result["turns"] == kca.MAX_TURNS
        assert "最大轮次" in result["output"]

    def test_user_message_carries_task_and_env_dir(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses.append(_final_resp())
        kca.agent("任务X", env_dir)

        user_msg = fake_litellm.calls[0]["messages"][1]
        assert user_msg["role"] == "user"
        assert "任务X" in user_msg["content"]
        assert env_dir in user_msg["content"]


class TestSandbox:
    def test_path_escape_rejected(self, storage, tracer, fake_litellm, env_dir):
        fake_litellm.responses += [
            _tool_call_resp([("read_file", {"path": "../secret.txt"})]),
            _final_resp(),
        ]
        kca.agent("读一下 ../secret.txt", env_dir)

        err = json.loads(_tool_msgs(fake_litellm.calls[1])[0]["content"])
        assert "escapes workspace" in err["error"]

    def test_write_outside_workspace_rejected(self, env_dir):
        with pytest.raises(ValueError, match="escapes workspace"):
            kca.write_file(env_dir, "../evil.py", "x")

    def test_absolute_path_outside_rejected(self, env_dir):
        with pytest.raises(ValueError, match="escapes workspace"):
            kca.read_file(env_dir, str(Path(env_dir).parent / "outside.txt"))


class TestToolImpls:
    def test_list_files(self, env_dir):
        result = kca.list_files(env_dir)
        assert sorted(result["files"]) == ["calc.py", "check.py"]

    def test_list_files_not_a_directory(self, env_dir):
        assert "error" in kca.list_files(env_dir, "calc.py")

    def test_read_missing_file(self, env_dir):
        assert "error" in kca.read_file(env_dir, "nope.py")

    def test_run_command_exit_code(self, env_dir):
        ok = kca.run_command(env_dir, "python -c \"print('hi')\"")
        assert ok["exit_code"] == 0
        assert "hi" in ok["output"]

        bad = kca.run_command(env_dir, "python -c \"import sys; sys.exit(3)\"")
        assert bad["exit_code"] == 3

    def test_run_command_runs_in_env_dir(self, env_dir):
        result = kca.run_command(env_dir, "python check.py")
        # buggy fixture: assertion fails -> non-zero, and the error names add()
        assert result["exit_code"] != 0
        assert "AssertionError" in result["output"]


class TestMissingDependency:
    def test_missing_litellm_raises_helpful_error(self, monkeypatch, env_dir):
        monkeypatch.setitem(sys.modules, "litellm", None)
        with pytest.raises(RuntimeError, match="agent-eval\\[llm\\]"):
            kca.agent("hi", env_dir)
