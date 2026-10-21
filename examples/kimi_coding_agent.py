"""A real Kimi K3-driven coding agent — the production-grade counterpart of
toy coding_agent.py, exposing the same (task, env_dir) contract.

The runner copies the case's `env` fixture into a scratch directory and passes
it as `env_dir`; this agent inspects and edits that workspace through tools,
and assertions (command_exit / file_diff_max_files) judge the final state.

Config (environment): KIMI_API_KEY (required), KIMI_API_BASE, KIMI_MODEL.

Tools (all sandboxed to env_dir — paths escaping the workspace are rejected):
- list_files(path)                 -> {"files": [...]}
- read_file(path)                  -> {"content": ...} (truncated)
- write_file(path, content)        -> {"bytes": n}
- run_command(command)             -> {"exit_code": n, "output": ...} (60s cap)

Instrumentation contract (same as kimi_agent):
- one `llm.chat` span per model call, with prompt.version and real token usage
- one `tool.<name>` span per tool call, with tool.name / tool.input / tool.output
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from opentelemetry import trace

from agent_eval.retry import call_with_retries
from agent_eval.tracing import prompt_version, set_token_usage

tracer = trace.get_tracer("kimi-coding-agent")

MODEL = os.environ.get("KIMI_MODEL", "k3")

MAX_TURNS = 10
COMMAND_TIMEOUT_S = 60
OUTPUT_CHAR_LIMIT = 8000

SYSTEM_PROMPT = """你是 coding agent，在一个隔离的工作目录副本里完成编程任务。

工作方式：
- 先用 list_files / read_file 看清现状，再动手改；不要凭空猜测文件内容。
- 修改用 write_file 写入完整文件内容。
- 改完后用 run_command 运行项目自带的检查（如 python check.py）验证，没通过就继续修。
- 只做任务要求的最小修改，不要重构无关代码、不要新增无关文件。
- 任务只是提问/解释时，不要修改任何文件，直接回答。
- 所有路径都是相对工作目录的相对路径。"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录下的文件（相对路径）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径，默认 ."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件完整内容（覆盖）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对路径"},
                    "content": {"type": "string", "description": "完整文件内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在工作目录里跑 shell 命令，返回退出码和输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "如 python check.py"},
                },
                "required": ["command"],
            },
        },
    },
]


def _resolve(env_dir: str, path: str) -> Path:
    """Resolve a relative path inside env_dir; reject anything escaping it."""
    root = Path(env_dir).resolve()
    p = (root / (path or ".")).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"path escapes workspace: {path!r}")
    return p


def _truncate(text: str, limit: int = OUTPUT_CHAR_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


def list_files(env_dir: str, path: str = ".") -> dict:
    base = _resolve(env_dir, path)
    if not base.is_dir():
        return {"error": f"not a directory: {path!r}"}
    files = sorted(
        str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()
    )
    return {"files": files[:200]}


def read_file(env_dir: str, path: str) -> dict:
    p = _resolve(env_dir, path)
    if not p.is_file():
        return {"error": f"not a file: {path!r}"}
    return {"content": _truncate(p.read_text(encoding="utf-8", errors="replace"))}


def write_file(env_dir: str, path: str, content: str) -> dict:
    p = _resolve(env_dir, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"bytes": len(content.encode("utf-8"))}


def run_command(env_dir: str, command: str) -> dict:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=env_dir,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"command timed out after {COMMAND_TIMEOUT_S}s"}
    return {
        "exit_code": proc.returncode,
        "output": _truncate((proc.stdout + proc.stderr).strip()),
    }


_TOOL_IMPLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}


def _completion(messages: list[dict]):
    try:
        from litellm import completion
    except ImportError as e:
        raise RuntimeError(
            "kimi_coding_agent requires `pip install agent-eval[llm]`"
        ) from e
    model = MODEL
    kwargs: dict = {}
    api_base = os.environ.get("KIMI_API_BASE")
    if api_base:
        # custom OpenAI-compatible endpoint: litellm needs the openai/ prefix
        if "/" not in model:
            model = f"openai/{model}"
        kwargs["api_base"] = api_base
        kwargs["api_key"] = os.environ.get("KIMI_API_KEY")
    # the plan may cap concurrency — retry 429/5xx with backoff instead of failing
    return call_with_retries(
        lambda: completion(model=model, messages=messages, tools=TOOLS, timeout=180, **kwargs)
    )


def _record_llm_span(messages: list[dict]):
    """One model call wrapped in an llm.chat span with real token usage."""
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("prompt.version", prompt_version(SYSTEM_PROMPT))
        resp = _completion(messages)
        usage = getattr(resp, "usage", None)
        set_token_usage(
            span,
            model=MODEL,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
    return resp.choices[0].message


def _execute_tool_call(tc, env_dir: str) -> dict:
    """Run one tool call inside its own span, sandboxed to env_dir."""
    name = tc.function.name
    with tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.input", tc.function.arguments)
        impl = _TOOL_IMPLS.get(name)
        if impl is None:
            result = {"error": f"unknown tool {name!r}"}
        else:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = impl(env_dir, **args)
            except Exception as e:  # bad args, escapes — feed back to the model
                result = {"error": str(e)}
        span.set_attribute("tool.output", json.dumps(result, ensure_ascii=False))
    return result


def _loop(messages: list[dict], env_dir: str, max_turns: int = MAX_TURNS) -> dict:
    turns = 0
    for _ in range(max_turns):
        turns += 1
        msg = _record_llm_span(messages)
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return {"output": msg.content or "", "turns": turns}
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            result = _execute_tool_call(tc, env_dir)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    return {"output": "达到最大轮次，任务未完成。", "turns": turns}


def agent(task: str, env_dir: str) -> dict:
    """run 契约（coding）：任务描述 + 工作目录副本 -> {"output", "turns"}"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"工作目录: {env_dir}\n\n任务: {task}"},
    ]
    return _loop(messages, env_dir)
