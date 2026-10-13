"""Toy coding agent: demonstrates the (task, env_dir) contract for coding cases.

The runner copies the case's `env` fixture into a scratch directory and passes
it as `env_dir`; the agent edits files there, and assertions check the final
workspace state (command_exit / file_diff_max_files) instead of the text output.

Offline and deterministic — the demo stand-in for a real LLM-driven coding agent
(opencode / codex style), which would read the task, inspect env_dir, and edit.
"""

from pathlib import Path


def agent(task: str, env_dir: str) -> dict:
    if "不要修改" in task or "explain" in task.lower():
        return {
            "output": "calc.py provides add(a, b) and multiply(a, b); no files changed",
            "turns": 1,
        }
    calc = Path(env_dir) / "calc.py"
    src = calc.read_text(encoding="utf-8")
    if "return a - b" in src:
        calc.write_text(src.replace("return a - b", "return a + b"), encoding="utf-8")
        return {"output": "fixed add(): a - b -> a + b", "turns": 1}
    return {"output": "nothing to fix", "turns": 1}
