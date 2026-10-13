from __future__ import annotations

import os
import re
import signal
import subprocess
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
    # Scratch copy of the case's fixture env (see Case.env); None for text-only cases.
    env_dir: str | None = None


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


class CommandExit:
    """Run a shell command inside the case env copy and assert on its exit code.

    Spec: {cmd: "python check.py", expect: 0, timeout: 60}. `expect` defaults to 0.
    `{env}` inside cmd expands to the env copy path. Requires the case to have an env.
    """

    def __init__(self, spec: dict[str, Any]):
        if not isinstance(spec, dict) or not spec.get("cmd"):
            raise ValueError(f"command_exit expects {{cmd: ..., expect?: int}}, got: {spec!r}")
        self.cmd = str(spec["cmd"])
        self.expect = int(spec.get("expect", 0))
        self.timeout = float(spec.get("timeout", 60))
        self.name = f"command_exit({self.cmd!r}, expect={self.expect})"

    def _run(self, cmd: str, cwd: str) -> subprocess.CompletedProcess:
        """Popen-based run so a timeout kills the whole process tree — with
        shell=True, killing only the shell would orphan the real command."""
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
                )
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def check(self, ctx: CaseContext) -> AssertionResult:
        if ctx.env_dir is None:
            return AssertionResult(self.name, False, "case has no env; command_exit requires one")
        cmd = self.cmd.replace("{env}", ctx.env_dir)
        try:
            proc = self._run(cmd, ctx.env_dir)
        except subprocess.TimeoutExpired:
            return AssertionResult(self.name, False, f"command timed out after {self.timeout}s")
        if proc.returncode == self.expect:
            return AssertionResult(self.name, True)
        tail = (proc.stdout + proc.stderr).strip()[-200:]
        return AssertionResult(
            self.name, False, f"exit {proc.returncode} != {self.expect}; output tail: {tail!r}"
        )


class FileDiffMaxFiles:
    """Limit how many files the agent may change inside the env copy.

    The runner git-initializes every env copy, so this diffs against the
    pristine fixture state (modified + untracked files both count).
    """

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.name = f"file_diff_max_files({self.limit})"

    def check(self, ctx: CaseContext) -> AssertionResult:
        if ctx.env_dir is None:
            return AssertionResult(self.name, False, "case has no env; file_diff_max_files requires one")
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ctx.env_dir,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return AssertionResult(self.name, False, f"env is not a git repo: {proc.stderr.strip()}")
        changed = [line for line in proc.stdout.splitlines() if line.strip()]
        passed = len(changed) <= self.limit
        detail = "" if passed else f"{len(changed)} files changed > {self.limit}: {changed[:5]}"
        return AssertionResult(self.name, passed, detail)


_REGISTRY = {
    "output_contains": OutputContains,
    "output_not_contains": OutputNotContains,
    "output_matches": OutputMatches,
    "tool_called": ToolCalled,
    "max_turns": MaxTurns,
    "max_cost_usd": MaxCostUsd,
    "max_latency_ms": MaxLatencyMs,
    "command_exit": CommandExit,
    "file_diff_max_files": FileDiffMaxFiles,
}


def build_assertions(spec_list: list[dict[str, Any]], judge=None) -> list[Assertion]:
    """Build assertion objects from a YAML/JSON spec like `- tool_called: lookup_order`.

    `judge` assertions require a Judge instance (see judge.py); pass judge=None
    only for case files that use purely rule-based assertions.
    """
    assertions: list[Assertion] = []
    for spec in spec_list or []:
        if not isinstance(spec, dict) or len(spec) != 1:
            raise ValueError(f"assertion spec must be a single-key mapping, got: {spec!r}")
        (kind, value), = spec.items()
        if kind == "judge":
            if judge is None:
                raise ValueError(
                    "case uses a `judge` assertion but no judge backend was given "
                    "(use --judge-backend litellm|stub)"
                )
            from .judge import JudgeAssertion  # lazy: judge.py imports this module

            assertions.append(JudgeAssertion(judge, value))
            continue
        cls = _REGISTRY.get(kind)
        if cls is None:
            raise ValueError(f"unknown assertion kind {kind!r}; available: {sorted(_REGISTRY)}")
        assertions.append(cls(value))
    return assertions
