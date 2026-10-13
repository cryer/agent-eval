import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from agent_eval.assertions import (
    CaseContext,
    CommandExit,
    FileDiffMaxFiles,
    build_assertions,
)
from agent_eval.runner import Case, _accepts_env, _prepare_env, load_cases, run_cases

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

PY = f'"{sys.executable}"'


def _ctx(env_dir=None, output=""):
    return CaseContext(case_id="c", input="", output=output, turns=1, env_dir=env_dir)


def _fixture(tmp_path: Path, files: dict[str, str]) -> str:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name, content in files.items():
        (fixture / name).write_text(content, encoding="utf-8")
    return str(fixture)


class TestCommandExit:
    def test_pass_on_expected_exit_code(self, tmp_path):
        (tmp_path / "ok.py").write_text("print('fine')", encoding="utf-8")
        a = CommandExit({"cmd": f"{PY} ok.py", "expect": 0})
        result = a.check(_ctx(env_dir=str(tmp_path)))
        assert result.passed is True

    def test_fail_on_wrong_exit_code_reports_tail(self, tmp_path):
        (tmp_path / "bad.py").write_text(
            "import sys; print('boom'); sys.exit(3)", encoding="utf-8"
        )
        a = CommandExit({"cmd": f"{PY} bad.py"})
        result = a.check(_ctx(env_dir=str(tmp_path)))
        assert result.passed is False
        assert "exit 3 != 0" in result.detail
        assert "boom" in result.detail

    def test_expect_nonzero_exit(self, tmp_path):
        (tmp_path / "bad.py").write_text("import sys; sys.exit(3)", encoding="utf-8")
        a = CommandExit({"cmd": f"{PY} bad.py", "expect": 3})
        assert a.check(_ctx(env_dir=str(tmp_path))).passed is True

    def test_env_placeholder_expands(self, tmp_path):
        (tmp_path / "ok.py").write_text("print('via placeholder')", encoding="utf-8")
        a = CommandExit({"cmd": f'{PY} "{{env}}/ok.py"'})
        assert a.check(_ctx(env_dir=str(tmp_path))).passed is True

    def test_requires_env(self):
        a = CommandExit({"cmd": f"{PY} -c pass"})
        result = a.check(_ctx(env_dir=None))
        assert result.passed is False
        assert "no env" in result.detail

    def test_timeout_fails(self, tmp_path):
        # shell=True spawns a grandchild that survives the timeout kill on
        # Windows, so keep the sleep short: the test pays roughly the sleep time
        (tmp_path / "slow.py").write_text(
            "import time; time.sleep(2)", encoding="utf-8"
        )
        a = CommandExit({"cmd": f"{PY} slow.py", "timeout": 0.2})
        result = a.check(_ctx(env_dir=str(tmp_path)))
        assert result.passed is False
        assert "timed out" in result.detail

    def test_invalid_spec_rejected(self):
        with pytest.raises(ValueError, match="command_exit expects"):
            build_assertions([{"command_exit": {"expect": 0}}])
        with pytest.raises(ValueError, match="command_exit expects"):
            build_assertions([{"command_exit": "not-a-mapping"}])


@needs_git
class TestFileDiffMaxFiles:
    def _prepared(self, tmp_path, files, limit):
        fixture = _fixture(tmp_path, files)
        case = Case(id="c", input="", assertions=[FileDiffMaxFiles(limit)], env=fixture)
        return _prepare_env(case)

    def test_no_changes_passes(self, tmp_path):
        env = self._prepared(tmp_path, {"a.txt": "hello"}, limit=0)
        assert FileDiffMaxFiles(0).check(_ctx(env_dir=env)).passed is True
        shutil.rmtree(env, ignore_errors=True)

    def test_modification_and_untracked_counted(self, tmp_path):
        env = self._prepared(tmp_path, {"a.txt": "hello", "b.txt": "world"}, limit=1)
        Path(env, "a.txt").write_text("changed", encoding="utf-8")  # modified
        Path(env, "c.txt").write_text("new", encoding="utf-8")  # untracked
        result = FileDiffMaxFiles(1).check(_ctx(env_dir=env))
        assert result.passed is False
        assert "2 files changed > 1" in result.detail
        shutil.rmtree(env, ignore_errors=True)

    def test_requires_env(self):
        result = FileDiffMaxFiles(1).check(_ctx(env_dir=None))
        assert result.passed is False
        assert "no env" in result.detail


class TestAcceptsEnv:
    def test_one_arg_agent(self):
        assert _accepts_env(lambda _: "x") is False

    def test_two_arg_agent(self):
        assert _accepts_env(lambda task, env_dir: "x") is True

    def test_varargs_agent(self):
        assert _accepts_env(lambda *args: "x") is True

    def test_keyword_only_second_param_does_not_count(self):
        assert _accepts_env(lambda task, *, env_dir=None: "x") is False


class TestLoadCasesEnv:
    def test_env_resolved_relative_to_cases_file(self, tmp_path):
        (tmp_path / "fixtures" / "f1").mkdir(parents=True)
        p = tmp_path / "cases.yaml"
        p.write_text(
            yaml.safe_dump(
                [
                    {"id": "with_env", "input": "x", "env": "fixtures/f1"},
                    {"id": "no_env", "input": "x"},
                ]
            ),
            encoding="utf-8",
        )
        cases = load_cases(p)
        by_id = {c.id: c for c in cases}
        assert by_id["with_env"].env == str((tmp_path / "fixtures" / "f1").resolve())
        assert by_id["no_env"].env is None


@needs_git
class TestRunCasesWithEnv:
    def test_env_case_requires_two_arg_agent(self, storage, tracer, tmp_path):
        fixture = _fixture(tmp_path, {"a.txt": "x"})
        case = Case(id="c", input="x", assertions=[], env=fixture)
        with pytest.raises(TypeError, match="env_dir"):
            run_cases(lambda _: "x", [case], storage, tracer)

    def test_agent_receives_writable_env_copy(self, storage, tracer, tmp_path):
        fixture = _fixture(tmp_path, {"a.txt": "x"})
        case = Case(id="c", input="x", assertions=[], env=fixture)

        def scratch_dirs():
            return {
                p
                for p in Path(tempfile.gettempdir()).iterdir()
                if p.name.startswith("agent_eval_c_")
            }

        before = scratch_dirs()

        def agent(task, env_dir):
            assert Path(env_dir) != Path(fixture)
            assert Path(env_dir, "a.txt").read_text() == "x"
            Path(env_dir, "out.txt").write_text("done", encoding="utf-8")
            return "ok"

        summary = run_cases(agent, [case], storage, tracer)
        assert summary.ok is True
        # original fixture untouched
        assert not Path(fixture, "out.txt").exists()
        # scratch copy cleaned up (compare against pre-existing leftovers)
        assert scratch_dirs() <= before

    def test_each_sample_gets_fresh_copy(self, storage, tracer, tmp_path):
        fixture = _fixture(tmp_path, {"counter.txt": ""})
        case = Case(id="c", input="x", assertions=[], env=fixture)

        def agent(task, env_dir):
            counter = Path(env_dir, "counter.txt")
            lines = counter.read_text(encoding="utf-8").splitlines()
            counter.write_text("\n".join(lines + ["tick"]), encoding="utf-8")
            return str(len(lines) + 1)

        summary = run_cases(agent, [case], storage, tracer, samples=2)
        # without isolation the second sample would see the first sample's tick
        assert [s.output for s in summary.reports[0].samples] == ["1", "1"]

    def test_end_to_end_coding_case(self, storage, tracer, tmp_path):
        fixture = _fixture(
            tmp_path,
            {
                "calc.py": "def add(a, b):\n    return a - b\n",
                "check.py": "from calc import add\nassert add(2, 3) == 5\n",
            },
        )
        case = Case(
            id="fix",
            input="fix the bug",
            assertions=[
                CommandExit({"cmd": f"{PY} check.py"}),
                FileDiffMaxFiles(1),
            ],
            env=fixture,
        )

        def fixing_agent(task, env_dir):
            calc = Path(env_dir, "calc.py")
            calc.write_text(
                calc.read_text(encoding="utf-8").replace("a - b", "a + b"),
                encoding="utf-8",
            )
            return "fixed"

        assert run_cases(fixing_agent, [case], storage, tracer).ok is True
        assert run_cases(lambda task, env_dir: "did nothing", [case], storage, tracer).ok is False
