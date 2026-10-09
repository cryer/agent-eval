import pytest
import yaml

from agent_eval.assertions import CaseContext, OutputContains
from agent_eval.judge import Judge, JudgeVerdict, StubBackend
from agent_eval.simulation import (
    GoalAssertion,
    GoalSpec,
    Persona,
    Scenario,
    SimulatedUser,
    format_transcript,
    load_scenarios,
    rollout,
    run_scenarios,
)


class FakeJudge:
    """Hand-rolled judge: returns a fixed verdict, records calls."""

    def __init__(self, passed: bool, score: float = 1.0):
        self.verdict = JudgeVerdict(passed, score, "fake rationale")
        self.backend_name = "fake"
        self.calls = []

    def judge(self, case_input, output, rubric):
        self.calls.append((case_input, output, rubric))
        return self.verdict


def chat_agent(history):
    return f"reply to: {history[-1]['content']}"


class TestSimulatedUserScriptMode:
    def test_replays_script_in_order(self):
        user = SimulatedUser(Persona(style="s", hidden_goal="g", script=["hi", "again"]))
        history = []
        assert user.next_message(history) == "hi"
        history.append({"role": "user", "content": "hi"})
        history.append({"role": "assistant", "content": "hello"})
        assert user.next_message(history) == "again"

    def test_returns_none_when_script_exhausted(self):
        user = SimulatedUser(Persona(style="s", hidden_goal="g", script=["hi"]))
        history = [{"role": "user", "content": "hi"}]
        assert user.next_message(history) is None

    def test_no_script_no_backend_raises(self):
        with pytest.raises(ValueError, match="no script"):
            SimulatedUser(Persona(style="s", hidden_goal="g"))


class FakeUserBackend:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class TestSimulatedUserLLMMode:
    def test_backend_reply_is_used(self):
        backend = FakeUserBackend("where is my order?")
        user = SimulatedUser(
            Persona(style="impatient", hidden_goal="find order"), backend=backend
        )
        assert user.next_message([]) == "where is my order?"
        prompt = backend.prompts[0]
        assert "impatient" in prompt
        assert "find order" in prompt

    def test_done_token_ends_conversation(self):
        backend = FakeUserBackend("[DONE]")
        user = SimulatedUser(Persona(style="s", hidden_goal="g"), backend=backend)
        assert user.next_message([]) is None


class TestGoalAssertionSemantics:
    def _ctx(self):
        return CaseContext(case_id="s1", input="goal", output="transcript", turns=2)

    def test_expect_achieved_and_achieved_passes(self):
        a = GoalAssertion(FakeJudge(passed=True, score=0.9), GoalSpec("r", True))
        assert a.check(self._ctx()).passed is True

    def test_expect_achieved_but_not_achieved_fails(self):
        a = GoalAssertion(FakeJudge(passed=False, score=0.0), GoalSpec("r", True))
        r = a.check(self._ctx())
        assert r.passed is False
        assert "not achieved" in r.detail

    def test_adversarial_expect_not_achieved_judge_says_achieved_fails(self):
        # the inversion: attacker's goal achieved -> agent loses
        a = GoalAssertion(FakeJudge(passed=True, score=0.9), GoalSpec("r", False))
        r = a.check(self._ctx())
        assert r.passed is False
        assert "expected not achieved" in r.detail

    def test_adversarial_expect_not_achieved_judge_says_not_achieved_passes(self):
        a = GoalAssertion(FakeJudge(passed=False, score=0.0), GoalSpec("r", False))
        assert a.check(self._ctx()).passed is True

    def test_low_score_counts_as_not_achieved(self):
        # passed=True but score < 0.5 -> treated as goal not achieved
        a = GoalAssertion(FakeJudge(passed=True, score=0.3), GoalSpec("r", True))
        assert a.check(self._ctx()).passed is False


class TestFormatTranscript:
    def test_format(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert format_transcript(history) == "user: hi\nassistant: hello"

    def test_empty(self):
        assert format_transcript([]) == ""


class TestRollout:
    def _scenario(self, script, max_turns=8):
        return Scenario(
            id="s1",
            persona=Persona(style="polite", hidden_goal="get answer", script=script),
            max_turns=max_turns,
        )

    def test_script_user_runs_to_completion(self, storage, tracer):
        scenario = self._scenario(["q1", "q2"])
        user = SimulatedUser(scenario.persona)
        history, ended, trace_id, duration_ms = rollout(
            chat_agent, user, scenario, tracer
        )
        assert ended == "user_done"
        assert [m["role"] for m in history] == ["user", "assistant"] * 2
        assert history[1]["content"] == "reply to: q1"
        assert duration_ms >= 0
        # root span + one span per turn landed in storage
        names = [s["name"] for s in storage.get_spans(trace_id)]
        assert names == ["scenario.s1", "turn.1", "turn.2"]

    def test_max_turns_caps_conversation(self, storage, tracer):
        scenario = self._scenario(["q1", "q2", "q3"], max_turns=2)
        user = SimulatedUser(scenario.persona)
        history, ended, _, _ = rollout(chat_agent, user, scenario, tracer)
        assert ended == "max_turns"
        assert sum(1 for m in history if m["role"] == "user") == 2

    def test_agent_dict_result_unwrapped(self, storage, tracer):
        scenario = self._scenario(["q1"])
        history, ended, _, _ = rollout(
            lambda h: {"output": "dict reply", "turns": 9},
            SimulatedUser(scenario.persona),
            scenario,
            tracer,
        )
        assert history[-1] == {"role": "assistant", "content": "dict reply"}


class TestLoadScenarios:
    def _write(self, tmp_path, entries):
        p = tmp_path / "scenarios.yaml"
        p.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
        return p

    def test_loads_persona_goal_and_assertions(self, tmp_path):
        p = self._write(
            tmp_path,
            [
                {
                    "id": "s1",
                    "persona": {
                        "style": "polite",
                        "hidden_goal": "get a refund",
                        "script": ["hi"],
                    },
                    "max_turns": 3,
                    "assertions": [{"output_contains": "refund"}],
                    "goal": {"rubric": 'goal is "refund"', "expect_achieved": False},
                }
            ],
        )
        scenarios = load_scenarios(p, judge=Judge(StubBackend()))
        (s,) = scenarios
        assert s.id == "s1"
        assert s.max_turns == 3
        assert s.persona.script == ["hi"]
        assert s.goal.expect_achieved is False
        # rule assertion + appended GoalAssertion
        assert len(s.assertions) == 2
        assert isinstance(s.assertions[-1], GoalAssertion)

    def test_max_turns_defaults_to_eight(self, tmp_path):
        p = self._write(
            tmp_path,
            [{"id": "s", "persona": {"style": "x", "hidden_goal": "g", "script": []}}],
        )
        assert load_scenarios(p)[0].max_turns == 8

    def test_goal_without_judge_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            [
                {
                    "id": "s1",
                    "persona": {"style": "x", "hidden_goal": "g", "script": []},
                    "goal": {"rubric": "r"},
                }
            ],
        )
        with pytest.raises(ValueError, match="no judge backend"):
            load_scenarios(p, judge=None)

    def test_non_list_file_raises(self, tmp_path):
        p = tmp_path / "scenarios.yaml"
        p.write_text("not: a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML list"):
            load_scenarios(p)


class TestRunScenarios:
    def test_passing_scenario_recorded_like_a_case(self, storage, tracer):
        scenario = Scenario(
            id="s1",
            persona=Persona(style="polite", hidden_goal="g", script=["hello"]),
            max_turns=4,
            assertions=[OutputContains("reply to: hello")],
        )
        summary = run_scenarios(
            chat_agent, [scenario], storage, tracer, run_id="sim1", samples=2
        )
        assert summary.ok is True
        assert summary.passed == 1
        rows = storage.get_run_results("sim1")
        assert len(rows) == 2
        assert all(r["case_id"] == "s1" for r in rows)
        assert "user: hello" in rows[0]["output"]

    def test_goal_assertion_with_stub_judge(self, storage, tracer):
        judge = Judge(StubBackend(), backend_name="stub")
        scenario = Scenario(
            id="s1",
            persona=Persona(style="polite", hidden_goal="g", script=["hello"]),
            max_turns=4,
            assertions=[GoalAssertion(judge, GoalSpec('transcript mentions "hello"', True))],
        )
        summary = run_scenarios(chat_agent, [scenario], storage, tracer)
        assert summary.ok is True

    def test_failing_assertion_marks_sample_failed(self, storage, tracer):
        scenario = Scenario(
            id="s1",
            persona=Persona(style="s", hidden_goal="g", script=["hi"]),
            max_turns=2,
            assertions=[OutputContains("never appears")],
        )
        summary = run_scenarios(chat_agent, [scenario], storage, tracer)
        assert summary.ok is False
        assert summary.reports[0].samples[0].passed is False

    def test_samples_below_one_raises(self, storage, tracer):
        scenario = Scenario(
            id="s1", persona=Persona(style="s", hidden_goal="g", script=[]), max_turns=1
        )
        with pytest.raises(ValueError, match="samples must be >= 1"):
            run_scenarios(chat_agent, [scenario], storage, tracer, samples=0)
