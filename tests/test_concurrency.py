"""Concurrency and retry: parallel workers must produce results identical to
serial execution (same content, deterministic order), and transient LLM API
failures must be absorbed by backoff instead of failing the run."""

from __future__ import annotations

import threading
import time

import pytest

from agent_eval.retry import call_with_retries, is_retryable
from agent_eval.runner import Case, run_cases
from agent_eval.simulation import Persona, Scenario, run_scenarios
from agent_eval.storage import Storage


def _make_cases(n: int) -> list[Case]:
    return [Case(id=f"case_{i}", input=f"input {i}", assertions=[]) for i in range(n)]


def _echo_agent(user_input: str) -> str:
    return f"reply to: {user_input}"


class TestParallelRunCases:
    def test_parallel_matches_serial(self, storage, tracer):
        cases = _make_cases(6)
        serial = run_cases(_echo_agent, cases, storage, tracer, run_id="s", samples=3)
        parallel = run_cases(
            _echo_agent, cases, storage, tracer, run_id="p", samples=3, concurrency=4
        )

        assert [r.case.id for r in parallel.reports] == [c.id for c in cases]
        for serial_report, parallel_report in zip(serial.reports, parallel.reports):
            assert [s.sample_idx for s in parallel_report.samples] == [0, 1, 2]
            assert [s.output for s in parallel_report.samples] == [
                s.output for s in serial_report.samples
            ]
        # every sample of both runs was persisted
        assert len(storage.get_run_results("s")) == 18
        assert len(storage.get_run_results("p")) == 18

    def test_report_order_follows_case_order_not_completion_order(self, storage, tracer):
        # the first case is the slowest — under concurrency it finishes last,
        # but the report must still list it first
        def slow_first(user_input: str) -> str:
            if user_input == "input 0":
                time.sleep(0.15)
            return user_input

        summary = run_cases(
            slow_first, _make_cases(4), storage, tracer, run_id="o", concurrency=4
        )
        assert [r.case.id for r in summary.reports] == [f"case_{i}" for i in range(4)]

    def test_samples_actually_overlap(self, storage, tracer):
        # barrier: if workers never run concurrently, all 4 would deadlock-timeout
        barrier = threading.Barrier(4, timeout=5)

        def barrier_agent(user_input: str) -> str:
            barrier.wait()
            return user_input

        summary = run_cases(
            barrier_agent, _make_cases(2), storage, tracer, run_id="b", samples=2, concurrency=4
        )
        assert summary.ok

    def test_worker_exception_propagates(self, storage, tracer):
        def boom(_: str) -> str:
            raise RuntimeError("agent crashed")

        with pytest.raises(RuntimeError, match="agent crashed"):
            run_cases(boom, _make_cases(1), storage, tracer, run_id="x", concurrency=2)

    def test_invalid_concurrency_rejected(self, storage, tracer):
        with pytest.raises(ValueError, match="concurrency"):
            run_cases(_echo_agent, _make_cases(1), storage, tracer, concurrency=0)


class TestParallelRunScenarios:
    def _scenarios(self, n: int) -> list[Scenario]:
        return [
            Scenario(
                id=f"sc_{i}",
                persona=Persona(style="polite", hidden_goal=f"goal {i}", script=[f"hi {i}"]),
                max_turns=2,
            )
            for i in range(n)
        ]

    def test_parallel_matches_serial(self, storage, tracer):
        def chat_agent(messages: list[dict]) -> str:
            return f"got: {messages[-1]['content']}"

        scenarios = self._scenarios(4)
        serial = run_scenarios(chat_agent, scenarios, storage, tracer, run_id="s", samples=2)
        parallel = run_scenarios(
            chat_agent, scenarios, storage, tracer, run_id="p", samples=2, concurrency=3
        )

        assert [r.case.id for r in parallel.reports] == [s.id for s in scenarios]
        for serial_report, parallel_report in zip(serial.reports, parallel.reports):
            assert [s.sample_idx for s in parallel_report.samples] == [0, 1]
            assert [s.output for s in parallel_report.samples] == [
                s.output for s in serial_report.samples
            ]
        assert len(storage.get_run_results("p")) == 8

    def test_invalid_concurrency_rejected(self, storage, tracer):
        with pytest.raises(ValueError, match="concurrency"):
            run_scenarios(
                lambda m: "x", self._scenarios(1), storage, tracer, concurrency=0
            )


class FakeRateLimitError(Exception):
    status_code = 429


class FakeAuthError(Exception):
    status_code = 401


class TestRetry:
    def test_succeeds_after_transient_failures(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise FakeRateLimitError("slow down")
            return "ok"

        result = call_with_retries(flaky, max_retries=5, sleep=lambda _: None)
        assert result == "ok"
        assert len(attempts) == 3

    def test_gives_up_after_max_retries(self):
        attempts = []

        def always_limited():
            attempts.append(1)
            raise FakeRateLimitError("slow down")

        with pytest.raises(FakeRateLimitError):
            call_with_retries(always_limited, max_retries=3, sleep=lambda _: None)
        assert len(attempts) == 4  # 1 initial + 3 retries

    def test_non_retryable_raises_immediately(self):
        attempts = []

        def unauthorized():
            attempts.append(1)
            raise FakeAuthError("bad key")

        with pytest.raises(FakeAuthError):
            call_with_retries(unauthorized, max_retries=5, sleep=lambda _: None)
        assert len(attempts) == 1

    def test_backoff_grows_exponentially_with_cap(self):
        delays = []

        def always_limited():
            raise FakeRateLimitError("slow down")

        with pytest.raises(FakeRateLimitError):
            call_with_retries(
                always_limited,
                max_retries=10,
                base_delay=1.0,
                max_delay=8.0,
                sleep=delays.append,
            )
        # jitter makes exact values random, but the envelope is bounded:
        # delay_i in [0.5 * 2^i, 1.5 * 2^i], capped at [4.0, 12.0]
        assert len(delays) == 10
        assert delays[0] < delays[3]  # growing before the cap
        assert all(d <= 12.0 for d in delays)  # capped (8.0 * 1.5 jitter)

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (FakeRateLimitError("429"), True),
            (FakeAuthError("401"), False),
            (Exception("plain error"), False),
        ],
    )
    def test_is_retryable_classification(self, exc, expected):
        assert is_retryable(exc) is expected

    def test_retryable_by_exception_name_without_status(self):
        class APITimeoutError(Exception):
            pass

        class SomeRandomError(Exception):
            pass

        assert is_retryable(APITimeoutError()) is True
        assert is_retryable(SomeRandomError()) is False
