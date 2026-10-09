"""Shared fixtures: OTel isolation, tmp-path storage, fake span factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.util import _once

from agent_eval.storage import Storage
from agent_eval.tracing import init_tracing


@pytest.fixture(autouse=True)
def _reset_otel_tracer_provider():
    """Each test gets a fresh global tracer provider.

    OTel's set_tracer_provider only honors the first call; without resetting
    these globals, later init_tracing() calls would silently keep exporting
    spans into the first test's (already closed) database.
    """
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = _once.Once()
    yield
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = _once.Once()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "eval.db")


@pytest.fixture
def storage(db_path):
    s = Storage(db_path)
    yield s
    s.close()


@pytest.fixture
def tracer(storage):
    return init_tracing(storage)


def make_fake_span(
    trace_id: int = 0xABC,
    span_id: int = 0x1,
    parent_span_id: int | None = None,
    name: str = "span",
    start_time: int = 1_000,
    end_time: int = 2_000,
    attributes: dict | None = None,
    status: str = "OK",
):
    """Minimal stand-in for an OTel ReadableSpan, matching what
    Storage.insert_span reads off a span object."""
    return SimpleNamespace(
        get_span_context=lambda: SimpleNamespace(trace_id=trace_id, span_id=span_id),
        parent=None if parent_span_id is None else SimpleNamespace(span_id=parent_span_id),
        attributes=attributes or {},
        name=name,
        start_time=start_time,
        end_time=end_time,
        status=SimpleNamespace(status_code=SimpleNamespace(name=status)),
    )


@pytest.fixture
def fake_span():
    return make_fake_span
