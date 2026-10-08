from __future__ import annotations

import hashlib

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from .storage import Storage


class SQLiteSpanExporter(SpanExporter):
    """Exports every finished span into the local trace DB."""

    def __init__(self, storage: Storage):
        self._storage = storage

    def export(self, spans) -> SpanExportResult:
        try:
            for span in spans:
                self._storage.insert_span(span)
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def init_tracing(storage: Storage, service_name: str = "agent-eval"):
    """Set up a global tracer provider that persists spans to SQLite."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(storage)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def prompt_version(prompt: str) -> str:
    """Stable short hash of a prompt — the hook for 'prompt changed -> rerun regression'."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# USD per 1M tokens: (prompt, completion)
COST_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


def set_token_usage(span, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record token usage and estimated cost on an LLM span."""
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
    span.set_attribute("gen_ai.usage.cost_usd", estimate_cost(model, prompt_tokens, completion_tokens))
