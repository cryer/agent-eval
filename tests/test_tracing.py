import hashlib

from opentelemetry.sdk.trace.export import SpanExportResult

from agent_eval.tracing import (
    COST_PER_MTOK,
    SQLiteSpanExporter,
    estimate_cost,
    init_tracing,
    prompt_version,
    set_token_usage,
)


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestPromptVersion:
    def test_stable_for_same_input(self):
        assert prompt_version("same prompt") == prompt_version("same prompt")

    def test_changes_with_input(self):
        assert prompt_version("v1") != prompt_version("v2")

    def test_matches_sha256_prefix(self):
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()[:12]
        assert prompt_version("hello") == expected
        assert len(prompt_version("anything")) == 12

    def test_unicode(self):
        assert prompt_version("退款政策") == prompt_version("退款政策")


class TestEstimateCost:
    def test_unknown_model_is_zero(self):
        assert estimate_cost("no-such-model", 1_000_000, 1_000_000) == 0.0

    def test_known_model(self):
        p_rate, c_rate = COST_PER_MTOK["gpt-4o-mini"]
        cost = estimate_cost("gpt-4o-mini", 2_000_000, 1_000_000)
        assert cost == (2_000_000 * p_rate + 1_000_000 * c_rate) / 1_000_000

    def test_zero_tokens(self):
        assert estimate_cost("gpt-4o", 0, 0) == 0.0


class TestSetTokenUsage:
    def test_sets_all_attributes(self):
        span = RecordingSpan()
        set_token_usage(span, "gpt-4o-mini", 1000, 500)
        attrs = span.attributes
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert attrs["gen_ai.usage.prompt_tokens"] == 1000
        assert attrs["gen_ai.usage.completion_tokens"] == 500
        assert attrs["gen_ai.usage.cost_usd"] == estimate_cost("gpt-4o-mini", 1000, 500)


class TestInitTracing:
    def test_spans_exported_to_sqlite(self, storage):
        tracer = init_tracing(storage)
        with tracer.start_as_current_span("root") as root:
            root.set_attribute("k", "v")
            trace_id = f"{root.get_span_context().trace_id:032x}"
            with tracer.start_as_current_span("child"):
                pass
        spans = storage.get_spans(trace_id)
        assert [s["name"] for s in spans] == ["root", "child"]
        assert spans[0]["attributes"] == {"k": "v"}
        assert spans[1]["parent_span_id"] == spans[0]["span_id"]

    def test_exporter_failure_returns_failure_result(self, storage):
        exporter = SQLiteSpanExporter(storage)
        # a bogus object without the span interface makes insert_span raise
        result = exporter.export([object()])
        assert result is SpanExportResult.FAILURE

    def test_exporter_success(self, storage, fake_span):
        exporter = SQLiteSpanExporter(storage)
        result = exporter.export([fake_span(trace_id=7, span_id=1)])
        assert result is SpanExportResult.SUCCESS
        assert len(storage.get_spans(f"{7:032x}")) == 1
