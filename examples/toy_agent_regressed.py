"""Simulates a prompt regression: someone 'simplified' the system prompt and
the agent stopped stating the refund policy and skipped the order lookup.
Used to demo `agent-eval compare` catching a significant regression."""

from opentelemetry import trace

from agent_eval.tracing import prompt_version, set_token_usage

tracer = trace.get_tracer("toy-agent-regressed")

# prompt diff: the refund policy sentence was dropped
SYSTEM_PROMPT = "你是订单客服 agent。"


def agent(user_input: str) -> dict:
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("prompt.version", prompt_version(SYSTEM_PROMPT))

        if "退款" in user_input or "退货" in user_input:
            answer = "抱歉，这个我得帮您转人工处理。"
        else:
            answer = "您好，这里是订单客服，请问有什么可以帮您？"

        set_token_usage(
            span,
            model="gpt-4o-mini",
            prompt_tokens=30 + len(user_input),
            completion_tokens=len(answer),
        )
    return {"output": answer, "turns": 1}
