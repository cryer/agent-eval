"""A deterministic stand-in for a real LLM agent, used to demo the eval loop
without needing an API key. It shows the instrumentation contract:
- one LLM span with prompt.version and token usage
- tool spans with tool.name
Replace `agent` with a real model call in later phases.

Note: the deterministic branching here intentionally mirrors what a
policy-following support agent should do, so the demo cases stay stable.
"""

from opentelemetry import trace

from agent_eval.tracing import prompt_version, set_token_usage

tracer = trace.get_tracer("toy-agent")

SYSTEM_PROMPT = "你是订单客服 agent。退款政策：签收 7 天内可无理由退款，运费由商家承担。"


def agent(user_input: str) -> dict:
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("prompt.version", prompt_version(SYSTEM_PROMPT))

        if "退款" in user_input or "退货" in user_input:
            with tracer.start_as_current_span("tool.lookup_order") as tool_span:
                tool_span.set_attribute("tool.name", "lookup_order")
                tool_span.set_attribute("tool.input", user_input)
            answer = "已为您查询订单。根据退款政策，签收 7 天内可无理由退款，运费由商家承担。"
        else:
            answer = "您好，这里是订单客服，请问有什么可以帮您？"

        set_token_usage(
            span,
            model="gpt-4o-mini",
            prompt_tokens=120 + len(user_input),
            completion_tokens=len(answer),
        )
    return {"output": answer, "turns": 1}
