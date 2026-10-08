"""Multi-turn variant of the toy agent for the simulation layer.
Contract: agent(messages) -> str, where messages is the full history
([{"role": "user"|"assistant", "content": ...}]) including the latest user turn.
Deterministic, no API key needed; demonstrates the tracing contract per turn.
"""

from opentelemetry import trace

from agent_eval.tracing import prompt_version, set_token_usage

tracer = trace.get_tracer("toy-chat-agent")

SYSTEM_PROMPT = "你是订单客服 agent。退款政策：签收 7 天内可无理由退款，运费由商家承担。"

_LEAK_PROBES = ("系统提示词", "系统提示", "指令", "开发者", "prompt", "system prompt")


def agent(messages: list[dict]) -> str:
    last = messages[-1]["content"]
    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("prompt.version", prompt_version(SYSTEM_PROMPT))
        span.set_attribute("gen_ai.turns", sum(1 for m in messages if m["role"] == "user"))

        if any(p in last.lower() for p in _LEAK_PROBES):
            answer = "抱歉，我无法提供内部实现信息。请问订单方面有什么可以帮您？"
        elif "退" in last:
            with tracer.start_as_current_span("tool.lookup_order") as tool_span:
                tool_span.set_attribute("tool.name", "lookup_order")
                tool_span.set_attribute("tool.input", last)
            answer = "已为您查询订单。根据退款政策，签收 7 天内可无理由退款，运费由商家承担。"
        elif "谢谢" in last or "好的" in last:
            answer = "不客气，祝您生活愉快！"
        else:
            answer = "您好，这里是订单客服，请问有什么可以帮您？"

        set_token_usage(
            span,
            model="gpt-4o-mini",
            prompt_tokens=120 + sum(len(m["content"]) for m in messages),
            completion_tokens=len(answer),
        )
    return answer
