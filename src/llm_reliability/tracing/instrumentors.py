"""SDK auto-instrumentation hooks and client wrappers for OpenAI, Anthropic, and tool functions."""

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

from llm_reliability.models.enums import SpanKind, SpanStatus
from llm_reliability.models.execution import LLMCall, TokenUsage
from llm_reliability.tracing.tracer import get_tracer

F = TypeVar("F", bound=Callable[..., Any])


def wrap_openai(client: Any) -> Any:
    """Wrap an OpenAI client instance to automatically record LLM completion spans."""
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        return client

    orig_create = client.chat.completions.create
    tracer = get_tracer()

    @functools.wraps(orig_create)
    def traced_create(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", "openai_model"))
        messages = kwargs.get("messages", [])
        prompt_repr = str(messages)

        with tracer.start_span(name=f"openai:{model}", kind=SpanKind.LLM) as active:
            active.llm_call = LLMCall(model=model, prompt=prompt_repr)
            try:
                response = orig_create(*args, **kwargs)
                # Extract completion text
                if hasattr(response, "choices") and response.choices:
                    first_choice = response.choices[0]
                    if hasattr(first_choice, "message") and hasattr(
                        first_choice.message, "content"
                    ):
                        active.llm_call.response = str(first_choice.message.content)
                    elif isinstance(first_choice, dict) and "message" in first_choice:
                        active.llm_call.response = str(first_choice["message"].get("content", ""))

                # Extract token usage
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    p_tok = getattr(usage, "prompt_tokens", 0)
                    c_tok = getattr(usage, "completion_tokens", 0)
                    t_tok = getattr(usage, "total_tokens", p_tok + c_tok)
                    active.llm_call.token_usage = TokenUsage(
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=t_tok,
                    )
                elif isinstance(response, dict) and "usage" in response:
                    u = response["usage"]
                    p_tok = u.get("prompt_tokens", 0)
                    c_tok = u.get("completion_tokens", 0)
                    active.llm_call.token_usage = TokenUsage(
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=p_tok + c_tok,
                    )

                return response
            except Exception as err:
                active.status = SpanStatus.ERROR
                active.error_message = str(err)
                raise

    client.chat.completions.create = traced_create
    return client


def wrap_anthropic(client: Any) -> Any:
    """Wrap an Anthropic client instance to automatically record LLM message spans."""
    if not hasattr(client, "messages") or not hasattr(client.messages, "create"):
        return client

    orig_create = client.messages.create
    tracer = get_tracer()

    @functools.wraps(orig_create)
    def traced_create(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", "claude_model"))
        messages = kwargs.get("messages", [])
        prompt_repr = str(messages)

        with tracer.start_span(name=f"anthropic:{model}", kind=SpanKind.LLM) as active:
            active.llm_call = LLMCall(model=model, prompt=prompt_repr)
            try:
                response = orig_create(*args, **kwargs)
                # Extract completion text
                if hasattr(response, "content") and response.content:
                    first_block = response.content[0]
                    if hasattr(first_block, "text"):
                        active.llm_call.response = str(first_block.text)
                    elif isinstance(first_block, dict) and "text" in first_block:
                        active.llm_call.response = str(first_block["text"])

                # Extract token usage
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    p_tok = getattr(usage, "input_tokens", 0)
                    c_tok = getattr(usage, "output_tokens", 0)
                    active.llm_call.token_usage = TokenUsage(
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=p_tok + c_tok,
                    )
                return response
            except Exception as err:
                active.status = SpanStatus.ERROR
                active.error_message = str(err)
                raise

    client.messages.create = traced_create
    return client


def wrap_tool(func: F, tool_name: str | None = None) -> F:
    """Wrap a tool execution function with automatic tool span emission."""
    t_name = str(tool_name or getattr(func, "__name__", "tool_func"))
    tracer = get_tracer()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        args_payload = kwargs if kwargs else ({"args": list(args)} if args else {})
        with tracer.trace_tool(tool_name=t_name, arguments=args_payload) as active:
            try:
                result = func(*args, **kwargs)
                if active.tool_result is not None:
                    active.tool_result.output = result
                    active.tool_result.status = SpanStatus.SUCCESS
                return result
            except Exception as err:
                if active.tool_result is not None:
                    active.tool_result.status = SpanStatus.ERROR
                    active.tool_result.error = str(err)
                raise

    return cast(F, wrapper)
