"""Tests for Tracing Middleware & SDK Hooks."""

from unittest.mock import MagicMock

from llm_reliability import (
    SpanKind,
    SpanStatus,
    Trace,
    Tracer,
    diagnose,
    get_tracer,
    trace_llm,
    trace_retrieval,
    trace_tool,
    wrap_anthropic,
    wrap_openai,
)


class TestTracingSDK:
    """Deterministic validation of live tracing middleware, context managers, decorators, and instrumentors."""

    def test_context_manager_trace_and_span_capture(self) -> None:
        tracer = Tracer()

        with tracer.start_trace(trace_id="trace-live-001", run_id="run-live-001") as t_mgr:
            with tracer.start_span(name="root_pipeline", kind=SpanKind.CHAIN) as root_span:
                assert root_span.span_id != ""

                with tracer.trace_retrieval(query="vector databases") as ret_span:
                    assert ret_span.kind == SpanKind.RETRIEVAL
                    assert ret_span.parent_span_id == root_span.span_id
                    assert ret_span.retrieval is not None
                    assert ret_span.retrieval.query == "vector databases"

                with tracer.trace_llm(
                    model="gpt-4o", prompt="Summarize vector databases"
                ) as llm_span:
                    assert llm_span.kind == SpanKind.LLM
                    assert llm_span.parent_span_id == root_span.span_id
                    assert llm_span.llm_call is not None
                    assert llm_span.llm_call.model == "gpt-4o"
                    llm_span.llm_call.response = (
                        "Vector databases index embeddings for similarity search."
                    )

            t_mgr.set_final_response("Vector databases index embeddings for similarity search.")

        trace = t_mgr.trace
        assert isinstance(trace, Trace)
        assert trace.trace_id == "trace-live-001"
        assert len(trace.runs) == 1
        assert len(trace.runs[0].spans) == 3

        kinds = [s.kind for s in trace.runs[0].spans]
        assert SpanKind.RETRIEVAL in kinds
        assert SpanKind.LLM in kinds
        assert SpanKind.CHAIN in kinds

        assert trace.runs[0].final_response is not None
        assert (
            trace.runs[0].final_response.text
            == "Vector databases index embeddings for similarity search."
        )

    def test_decorators_trace_llm_tool_retrieval(self) -> None:
        tracer = get_tracer()

        @trace_retrieval(query_param="query")
        def search_knowledge(query: str) -> list[dict[str, str]]:
            return [{"id": "doc-1", "content": "Knowledge content", "score": "0.95"}]

        @trace_tool(tool_name="calculator")
        def calc(a: int, b: int) -> int:
            return a + b

        @trace_llm(model="claude-3-5-sonnet")
        def generate_answer(prompt: str) -> str:
            return f"Answer to: {prompt}"

        with tracer.start_trace(trace_id="trace-dec-001") as t_mgr:
            docs = search_knowledge("test query")
            assert len(docs) == 1

            total = calc(10, 20)
            assert total == 30

            ans = generate_answer("What is 10 + 20?")
            assert "30" in ans or "Answer to:" in ans

        trace = t_mgr.trace
        assert isinstance(trace, Trace)
        assert len(trace.runs[0].spans) == 3

        ret_span = next(s for s in trace.runs[0].spans if s.kind == SpanKind.RETRIEVAL)
        assert ret_span.retrieval is not None
        assert ret_span.retrieval.query == "test query"
        assert len(ret_span.retrieval.documents) == 1

        tool_span = next(s for s in trace.runs[0].spans if s.kind == SpanKind.TOOL)
        assert tool_span.tool_call is not None
        assert tool_span.tool_call.tool_name == "calculator"
        assert tool_span.tool_result is not None
        assert tool_span.tool_result.output == 30

    def test_span_exception_recording(self) -> None:
        tracer = Tracer()

        with tracer.start_trace(trace_id="trace-err-001") as t_mgr:
            try:
                with tracer.start_span(name="failing_service", kind=SpanKind.TOOL):
                    raise RuntimeError("Database connection timed out")
            except RuntimeError:
                pass

        trace = t_mgr.trace
        assert isinstance(trace, Trace)
        assert len(trace.runs[0].spans) == 1
        err_span = trace.runs[0].spans[0]
        assert err_span.status == SpanStatus.ERROR
        assert err_span.error_message is not None
        assert "Database connection timed out" in err_span.error_message

    def test_openai_client_wrapper(self) -> None:
        tracer = get_tracer()

        # Create mock OpenAI client
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Paris is the capital of France."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 12
        mock_usage.completion_tokens = 8
        mock_usage.total_tokens = 20

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client.chat.completions.create.return_value = mock_response

        # Wrap client
        wrapped_client = wrap_openai(mock_client)

        with tracer.start_trace(trace_id="trace-openai-001") as t_mgr:
            resp = wrapped_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "What is the capital of France?"}],
            )
            assert resp.choices[0].message.content == "Paris is the capital of France."

        trace = t_mgr.trace
        assert isinstance(trace, Trace)
        assert len(trace.runs[0].spans) == 1

        llm_span = trace.runs[0].spans[0]
        assert llm_span.kind == SpanKind.LLM
        assert llm_span.llm_call is not None
        assert llm_span.llm_call.model == "gpt-4o"
        assert llm_span.llm_call.response == "Paris is the capital of France."
        assert llm_span.llm_call.token_usage is not None
        assert llm_span.llm_call.token_usage.total_tokens == 20

    def test_anthropic_client_wrapper(self) -> None:
        tracer = get_tracer()

        # Create mock Anthropic client
        mock_client = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Mars has two moons: Phobos and Deimos."
        mock_usage = MagicMock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 10

        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_response.usage = mock_usage

        mock_client.messages.create.return_value = mock_response

        wrapped_client = wrap_anthropic(mock_client)

        with tracer.start_trace(trace_id="trace-anthropic-001") as t_mgr:
            resp = wrapped_client.messages.create(
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "How many moons does Mars have?"}],
            )
            assert resp.content[0].text == "Mars has two moons: Phobos and Deimos."

        trace = t_mgr.trace
        assert isinstance(trace, Trace)
        assert len(trace.runs[0].spans) == 1
        llm_span = trace.runs[0].spans[0]
        assert llm_span.llm_call is not None
        assert llm_span.llm_call.model == "claude-3-5-sonnet"
        assert llm_span.llm_call.token_usage is not None
        assert llm_span.llm_call.token_usage.total_tokens == 25

    def test_live_trace_end_to_end_diagnose(self) -> None:
        tracer = Tracer()

        with tracer.start_trace(trace_id="trace-live-diag") as t_mgr:
            with tracer.trace_retrieval(query="quantum mechanics") as ret:
                assert ret.retrieval is not None
                ret.retrieval.documents = []  # Injected empty retrieval
            with tracer.trace_llm(model="gpt-4o", prompt="Explain quantum physics"):
                pass

        trace = t_mgr.trace
        assert isinstance(trace, Trace)

        # Run diagnosis directly on live captured trace
        diagnosis = diagnose(trace)
        assert diagnosis.trace_id == "trace-live-diag"
        assert len(diagnosis.failures) >= 1
        assert diagnosis.primary_category.value == "RETRIEVAL_FAILURE"
