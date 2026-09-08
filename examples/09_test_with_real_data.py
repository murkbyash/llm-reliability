"""Example 09: Testing LMM Reliability Analyzer with Real Production Data."""

from pathlib import Path

from llm_reliability import (
    DiagnosticEngine,
    RecommendationEngine,
    Tracer,
    load_trace,
    save_html_report,
)
from llm_reliability.models.execution import RetrievedDocument


def demo_method_1_real_json_logs() -> None:
    """Method 1: Ingesting real production JSON logs."""
    print("\n" + "=" * 70)
    print("METHOD 1: Ingesting Real RAG / Agent JSON Logs")
    print("=" * 70)

    # Imagine this is real logged telemetry from your application
    real_rag_log = {
        "trace_id": "real-prod-trace-101",
        "runs": [
            {
                "run_id": "run-real-01",
                "trace_id": "real-prod-trace-101",
                "spans": [
                    {
                        "span_id": "span-ret-01",
                        "name": "vector_search",
                        "kind": "RETRIEVAL",
                        "status": "SUCCESS",
                        "retrieval": {
                            "query": "What are the cancellation terms for Enterprise plans?",
                            "documents": [
                                {
                                    "doc_id": "doc-legal-01",
                                    "content": "Enterprise plans require a 30-day written notice for cancellation and are subject to annual commitments.",
                                    "score": 0.88,
                                }
                            ],
                        },
                    },
                    {
                        "span_id": "span-llm-02",
                        "name": "gpt4_completion",
                        "kind": "LLM",
                        "status": "SUCCESS",
                        "llm_call": {
                            "model": "gpt-4o",
                            "prompt": "Answer the user question using the provided legal policy.",
                            "response": "Enterprise plans can be canceled instantly with a full refund within 90 days with no notice required.",
                            "token_usage": {
                                "prompt_tokens": 120,
                                "completion_tokens": 25,
                                "total_tokens": 145,
                            },
                        },
                    },
                ],
                "final_response": {
                    "text": "Enterprise plans can be canceled instantly with a full refund within 90 days with no notice required."
                },
            }
        ],
    }

    # Load and normalize real JSON directly
    trace = load_trace(real_rag_log)
    engine = DiagnosticEngine()
    diagnosis = engine.diagnose(trace)

    print(f"[+] Diagnosed Real Trace: {trace.trace_id}")
    print(f"    Primary Root Cause: {diagnosis.primary_category.value}")
    print(f"    Severity          : {diagnosis.severity.value.upper()}")
    print(f"    Summary           : {diagnosis.summary}")

    # Prioritized developer fix recommendations
    recs = RecommendationEngine().recommend(diagnosis)
    print("\n[+] Actionable Recommendations for Engineering Team:")
    for r in recs:
        print(f"    * [P{r.priority}] {r.title} ({r.action_type})")


def demo_method_2_live_python_tracing() -> None:
    """Method 2: Instrumenting your actual Python functions in real-time."""
    print("\n" + "=" * 70)
    print("METHOD 2: Live Programmatic Tracing of Real Python Execution")
    print("=" * 70)

    tracer = Tracer()

    # Suppose you are running real retrieval from your database / Elasticsearch / Chroma / Pinecone
    real_user_query = "What is the capital of France?"

    with tracer.start_trace(trace_id="real-live-trace-202") as active_trace:
        # Step 1: Trace document retrieval
        with tracer.trace_retrieval(query=real_user_query) as ret_ctx:
            # Here you call your real search / embedding engine
            if ret_ctx.retrieval is not None:
                ret_ctx.retrieval.documents = [
                    RetrievedDocument(
                        doc_id="doc-geo-1",
                        content="Paris is the capital and most populous city of France.",
                        score=0.97,
                    )
                ]

        # Step 2: Trace LLM generation
        with tracer.trace_llm(model="claude-3-5-sonnet", prompt=real_user_query) as llm_ctx:
            # Here you receive the actual LLM output
            real_model_output = "The capital of France is Paris."
            if llm_ctx.llm_call is not None:
                llm_ctx.llm_call.response = real_model_output

        active_trace.set_final_response(real_model_output)

    # Automatically diagnose the captured real trace
    diagnosis = DiagnosticEngine().diagnose(active_trace.trace)
    print(f"[+] Live Trace Captured: {active_trace.trace_id}")
    print(f"    Failure Status : {diagnosis.primary_category.value}")
    print(f"    Severity       : {diagnosis.severity.value.upper()}")


def demo_method_3_generate_visual_html() -> None:
    """Method 3: Exporting real diagnostic report to standalone HTML."""
    print("\n" + "=" * 70)
    print("METHOD 3: Saving Real Diagnostic Report as Offline HTML Dashboard")
    print("=" * 70)

    from llm_reliability.models.enums import SpanStatus

    tracer = Tracer()
    with tracer.start_trace(trace_id="real-production-incident-909") as t:
        with tracer.trace_tool(
            tool_name="database_query", arguments={"table": "users", "query": "SELECT *"}
        ) as tool_ctx:
            tool_ctx.status = SpanStatus.ERROR
            tool_ctx.error_message = "OperationalError: Connection pool exhausted (timeout=5000ms)"
        t.set_final_response("Database query failed due to pool exhaustion.")

    diagnosis = DiagnosticEngine().diagnose(t.trace)
    report_file = Path("real_data_diagnostic_report.html")
    save_html_report(diagnosis, report_file, trace=t.trace, title="Real Production Incident 909")
    print(f"[+] Saved Real Data HTML Report to: {report_file.resolve()}")
    print("=" * 70 + "\n")


def main() -> None:
    print("*" * 70)
    print("  LLM RELIABILITY ANALYZER - REAL DATA TESTING SUITE")
    print("*" * 70)
    demo_method_1_real_json_logs()
    demo_method_2_live_python_tracing()
    demo_method_3_generate_visual_html()


if __name__ == "__main__":
    main()
