# Live Tracing SDK & Provider Hooks

The Live Tracing SDK (`llm_reliability.tracing`) allows developers to instrument custom applications, functions, and LLM providers with zero-overhead async and thread-safe tracing hooks.

---

## 1. Zero-Overhead Function Decorators

Instrument functions with lightweight decorators:

```python
from llm_reliability import get_tracer, trace, trace_llm, trace_tool, trace_retrieval

tracer = get_tracer()


@trace_retrieval(query_param="query")
def vector_search(query: str) -> list[dict]:
    # Returns list of dicts or RetrievedDocument models
    return [{"id": "doc-1", "content": "RAG architecture guide", "score": 0.94}]


@trace_tool(tool_name="database_query")
def run_db_query(sql: str) -> dict:
    return {"status": "ok", "rows": 12}


@trace_llm(model="gpt-4o")
def call_model(prompt: str) -> str:
    return "Summary output"


# Execute inside a trace context manager
with tracer.start_trace("user-session-42") as trace_ctx:
    docs = vector_search("RAG guide")
    db_res = run_db_query("SELECT 1")
    response = call_model("Synthesize docs")

# Access canonical Trace object
completed_trace = trace_ctx.trace
```

---

## 2. OpenAI & Anthropic Client Auto-Instrumentation

Wrap standard Python SDK clients to capture spans and token usages transparently:

```python
import openai
import anthropic
from llm_reliability import wrap_openai, wrap_anthropic

# Wrap OpenAI Client
client = wrap_openai(openai.OpenAI())

# Standard completions are automatically traced
response = client.chat.completions.create(
    model="gpt-4o", messages=[{"role": "user", "content": "Explain async Python"}]
)

# Wrap Anthropic Client
anthropic_client = wrap_anthropic(anthropic.Anthropic())
msg = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello Claude"}],
)
```

