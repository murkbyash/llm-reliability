# Cookbook & Common Recipes

Practical, end-to-end implementation recipes for common real-world AI pipelines.

---

## 1. Recipe: Debugging LangChain RAG & Agent Traces

Export your LangChain run tree as JSON and diagnose it with `llm-reliability`:

```python
from langchain.callbacks.tracers import LangChainTracer
from llm_reliability import diagnose

# Extract or export LangSmith run dictionary
langsmith_run_dict = {
    "id": "run-root-1",
    "name": "RetrievalQA",
    "run_type": "chain",
    "inputs": {"query": "What is Python?"},
    "outputs": {"result": "Python is a language."},
    "child_runs": [
        {
            "id": "run-ret-1",
            "name": "VectorRetriever",
            "run_type": "retriever",
            "inputs": {"query": "What is Python?"},
            "outputs": {"documents": []},  # Empty retrieval failure
        }
    ],
}

# Run diagnosis directly on LangSmith dictionary
diagnosis = diagnose(langsmith_run_dict)
print(f"Diagnosed Category: {diagnosis.primary_category.value}")
print(f"Remediation: {diagnosis.recommendations[0].title}")
```

---

## 2. Recipe: FastAPI Endpoint Latency & Failure Middleware

Add automated diagnostic hooks to your FastAPI streaming endpoints:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm_reliability import get_tracer, wrap_stream, diagnose

app = FastAPI()
tracer = get_tracer()


def generate_llm_stream(prompt: str):
    yield "Streamed "
    yield "response."


@app.post("/chat")
async def chat_endpoint(prompt: str):
    with tracer.start_trace("fastapi-chat-run") as trace_ctx:

        def on_complete(profile, text):
            # Check for excessive TTFT or streaming stalls
            if profile.stall_count > 0:
                print(f"Alert: {profile.stall_count} stalls observed in request!")
            diagnosis = diagnose(trace_ctx.trace)

        stream = wrap_stream(generate_llm_stream(prompt), on_complete=on_complete)
        return StreamingResponse(stream, media_type="text/plain")
```

