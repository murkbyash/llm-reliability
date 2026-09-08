# RAG Failure Analysis

Retrieval-Augmented Generation (RAG) pipelines frequently fail due to silent retrieval degradation rather than model reasoning flaws.

---

## 1. Analyzed RAG Failure Modes

`llm_reliability.rag.RAGAnalyzer` inspects candidate documents across four primary vectors:

1. **Empty Retrieval (`RETRIEVAL_FAILURE`):**
   - Zero documents returned by the vector search index or BM25 retriever.
2. **Low Semantic Relevance (`RETRIEVAL_FAILURE`):**
   - Candidate documents fail to satisfy the relevance cutoff ($\text{similarity score} < 0.60$).
   - High score dropoff between Rank 1 and subsequent candidates ($\text{dropoff ratio} > 0.40$).
3. **Duplicate & Redundant Chunks (`CONTEXT_CONSTRUCTION_FAILURE`):**
   - High Jaccard token overlap ($\ge 0.85$) between retrieved chunks causing wasted prompt context and lost needle-in-a-haystack attention.
4. **Context Volume Deficit or Bloat (`CONTEXT_CONSTRUCTION_FAILURE`):**
   - Insufficient context characters ($< 50\text{ characters}$) or excessive unranked token bloat ($> 3000\text{ tokens}$).

---

## 2. RAG Metrics Extracted

The analyzer produces a structured `RetrievalMetrics` model containing:

- `total_candidates`: Number of documents returned.
- `mean_score`, `max_score`, `min_score`: Statistical distribution of vector similarity scores.
- `score_variance`, `score_spread`: Spread and variance across top-k candidates.
- `score_dropoff_ratio`: Ratio difference between first and last candidate.
- `relevance_ratio`: Proportion of candidates exceeding relevance threshold ($0.60$).
- `duplicate_chunk_ratio`: Proportion of pairwise duplicate chunks (Jaccard token similarity $\ge 0.85$).
- `total_context_tokens`: Approximate token volume calculated from context text length.

---

## 3. Python Example

```python
from llm_reliability.rag import RAGAnalyzer
from llm_reliability.models.execution import RetrievalStep, RetrievedDocument

step = RetrievalStep(
    query="quantum computing superposition",
    documents=[
        RetrievedDocument(
            doc_id="d1", content="Quantum computers use qubits in superposition.", score=0.92
        ),
        RetrievedDocument(
            doc_id="d2", content="Quantum computers use qubits in superposition.", score=0.91
        ),  # Duplicate
        RetrievedDocument(
            doc_id="d3", content="Classical computers use standard silicon transistors.", score=0.35
        ),  # Low relevance
    ],
)

analyzer = RAGAnalyzer()
metrics = analyzer.analyze(step)

print(f"Total Candidates: {metrics.total_candidates}")
print(f"Duplicate Ratio: {metrics.duplicate_chunk_ratio:.2f}")
print(f"Relevance Ratio: {metrics.relevance_ratio:.2f}")
print(f"Score Dropoff: {metrics.score_dropoff_ratio:.2f}")
```

