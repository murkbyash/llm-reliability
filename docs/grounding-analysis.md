# Grounding & Faithfulness Analysis

`llm_reliability.grounding.GroundingAnalyzer` provides sentence-level claim evaluation to identify answer hallucinations, fabricated entities, and factual contradictions relative to the provided retrieval context.

---

## 1. Grounding Evaluation Technique

The analyzer splits the generated answer into individual propositional sentences and evaluates each claim:

1. **Sentence-Level Claim Extraction:**
   - Splits answer on punctuation boundaries while preserving numerical and quote integrity.
2. **Lexical & Semantic Overlap (`SupportStatus.SUPPORTED` vs `UNSUPPORTED`):**
   - Measures token overlap, longest common subsequence, and lexical density against context documents.
   - Claims with overlap score $\ge 0.40$ are categorized as `SUPPORTED`.
   - Claims below $0.40$ are categorized as `UNSUPPORTED`.
3. **Contradiction Detection (`SupportStatus.CONTRADICTED`):**
   - Identifies polar contradictions (e.g. context: *"Mars gravity is 3.7 m/s²"*, answer: *"Mars has no gravity"*).
   - Tracks numerical discrepancies and negation flips.
4. **Entity Hallucination Tracking:**
   - Detects named entities (proper nouns, capitalized terms, technical identifiers) present in the answer but completely absent from the source context.

---

## 2. Grounding Metrics Model

- `total_claims`: Total sentences extracted.
- `supported_claims_count`: Number of verified claims.
- `unsupported_claims_count`: Number of claims lacking source support.
- `contradicted_claims_count`: Number of explicitly conflicting claims.
- `grounding_score`: Overall support ratio ($0.0 \dots 1.0$).
- `hallucinated_entities_count`: Count of fabricated named entities.
- `claims`: List of `ClaimSupport` objects detailing each sentence, support status, overlap score, and matched document IDs.

---

## 3. Python Example

```python
from llm_reliability.grounding import GroundingAnalyzer
from llm_reliability.models.execution import RetrievedDocument

context_docs = [
    RetrievedDocument(
        doc_id="d1",
        content="The James Webb Space Telescope was launched on December 25, 2021 from French Guiana.",
    )
]

generated_answer = (
    "The James Webb Space Telescope launched on December 25, 2021. "
    "It was manufactured entirely by SpaceX in Texas."  # Hallucination
)

analyzer = GroundingAnalyzer()
metrics = analyzer.analyze(generated_answer, context_docs)

print(f"Grounding Score: {metrics.grounding_score:.2f}")
print(f"Unsupported Claims: {metrics.unsupported_claims_count}")
for claim in metrics.claims:
    print(f"[{claim.status.value}] {claim.claim_text} (Overlap: {claim.support_score:.2f})")
```

