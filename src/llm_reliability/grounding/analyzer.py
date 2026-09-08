"""Grounding and answer faithfulness analysis engine."""

import re

from llm_reliability.grounding.enums import SupportStatus
from llm_reliability.grounding.models import ClaimSupport, GroundingMetrics
from llm_reliability.models.execution import RetrievedDocument
from llm_reliability.models.trace import Trace

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "of",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "their",
    "we",
    "us",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "which",
    "who",
    "whom",
    "as",
    "if",
}

_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "cannot",
    "cant",
    "wont",
    "failed",
    "denied",
    "rejected",
    "disabled",
    "prevented",
    "unable",
    "without",
}

_AFFIRMATION_WORDS = {
    "success",
    "succeeded",
    "successful",
    "passed",
    "enabled",
    "allowed",
    "approved",
    "completed",
}


class GroundingAnalyzer:
    """Deterministic analyzer evaluating whether an LLM answer is faithful to and supported by retrieved context."""

    def __init__(
        self,
        supported_threshold: float = 0.60,
        partial_threshold: float = 0.30,
        min_claim_length: int = 8,
    ) -> None:
        """Initialize grounding analyzer with configurable thresholds.

        Args:
            supported_threshold: Overlap ratio required to classify a claim as SUPPORTED.
            partial_threshold: Overlap ratio required to classify a claim as PARTIALLY_SUPPORTED.
            min_claim_length: Minimum character length for a segment to be evaluated as a claim.
        """
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold
        self.min_claim_length = min_claim_length

    def analyze_response(
        self,
        response_text: str | None,
        context: str | list[str] | list[RetrievedDocument] | None,
    ) -> GroundingMetrics:
        """Analyze factual support of a generated response against supplied context documents."""
        clean_response = (response_text or "").strip()
        context_corpus, context_snippets = self._normalize_context(context)

        if not clean_response:
            return GroundingMetrics(
                status=SupportStatus.SUPPORTED,
                grounding_score=1.0,
                confidence=1.0,
                total_claims=0,
            )

        if not context_corpus:
            claims = self._split_into_claims(clean_response)
            claim_supports = [
                ClaimSupport(
                    claim_text=c,
                    status=SupportStatus.UNSUPPORTED,
                    confidence=1.0,
                    overlap_score=0.0,
                    missing_entities=self._extract_entities(c),
                )
                for c in claims
            ]
            all_entities = self._extract_entities(clean_response)
            return GroundingMetrics(
                status=SupportStatus.UNSUPPORTED,
                grounding_score=0.0,
                confidence=1.0,
                total_claims=len(claims),
                supported_claims=0,
                unsupported_claims=len(claims),
                contradicted_claims=0,
                claims=claim_supports,
                hallucinated_entities=all_entities,
            )

        # Segment response into individual claims / sentences
        raw_claims = self._split_into_claims(clean_response)
        if not raw_claims:
            raw_claims = [clean_response]

        evaluated_claims: list[ClaimSupport] = []
        supported_count = 0
        partial_count = 0
        unsupported_count = 0
        contradicted_count = 0
        all_hallucinated_entities: list[str] = []

        context_entities = set(self._extract_entities(context_corpus))

        for claim_text in raw_claims:
            claim_tokens = self._tokenize(claim_text)
            claim_entities = self._extract_entities(claim_text)
            missing_entities = [e for e in claim_entities if e not in context_entities]

            # Best matching snippet and overlap
            best_snippet, overlap_score = self._find_best_matching_snippet(
                claim_tokens, context_snippets
            )

            # Check contradiction
            is_contradiction, contradiction_reason = self._check_contradiction(
                claim_text, best_snippet or context_corpus
            )

            if is_contradiction:
                status = SupportStatus.CONTRADICTED
                contradicted_count += 1
            elif overlap_score >= self.supported_threshold and len(missing_entities) == 0:
                status = SupportStatus.SUPPORTED
                supported_count += 1
            elif overlap_score >= self.partial_threshold or (
                overlap_score > 0.15 and not missing_entities
            ):
                status = SupportStatus.PARTIALLY_SUPPORTED
                partial_count += 1
            else:
                status = SupportStatus.UNSUPPORTED
                unsupported_count += 1

            if status in (
                SupportStatus.UNSUPPORTED,
                SupportStatus.PARTIALLY_SUPPORTED,
                SupportStatus.CONTRADICTED,
            ):
                all_hallucinated_entities.extend(missing_entities)

            evaluated_claims.append(
                ClaimSupport(
                    claim_text=claim_text,
                    status=status,
                    confidence=0.90 if is_contradiction else 0.85,
                    overlap_score=round(overlap_score, 4),
                    matching_context_snippet=best_snippet,
                    contradiction_reason=contradiction_reason,
                    missing_entities=missing_entities,
                )
            )

        total_claims = len(evaluated_claims)
        # Compute continuous grounding score
        if total_claims > 0:
            raw_score = (
                (supported_count * 1.0)
                + (partial_count * 0.5)
                + (unsupported_count * 0.0)
                - (contradicted_count * 0.5)
            ) / total_claims
            grounding_score = round(max(0.0, min(1.0, raw_score)), 4)
        else:
            grounding_score = 1.0

        # Assign aggregate status
        if contradicted_count > 0:
            overall_status = SupportStatus.CONTRADICTED
        elif grounding_score >= 0.70:
            overall_status = SupportStatus.SUPPORTED
        elif grounding_score >= 0.30:
            overall_status = SupportStatus.PARTIALLY_SUPPORTED
        else:
            overall_status = SupportStatus.UNSUPPORTED

        # Deduplicate hallucinated entities preserving order
        unique_hallucinated: list[str] = []
        for entity in all_hallucinated_entities:
            if entity not in unique_hallucinated:
                unique_hallucinated.append(entity)

        return GroundingMetrics(
            status=overall_status,
            grounding_score=grounding_score,
            confidence=0.90 if contradicted_count > 0 else 0.85,
            total_claims=total_claims,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            contradicted_claims=contradicted_count,
            claims=evaluated_claims,
            hallucinated_entities=unique_hallucinated,
            evidence_snippets=context_snippets[:5],
        )

    def analyze_trace(self, trace: Trace) -> list[GroundingMetrics]:
        """Evaluate grounding across all runs and synthesized responses in a trace."""
        results: list[GroundingMetrics] = []
        for run in trace.runs:
            retrieved_docs: list[RetrievedDocument] = []
            for step in run.get_retrievals():
                retrieved_docs.extend(step.documents)

            response_text = run.final_response.text if run.final_response else None

            if not response_text:
                llm_calls = run.get_llm_calls()
                if llm_calls and llm_calls[-1].response:
                    response_text = llm_calls[-1].response

            if response_text is not None:
                results.append(self.analyze_response(response_text, retrieved_docs))

        return results

    def _normalize_context(
        self,
        context: str | list[str] | list[RetrievedDocument] | None,
    ) -> tuple[str, list[str]]:
        """Normalize various context representations into unified text and snippet list."""
        if context is None:
            return "", []

        if isinstance(context, str):
            clean = context.strip()
            snippets = [
                s.strip() for s in re.split(r"(?<=[.!?\n])\s+", clean) if len(s.strip()) > 5
            ]
            return clean, snippets

        snippets_list: list[str] = []
        for item in context:
            if isinstance(item, RetrievedDocument):
                if item.content and item.content.strip():
                    snippets_list.append(item.content.strip())
            elif isinstance(item, str):
                if item.strip():
                    snippets_list.append(item.strip())

        full_text = "\n\n".join(snippets_list)
        return full_text, snippets_list

    def _split_into_claims(self, text: str) -> list[str]:
        """Segment text into individual claim sentences."""
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        claims = [s.strip() for s in sentences if len(s.strip()) >= self.min_claim_length]
        return claims if claims else ([text.strip()] if text.strip() else [])

    def _tokenize(self, text: str) -> set[str]:
        """Extract meaningful normalized content words and n-grams."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\$%\.]+\b", text.lower())
        meaningful = {w for w in words if w not in _STOPWORDS and len(w) > 1}
        bigrams = {f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)}
        return meaningful | bigrams

    def _extract_entities(self, text: str) -> list[str]:
        """Extract numbers, currency, dates, percentages, and capitalized named entities."""
        entities: list[str] = []
        numbers = re.findall(r"\d+(?:[,\.]\d+)?", text)
        entities.extend(numbers)

        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word[0].isupper() and clean_word.lower() not in _STOPWORDS:
                if i > 0:
                    entities.append(clean_word.lower())

        return list(dict.fromkeys(entities))

    def _find_best_matching_snippet(
        self,
        claim_tokens: set[str],
        snippets: list[str],
    ) -> tuple[str | None, float]:
        """Find the snippet with highest lexical overlap with the claim tokens."""
        if not claim_tokens or not snippets:
            return None, 0.0

        best_snippet: str | None = None
        best_overlap = 0.0

        for snippet in snippets:
            snippet_tokens = self._tokenize(snippet)
            if not snippet_tokens:
                continue
            intersection = len(claim_tokens & snippet_tokens)
            recall = intersection / len(claim_tokens)
            if recall > best_overlap:
                best_overlap = recall
                best_snippet = snippet

        return best_snippet, best_overlap

    def _check_contradiction(
        self,
        claim_text: str,
        context_snippet: str,
    ) -> tuple[bool, str | None]:
        """Detect explicit polarity or numeric contradictions between claim and context snippet."""
        claim_lower = claim_text.lower()
        context_lower = context_snippet.lower()

        claim_has_negation = any(w in claim_lower.split() for w in _NEGATION_WORDS)
        context_has_negation = any(w in context_lower.split() for w in _NEGATION_WORDS)

        claim_has_affirm = any(w in claim_lower.split() for w in _AFFIRMATION_WORDS)
        context_has_affirm = any(w in context_lower.split() for w in _AFFIRMATION_WORDS)

        claim_words = (
            set(re.findall(r"\w+", claim_lower)) - _STOPWORDS - _NEGATION_WORDS - _AFFIRMATION_WORDS
        )
        context_words = (
            set(re.findall(r"\w+", context_lower))
            - _STOPWORDS
            - _NEGATION_WORDS
            - _AFFIRMATION_WORDS
        )
        common_subjects = claim_words & context_words

        if len(common_subjects) >= 2:
            if (claim_has_negation and context_has_affirm and not context_has_negation) or (
                claim_has_affirm and context_has_negation and not claim_has_negation
            ):
                return (
                    True,
                    f"Polarity conflict on subject ({', '.join(list(common_subjects)[:3])})",
                )

        # Numeric conflict: same entity subject with differing numbers
        claim_numbers = re.findall(r"\d+(?:[\.,]\d+)?", claim_text)
        context_numbers = re.findall(r"\d+(?:[\.,]\d+)?", context_snippet)

        if len(common_subjects) >= 2 and claim_numbers and context_numbers:
            if set(claim_numbers) != set(context_numbers) and not (
                set(claim_numbers) & set(context_numbers)
            ):
                return (
                    True,
                    f"Numeric conflict: answer has {claim_numbers} while context has {context_numbers}",
                )

        return False, None
