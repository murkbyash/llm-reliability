"""Developer recommendation engine producing prioritized remediation advice."""

from llm_reliability.models.diagnosis import Diagnosis, Recommendation
from llm_reliability.recommendations.rules import (
    generate_agent_recommendations,
    generate_grounding_recommendations,
    generate_llm_recommendations,
    generate_rag_recommendations,
)


class RecommendationEngine:
    """Deterministic recommendation engine mapping diagnostic findings to actionable developer fixes."""

    def generate_recommendations(self, diagnosis: Diagnosis) -> list[Recommendation]:
        """Generate prioritized, actionable developer recommendations from a complete Diagnosis.

        Args:
            diagnosis: The synthesized Diagnosis object containing failures, hypotheses, metrics, and evidence.

        Returns:
            Ranked list of Recommendation objects sorted by priority.
        """
        all_recs: list[Recommendation] = []

        # Collect candidate recommendations across all domains
        all_recs.extend(generate_rag_recommendations(diagnosis))
        all_recs.extend(generate_grounding_recommendations(diagnosis))
        all_recs.extend(generate_agent_recommendations(diagnosis))
        all_recs.extend(generate_llm_recommendations(diagnosis))

        # Deduplicate recommendations by title, preserving highest priority
        unique_map: dict[str, Recommendation] = {}
        for rec in all_recs:
            if rec.title not in unique_map or rec.priority < unique_map[rec.title].priority:
                unique_map[rec.title] = rec

        # Sort by priority ascending (1 is highest priority)
        sorted_recs = sorted(unique_map.values(), key=lambda r: r.priority)
        return sorted_recs

    def recommend(self, diagnosis: Diagnosis) -> list[Recommendation]:
        """Alias for generate_recommendations."""
        return self.generate_recommendations(diagnosis)
