"""End-to-end SCGR pipeline orchestrating the three stages."""
from __future__ import annotations

from typing import Iterable

from .concept_align import ConceptAligner, ConstraintSet
from .refinement import KnowledgeConstrainedRefiner


class SCGRPipeline:
    """Structured Concept-Grounded Refinement (SCGR).

    Stage 1 -- Concept-Knowledge Alignment (UMLS + NER)
    Stage 2 -- Knowledge-Constrained Refinement (MedlinePlus draft)
    Stage 3 -- LLM Refinement (Llama-3.1-8B, fluency under constraints)
    """

    def __init__(
        self,
        umls_api_key: str | None = None,
        llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        scispacy_model: str = "en_core_sci_md",
        device: str | None = None,
    ):
        self.aligner = ConceptAligner(
            umls_api_key=umls_api_key,
            scispacy_model=scispacy_model,
        )
        self.refiner = KnowledgeConstrainedRefiner(
            llm_model=llm_model,
            device=device,
        )

    def refine(self, expert: str, cuis: Iterable[str]) -> str:
        """Return a lay caption for an expert caption + its UMLS CUIs."""
        constraints: ConstraintSet = self.aligner(expert, cuis)
        draft = self.refiner.make_draft(expert, constraints)
        return self.refiner.refine(expert, constraints, draft)

    def refine_batch(
        self,
        expert_list: list[str],
        cui_list: list[list[str]],
    ) -> list[str]:
        return [self.refine(e, c) for e, c in zip(expert_list, cui_list)]
