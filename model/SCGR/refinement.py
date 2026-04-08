"""Knowledge-Constrained Refinement and final LLM rewrite."""
from __future__ import annotations

from typing import Iterable

import requests

from .concept_align import Constraint, ConstraintSet
from .prompts import REFINE_SYSTEM, REFINE_USER_TEMPLATE


MEDLINEPLUS_BASE = "https://wsearch.nlm.nih.gov/ws/query"


class KnowledgeConstrainedRefiner:
    """MedlinePlus draft + LLM refinement under constraints."""

    def __init__(
        self,
        llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 160,
    ):
        self.llm_model = llm_model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._llm = None
        self._tok = None

    # ---- MedlinePlus dictionary ----------------------------------------
    @staticmethod
    def medlineplus_lookup(term: str) -> str | None:
        """Return a short patient-friendly definition for a clinical term."""
        params = {"db": "healthTopics", "term": term, "retmax": 1}
        try:
            r = requests.get(MEDLINEPLUS_BASE, params=params, timeout=10)
            if r.status_code != 200:
                return None
        except requests.RequestException:
            return None
        # Strip the XML envelope to a one-line lay phrase.
        text = r.text
        start = text.find("<FullSummary>")
        end = text.find("</FullSummary>")
        if start == -1 or end == -1:
            return None
        return text[start + len("<FullSummary>") : end].strip()

    # ---- Stage 2: deterministic draft ----------------------------------
    def make_draft(self, expert: str, constraints: ConstraintSet) -> str:
        # Hydrate lay phrases for each constraint when missing.
        for c in constraints.all:
            if c.lay is None:
                c.lay = self.medlineplus_lookup(c.surface) or c.surface

        # Dictionary-based substitution. Longer surfaces first to avoid
        # partial overshadowing ("CT scan" before "CT").
        draft = expert
        for c in sorted(constraints.all, key=lambda c: -len(c.surface)):
            if c.surface and c.lay and c.surface in draft:
                draft = draft.replace(c.surface, c.lay)
        return draft

    # ---- Stage 3: LLM refinement ---------------------------------------
    def _ensure_llm(self):
        if self._llm is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        import torch
        self._tok = AutoTokenizer.from_pretrained(self.llm_model)
        self._llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device or "auto",
        )

    def refine(
        self,
        expert: str,
        constraints: ConstraintSet,
        draft: str,
    ) -> str:
        self._ensure_llm()
        constraint_block = "\n".join(
            f"- {c.surface} -> {c.lay}" for c in constraints.all if c.lay
        )
        user = REFINE_USER_TEMPLATE.format(
            expert=expert,
            constraints=constraint_block,
            draft=draft,
        )
        messages = [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._llm.device)
        out = self._llm.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        gen = self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return gen.strip()
