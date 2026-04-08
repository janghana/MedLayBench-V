"""Concept-Knowledge Alignment: UMLS CUI mapping + SciSpacy NER."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import requests


UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"


@dataclass
class Constraint:
    surface: str
    kind: str           # "cui" | "entity"
    cui: str | None = None
    lay: str | None = None


@dataclass
class ConstraintSet:
    onto: list[Constraint] = field(default_factory=list)
    ent:  list[Constraint] = field(default_factory=list)

    @property
    def all(self) -> list[Constraint]:
        return self.onto + self.ent


class ConceptAligner:
    """Build C = C_onto ∪ C_ent from an expert caption."""

    def __init__(
        self,
        umls_api_key: str | None = None,
        scispacy_model: str = "en_core_sci_md",
    ):
        self.umls_api_key = umls_api_key
        self._nlp = None
        self._scispacy_model = scispacy_model

    # ---- C_onto ---------------------------------------------------------
    def cui_to_preferred(self, cui: str) -> str | None:
        """Resolve a CUI to its preferred clinical name via the UMLS REST API."""
        if not self.umls_api_key:
            return None
        url = f"{UMLS_BASE}/content/current/CUI/{cui}"
        r = requests.get(url, params={"apiKey": self.umls_api_key}, timeout=10)
        if r.status_code != 200:
            return None
        return r.json().get("result", {}).get("name")

    def build_onto(self, cuis: Iterable[str]) -> list[Constraint]:
        out = []
        for cui in cuis:
            pref = self.cui_to_preferred(cui)
            out.append(Constraint(surface=pref or cui, kind="cui", cui=cui))
        return out

    # ---- C_ent ----------------------------------------------------------
    def _ensure_nlp(self):
        if self._nlp is None:
            import spacy  # lazy
            self._nlp = spacy.load(self._scispacy_model)
        return self._nlp

    def build_ent(self, expert_text: str) -> list[Constraint]:
        nlp = self._ensure_nlp()
        doc = nlp(expert_text)
        # Quantitative attributes, anatomy, descriptors -- whatever NER caught.
        return [Constraint(surface=ent.text, kind="entity") for ent in doc.ents]

    # ---- C = C_onto ∪ C_ent --------------------------------------------
    def __call__(self, expert_text: str, cuis: Iterable[str]) -> ConstraintSet:
        return ConstraintSet(
            onto=self.build_onto(cuis),
            ent=self.build_ent(expert_text),
        )
