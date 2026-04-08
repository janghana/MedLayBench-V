# SCGR &mdash; Structured Concept-Grounded Refinement

A reference implementation of the **SCGR** pipeline used to construct
MedLayBench-V from ROCOv2.

## Pipeline

```
expert caption  +  CUIs
        |
        v
[1] Concept-Knowledge Alignment   --> C = C_onto ∪ C_ent
        |                              UMLS API + SciSpacy NER
        v
[2] Knowledge-Constrained Refinement --> T_draft
        |                                MedlinePlus dictionary substitution
        v
[3] LLM Refinement (Llama-3.1-8B) --> T_lay
        constrained by C, grounded in T_exp, seeded by T_draft
```

The crucial design choice is that **only stage 3 is generative**. The semantic
content is fully determined by stages 1&ndash;2; the LLM is restricted to
grammar and fluency.

## Files

| File              | Stage | Purpose                                       |
|-------------------|------:|-----------------------------------------------|
| `concept_align.py`|     1 | UMLS CUI lookup + SciSpacy NER -> `ConstraintSet` |
| `refinement.py`   |   2,3 | MedlinePlus draft + Llama-3.1 refinement         |
| `prompts.py`      |     3 | Constrained prompt templates                     |
| `pipeline.py`     | 1-3   | `SCGRPipeline.refine(expert, cuis) -> lay`       |

## Quick start

```python
from model.SCGR import SCGRPipeline

scgr = SCGRPipeline(
    umls_api_key="YOUR_UMLS_KEY",
    llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
)

expert = "Thoracic CT scan showing perihilar lymphadenomegaly."
lay = scgr.refine(expert, cuis=["C0040405", "C0024265"])
print(lay)
# The Chest CT scan shows enlarged lymph nodes near the center of the lungs.
```

## Notes

- A free UMLS API key is required for `cui_to_preferred`. Without it the
  surface form falls back to the raw CUI.
- `en_core_sci_md` from SciSpacy must be installed separately:
  `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz`
- `meta-llama/Meta-Llama-3.1-8B-Instruct` is gated on Hugging Face.
