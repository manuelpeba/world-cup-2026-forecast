# Experiments

This directory contains research-oriented notebooks used to explore modeling approaches and validate assumptions before integrating them into the production pipeline.

Experiments are typically exploratory and may include:

- Model comparisons
- Feature engineering trials
- Hyperparameter exploration
- Evaluation of alternative modeling strategies

Code in this directory **is not considered production-ready**.  
Reusable components and finalized implementations should be moved to the `src/` package.

---

## Current Experiments

### 01_match_model_experiments.ipynb

Exploratory notebook used to evaluate candidate approaches for predicting football match outcomes.

Topics explored include:

- Baseline models for match outcome prediction
- Feature impact analysis
- Model comparison and validation
- Early evaluation of prediction performance

Insights from this notebook informed the development of the production model located in:

`src/models/match_outcome/`

---

## Relationship to the Production Pipeline

The repository separates **research code** from **production code**:

| Directory | Purpose |
|----------|--------|
| `experiments/` | Research and exploratory modeling |
| `notebooks/` | Analytical and storytelling notebooks |
| `src/` | Production pipeline and simulation engine |

This separation helps keep the codebase maintainable and ensures that the production system remains stable while experimentation continues.

---

## Notes

Notebooks in this directory may:

- Contain exploratory code
- Include partially tested ideas
- Be refactored or removed as research evolves

Finalized logic should always be migrated to the `src/` modules.