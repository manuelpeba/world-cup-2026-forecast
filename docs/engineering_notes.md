# Engineering Notes – World Cup 2026 Forecast

This document tracks engineering decisions, pitfalls, and development conventions used in the project.

The goal is to avoid repeating common errors and maintain a consistent development workflow.

---

# 1. Project Execution Rules

## Running Python modules

Python modules inside the `src/` directory must always be executed using the module syntax:

```bash
python -m src.module.submodule
```
Examples:

`python -m src.pipelines.ingest_data`
`python -m src.features.team_features`

Running scripts directly (e.g. python src/.../file.py) may break imports.

# 2. Project Path Management

Paths are centralized in:

`src/utils/config.py`

This file defines:

- PROJECT_ROOT

- DATA_DIR

- RAW_DATA_DIR

- INTERIM_DATA_DIR

- PROCESSED_DATA_DIR

- ARTIFACTS_DIR

Example usage:

`from src.utils.config import PROCESSED_DATA_DIR`

Avoid hardcoded paths in scripts or notebooks.

# 3. Notebook Bootstrap

All notebooks must begin with the following setup cell:

```bash
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))
```

This ensures that the src module can be imported.

# 4. Data Pipeline Rule

Derived datasets must not be used before running the pipeline that generates them.

Example:

Before using:

`data/interim/matches_clean.parquet`

Run:

`python -m src.pipelines.ingest_data`

# 5. Directory Creation

When creating new modules, ensure the directory exists first.

Example:

```bash
mkdir -p src/pipelines
touch src/pipelines/new_pipeline.py
```

# 6. Common Pitfalls Encountered

## Relative paths in notebooks

Notebooks run from the `notebooks/` directory.

Therefore:

`data/...`

may not resolve correctly.

Use PROJECT_ROOT with Path.

## Python module resolution errors

Error example:

`ModuleNotFoundError: No module named 'src'`

Cause:

Running scripts directly instead of with python -m.

Solution:

Use module execution.

# 7. Development Guidelines

- Keep notebooks for exploration only

- Move reusable logic into src/

- Pipelines must produce reproducible datasets

- Models should only consume processed datasets

- Avoid circular imports in src/

# 8. Future Engineering Topics

This document will expand with:

- Data validation rules

- Feature store decisions

- Model versioning

- Simulation reproducibility

- Experiment tracking

---

# 9. Dataset Filtering Strategy (v2)

During exploratory analysis of the `team_match_features` dataset, the following issue was identified:

Even after removing many non-FIFA teams through an explicit exclusion list, several regional or non-eligible teams still remained in the dataset (e.g., Brittany).

This revealed that the blacklist approach was not sufficiently robust.

## Decision

The filtering strategy will move from a **blacklist model** to an **allowlist model**.

Instead of excluding specific teams, the pipeline will explicitly keep only teams that belong to a curated list of valid national teams.

This list will be stored in:

`configs/allowed_teams.yaml`


## Benefits

This approach:

- prevents regional or CONIFA teams from entering the dataset
- makes the filtering logic explicit and auditable
- improves reproducibility
- simplifies future maintenance

## Pipeline Update

The filtering pipeline will now follow this order:

team_match_features
↓
filter by year
↓
filter by tournament
↓
filter by allowed national teams
↓
drop rows without rolling history
↓
matches_filtered


## Future Improvements

Future versions may include:

- team name normalization (aliases)
- historical team mappings (e.g., Yugoslavia → Serbia/Croatia etc.)
- confederation-level metadata
- FIFA membership metadata

## Dataset filtering reduced matches from 98k to 63k rows.

Final dataset properties:
- ~31k real matches
- 185 national teams
- time range: 1950–present
- tournaments: 26 relevant international competitions

---

# 10. Team Strength Model — Elo Rating System

A dynamic Elo rating model was implemented to estimate the evolving strength of national teams over time.

Script:

`src/models/team_strength/elo_model.py`

Input dataset:

`data/processed/matches_filtered.parquet`

Output dataset:

`data/processed/team_elo_ratings.parquet`


---

## Model Design

The Elo system estimates team strength using historical match results and updates ratings sequentially in chronological order.

Each national team starts with a base rating of: 5.000

Ratings are updated after every match using the standard Elo update formula:

    R_new = R_old + K × G × (S − E)


Where:

| Variable | Meaning |
|--------|--------|
| K | Tournament importance factor |
| G | Goal difference multiplier |
| S | Actual match result |
| E | Expected match result |

---

## Expected Result

The expected result is computed as:

    E = 1 / (1 + 10^((R_opponent − R_team) / 400))


---

## Tournament Weighting (K-factor)

Different competitions have different importance levels:

| Competition Type | K |
|------------------|--|
| Friendly | 20 |
| Qualification matches | 30 |
| Regional competitions / Nations League | 35 |
| Continental tournaments | 40 |
| FIFA World Cup | 50 |

This ensures that high-stakes matches impact ratings more strongly.

---

## Goal Difference Adjustment

To reflect the information contained in large victories:

| Goal Difference | Multiplier |
|-----------------|-----------|
| 1 goal | 1.0 |
| 2 goals | 1.5 |
| 3+ goals | 1.75 |

---

## Home Advantage

If the match is not played at a neutral venue, the home team receives an Elo adjustment: +80 Elo


before computing expected results.

---

## Output Features

The resulting dataset contains the following key variables:

| Feature | Description |
|-------|-------------|
| elo_before | Team rating before the match |
| elo_after | Team rating after the match |
| opponent_elo_before | Opponent rating before match |
| elo_diff_before | Rating difference between teams |
| expected_result | Predicted probability of winning |
| actual_result | Actual match outcome |

This dataset forms the **team strength layer** of the project and will be used to construct match prediction features.

---

## Dataset Summary

Current Elo dataset:

- ~63k team-level observations
- ~31k real matches
- 185 national teams
- Time span: **1950 → present**

---

## Role in the Modeling Framework

The Elo model corresponds to **Layer 2** of the project architecture:

Layer 1 — Player Impact Model
Layer 2 — Team Strength Model (Elo)
Layer 3 — Match Outcome ML Model

The Elo ratings will be combined with rolling team performance features to build the final match prediction dataset.
