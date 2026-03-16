# Project Status — World Cup 2026 Forecast

This document tracks the current development status, completed components, and future roadmap of the **World Cup 2026 Forecasting Engine**.

The goal of the project is to build a **production-style sports analytics pipeline** capable of forecasting international tournaments using probabilistic modeling and Monte Carlo simulation.

---

# Current Status

The core forecasting framework is fully implemented.

Completed components:

* Match outcome prediction model
* Tournament simulation engine
* Monte Carlo simulation pipeline
* Analytical notebooks for simulation interpretation
* Forecast storytelling notebook
* Streamlit dashboard for interactive visualization
* Model benchmarking experiment
* Project documentation and repository structure

The project now provides a **complete pipeline from raw historical data to tournament forecasts and analytical insights.**

---

# 1. Project Overview

The system estimates the probability that each national team:

* advances from the group stage
* reaches the Round of 16
* reaches the quarterfinals
* reaches the semifinals
* reaches the final
* wins the tournament

This is achieved by combining:

* machine learning match prediction
* team strength feature engineering
* tournament simulation
* Monte Carlo forecasting
* analytics dashboards

---

# 2. Current Development Status

| Component                         | Status                          |
| --------------------------------- | ------------------------------- |
| Data ingestion pipeline           | Complete                        |
| Feature engineering               | Complete                        |
| Team strength snapshot generation | Complete                        |
| Match prediction model            | Production baseline implemented |
| Tournament simulation engine      | Complete                        |
| 32-team tournament format         | Complete                        |
| 48-team World Cup format          | Implemented                     |
| Monte Carlo simulation pipeline   | Complete                        |
| Simulation artifact export        | Complete                        |
| Streamlit dashboard               | Implemented                     |
| Research notebooks                | Complete                        |
| Model benchmarking                | Complete                        |
| Project documentation             | In progress                     |

---

# 3. Data Pipeline

Implemented components:

* historical international match ingestion
* feature engineering for team strength metrics
* rolling performance indicators
* Elo rating features

Primary feature snapshot:

```
data/processed/latest_team_features.parquet
```

Dataset size:

~31,000 historical international matches.

---

# 4. Match Outcome Modeling

Current production model:

**Multiclass Logistic Regression**

Prediction target:

```
P(win)
P(draw)
P(loss)
```

Features include:

* Elo rating difference
* rolling goals scored
* rolling goals conceded
* rolling goal difference
* rolling win rate
* rolling points

The model produces **probabilistic match outcome predictions**, which are used by the simulation engine.

---

# 5. Model Benchmarking

A dedicated benchmark experiment was conducted to evaluate candidate models for match outcome prediction.

The evaluation compared:

* Logistic Regression
* Logistic Regression (Calibrated)
* Random Forest

Models were assessed using probabilistic forecasting metrics:

* Log Loss
* Multiclass Brier Score
* Accuracy

During this experiment, **data leakage was detected and removed**, specifically:

* match goals
* post-match Elo features

These variables contained post-match information and artificially inflated model performance.

After removing leakage features, the benchmark produced the following results:

| Model                            | Accuracy   | Log Loss   | Brier Score |
| -------------------------------- | ---------- | ---------- | ----------- |
| Logistic Regression              | **0.5996** | **0.8731** | **0.5149**  |
| Logistic Regression (Calibrated) | 0.5992     | 0.8816     | 0.5186      |
| Random Forest                    | 0.5733     | 0.8950     | 0.5295      |

Logistic Regression achieved the best probabilistic performance and remains the **production model used by the tournament simulation engine**.

The benchmark experiment is documented in:

```
experiments/05_match_model_benchmark.ipynb
```

---

# 6. Tournament Simulation Engine

The simulation engine generates probabilistic tournament outcomes using Monte Carlo methods.

Core logic:

```
predict_match(team_a, team_b)
↓
sample outcome
↓
simulate match
↓
simulate tournament
↓
repeat N times
```

Each simulation generates:

* group tables
* qualified teams
* knockout progression
* finalists
* champion

---

# 7. Tournament Format Support

### Classic World Cup Format

* 32 teams
* 8 groups
* Round of 16

### World Cup 2026 Format

* 48 teams
* 12 groups
* best 8 third-place teams qualify
* Round of 32 knockout stage

Key modules:

```
src/simulation/
    bracket_builder.py
    qualification.py
    group_stage.py
    knockout_stage.py
```

---

# 8. Simulation Outputs

Export directory:

```
data/outputs/simulation
```

Generated artifacts:

* `team_probabilities.csv`
* `champion_distribution.csv`
* `match_logs.parquet`
* `summary_metadata.json`

These outputs allow detailed analysis of tournament outcomes and team progression probabilities.

---

# 9. Dashboard

Location:

```
src/dashboard/
```

Features:

* champion probability chart
* team probability leaderboard
* team explorer
* match log preview
* simulation metadata display

The dashboard allows interactive exploration of simulation results.

---

# 10. Analytical Notebooks

The analytical notebook layer has been completed and includes:

* **00_eda_match_dataset.ipynb**
  Exploratory analysis of the historical match dataset.

* **02_simulation_analysis.ipynb**
  Analysis of Monte Carlo simulation outputs and team progression probabilities.

* **03_world_cup_forecast_story.ipynb**
  Narrative forecast of the FIFA World Cup 2026.

These notebooks provide transparency into the forecasting process and communicate model insights and simulation outcomes.

---

# 11. Research Experiments

Exploratory modeling experiments are stored in the `experiments/` directory.

* **01_match_model_experiments.ipynb**
  Early experimentation with match outcome models and feature engineering.

* **05_match_model_benchmark.ipynb**
  Formal benchmark experiment used to evaluate candidate models and select the production model.

This separation keeps exploratory work isolated from the production forecasting pipeline.

---

# 12. Documentation

Project documentation currently includes:

* Architecture overview — `docs/architecture.md`
* Engineering documentation — `docs/engineering.md`
* Modeling documentation — `docs/modeling.md`
* Project status — `docs/project_status.md`
* Visual documentation assets — `docs/images/`

The main project README includes diagrams and simulation visualizations.

---

# 13. Next Development Steps

## Final Tournament Configuration

Update:

```
configs/world_cup_groups_48.json
```

with official teams and groups once World Cup qualification and the final draw are completed.

---

## Model Improvements

Potential improvements include:

* Gradient Boosting
* XGBoost
* LightGBM
* Bayesian hierarchical models

Future experiments may explore:

* hyperparameter tuning
* improved calibration strategies
* cross-validation schemes

---

## Goal-Based Match Modeling

Future work may incorporate **goal-scoring models**, such as:

* Poisson goal models
* bivariate Poisson models
* expected goals (xG) models

These approaches would allow simulation of **scorelines instead of only match outcomes**.

---

# 14. Repository Maintenance

Planned maintenance tasks:

* remove deprecated experiments
* clean unused configuration files
* archive early notebooks
* remove temporary artifacts

Goal: maintain a **clean production-style repository structure**.

---

# 15. Long-Term Extensions

Potential long-term improvements include:

* player-level impact models
* expected goals (xG) features
* dynamic Elo updates during tournaments
* distributed simulation engine
* automated tournament updates