# Project Status --- World Cup 2026 Forecast

This document tracks the current development status, completed
components, and future roadmap of the **World Cup 2026 Forecasting
Engine**.

The goal of the project is to build a **production-style sports
analytics pipeline** capable of forecasting international tournaments
using probabilistic modeling and Monte Carlo simulation.

---

# 1. Project Overview

The system estimates the probability that each national team:

-   advances from the group stage
-   reaches the Round of 16
-   reaches the quarterfinals
-   reaches the semifinals
-   reaches the final
-   wins the tournament

This is achieved by combining:

-   machine learning match prediction
-   team strength feature engineering
-   tournament simulation
-   Monte Carlo forecasting
-   analytics dashboards

---

# 2. Current Development Status

| Component | Status |
| --- | --- |
| Data ingestion pipeline | Complete |
| Feature engineering | Complete |
| Team strength snapshot generation | Complete |
| Match prediction model | Baseline implemented |
| Tournament simulation engine | Complete |
| 32-team tournament format | Complete |
| 48-team World Cup format | Implemented |
| Monte Carlo simulation pipeline | Complete |
| Simulation artifact export | Complete |
| Streamlit dashboard | Implemented |
| Research notebooks | Complete |
| Project documentation | In progress |

---

# 3. Completed Components

## Data Pipeline

Implemented:

-   historical international match ingestion
-   feature engineering for team strength metrics
-   rolling performance indicators
-   Elo rating features

Primary feature snapshot:

data/processed/latest_team_features.parquet

---

## Match Outcome Model

Current model: **Multiclass Logistic Regression**

Predicts:

P(win)\
P(draw)\
P(loss)

Features include:

-   Elo difference
-   rolling goals scored
-   rolling goals conceded
-   rolling goal difference
-   rolling win rate
-   rolling points

Dataset size: \~31k historical international matches.

---

## Tournament Simulation Engine

Core logic:

predict_match(team_a, team_b)\
↓\
sample outcome\
↓\
simulate match\
↓\
simulate tournament\
↓\
repeat N times

Each simulation generates:

-   group tables
-   qualified teams
-   knockout progression
-   finalists
-   champion

---

# 4. Tournament Format Support

### V1 --- Classic World Cup

-   32 teams
-   8 groups
-   Round of 16

### V2 --- World Cup 2026

-   48 teams
-   12 groups
-   best 8 third-place teams qualify
-   Round of 32 knockout stage

Key modules:

src/simulation/

-   bracket_builder.py
-   qualification.py
-   tournament.py

---

# 5. Simulation Outputs

Export directory:

data/outputs/simulation

Generated artifacts:

-   team_probabilities.csv
-   champion_distribution.csv
-   match_logs.parquet
-   summary_metadata.json

---

# 6. Dashboard

Location:

app/streamlit_app.py

Features:

-   champion probability chart
-   team probability leaderboard
-   team explorer
-   match log preview
-   simulation metadata display

---

# 7. Next Development Steps

## Complete Real Tournament Configuration

Update:

configs/world_cup_groups_48.json

with official teams once qualification and group draw are finalized.

---

## Improve Match Prediction Model

Current baseline: Logistic Regression

Potential improvements:

-   Gradient Boosting
-   XGBoost
-   LightGBM
-   Bayesian hierarchical models

Add:

-   hyperparameter tuning
-   probability calibration
-   cross-validation

---

## Introduce Goal-Based Models

Future improvement:

Poisson goal model

predict goals scored\
↓\
simulate scoreline\
↓\
derive match result

---

## Improve Dashboard

Planned visualizations:

-   stacked progression chart
-   probability evolution plots
-   team comparison charts
-   tournament bracket visualization

---

# 8. Documentation Tasks

Files to complete:

docs/engineering_notes.md\
docs/modeling_notes.md\
docs/architecture.md

---

# 9. Repository Cleanup

Tasks:

-   remove deprecated experiments
-   clean unused configuration files
-   archive early notebooks
-   remove temporary artifacts

Goal:  maintain a clean production-style repository.

---

# 10. Analytical notebooks

Status: Completed

The repository includes three notebooks used for exploratory analysis and communication:

- Match model experiments
- Tournament simulation analysis
- Forecast storytelling and interpretation

These notebooks complement the production pipeline by providing transparency into model behavior and tournament forecast structure.

---

# 11. Long-Term Extensions

-   player-level impact models
-   expected goals (xG) features
-   distributed simulation engine
-   automated tournament updates
