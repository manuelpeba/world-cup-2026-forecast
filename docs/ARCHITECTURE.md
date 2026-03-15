# System Architecture --- World Cup 2026 Forecast

This document describes the technical architecture of the **World Cup
2026 Forecasting Engine**.

The system is designed as a modular sports analytics pipeline combining:

-   data engineering
-   machine learning
-   probabilistic simulation
-   analytics reporting
-   interactive visualization

------------------------------------------------------------------------

# 1. High-Level Architecture

Pipeline overview:

Historical Match Data\
↓\
Data Ingestion\
↓\
Feature Engineering\
↓\
Team Strength Features\
↓\
Match Outcome Model\
↓\
Monte Carlo Tournament Simulation\
↓\
Aggregation Layer\
↓\
Forecast Outputs\
↓\
Dashboard

------------------------------------------------------------------------

# 2. Data Layer

Input sources:

-   historical international match results
-   tournament metadata

Processing steps:

1.  ingest raw match data
2.  clean and normalize datasets
3.  compute rolling team statistics
4.  compute Elo ratings
5.  generate modeling dataset

Output artifact:

data/processed/latest_team_features.parquet

------------------------------------------------------------------------

# 3. Feature Engineering

Team strength features include:

-   Elo rating
-   Elo difference
-   rolling goals scored
-   rolling goals conceded
-   rolling goal difference
-   rolling win rate
-   rolling points

Features are computed using historical match windows to capture recent
team performance.

------------------------------------------------------------------------

# 4. Match Outcome Model

The match prediction model estimates:

P(win)\
P(draw)\
P(loss)

from the perspective of Team A.

Current baseline:

Multiclass Logistic Regression

Model inputs:

-   team strength metrics
-   Elo difference
-   rolling performance indicators

Model output:

Probability distribution used by the tournament simulator.

------------------------------------------------------------------------

# 5. Simulation Engine

The simulation engine converts match probabilities into full tournament
outcomes.

Core workflow:

predict_match(team_a, team_b)\
↓\
sample outcome from probability distribution\
↓\
simulate match result\
↓\
update tournament state\
↓\
continue tournament progression

------------------------------------------------------------------------

# 6. Tournament Engine

Main modules:

src/simulation/

run_simulation.py --- CLI entry point\
tournament.py --- tournament orchestration\
group_stage.py --- group stage simulation\
knockout_stage.py --- knockout round simulation\
qualification.py --- third-place qualification logic\
bracket_builder.py --- knockout bracket construction\
aggregation.py --- probability aggregation

------------------------------------------------------------------------

# 7. Tournament Configuration

Tournament formats are configurable via JSON.

configs/

world_cup_groups.json --- 32-team format\
world_cup_groups_48.json --- 48-team format\
world_cup_2026_bracket.json --- knockout bracket configuration

------------------------------------------------------------------------

# 8. Monte Carlo Simulation

The system runs thousands of simulated tournaments.

Typical runs:

10,000  --  100,000 simulations

Each simulation produces:

-   group standings
-   knockout progression
-   finalists
-   champion

Results are aggregated across all simulations to compute probabilities.

------------------------------------------------------------------------

# 9. Aggregation Layer

Aggregates simulation results into:

team advancement probabilities\
champion distribution\
simulation match logs

Artifacts exported to:

data/outputs/simulation

------------------------------------------------------------------------

# 10. Reporting Layer

Outputs include:

team_probabilities.csv\
champion_distribution.csv\
match_logs.parquet\
summary_metadata.json

These files power the dashboard and external analysis.

------------------------------------------------------------------------

# 11. Visualization Layer

Interactive analytics dashboard built with **Streamlit**.

app/streamlit_app.py

Capabilities:

-   explore championship probabilities
-   inspect team progression probabilities
-   preview simulated matches
-   analyze forecast outputs

------------------------------------------------------------------------

# 12. Future Architecture Improvements

Possible architectural extensions:

-   goal-based simulation models (Poisson)
-   distributed simulation engine
-   automated data refresh pipelines
-   player-level modeling
