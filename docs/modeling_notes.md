A continuación tienes **dos documentos completos en inglés**, listos para pegar en:

```
docs/engineering_notes.md
docs/modeling_notes.md
```

Están escritos en un estilo **técnico y profesional**, pensado para recruiters de:

* football clubs
* sports analytics companies
* betting analytics
* data science teams

---

# `docs/engineering_notes.md`

```markdown
# Engineering Notes

## Project Overview

This project implements a production-style football forecasting pipeline that combines:

- probabilistic match outcome prediction
- tournament-level Monte Carlo simulation
- scalable simulation execution
- structured aggregation of results
- reproducible artifact exports
- interactive visualization via Streamlit

The system simulates complete international tournaments (currently a 32-team World Cup format) and estimates advancement probabilities for each team.

Outputs include:

- probability of advancing from the group stage
- probability of reaching each knockout round
- probability of winning the tournament

The architecture is modular and designed to resemble a simplified production forecasting system.

---

# System Architecture

The simulation pipeline is divided into several independent modules.

```

data → features → match prediction → tournament simulation → aggregation → reporting → dashboard

```

Core modules are located under:

```

src/
├─ models/
├─ simulation/
├─ utils/

````

The simulation layer contains most of the engineering logic.

---

# Simulation Engine

The tournament simulator is built as a Monte Carlo engine.

For each simulation run:

1. Group stage matches are simulated.
2. Group tables are computed.
3. Teams advance to the knockout bracket.
4. Knockout rounds are simulated until a champion is determined.

This process is repeated **N times** to estimate stage probabilities.

Example command:

```bash
py -m src.simulation.run_simulation \
--groups-path configs/world_cup_groups.json \
--num-simulations 100000 \
--num-workers 4
````

---

# Module Responsibilities

## `group_stage.py`

Responsible for:

* simulating round-robin matches within groups
* updating group tables
* computing points and tie-breakers
* determining which teams advance

Each group produces:

```
GroupTable
QualifiedTeams
MatchSimulationResults
```

---

## `knockout_stage.py`

Builds and simulates the knockout bracket.

Rounds simulated:

```
Round of 16
Quarterfinals
Semifinals
Final
```

Outputs:

```
quarterfinalists
semifinalists
finalists
champion
```

Each knockout match:

1. obtains match probabilities from the prediction model
2. samples an outcome
3. resolves draws using a configured rule

Draw resolution methods include:

```
coin_flip
elo_weighted
```

---

## `tournament.py`

This module orchestrates a **single tournament simulation**.

Main steps:

```
simulate_group_stage()
simulate_knockout_stage()
assemble TournamentRunResult
```

It also provides:

```
simulate_many_tournaments()
```

which performs repeated simulations using a shared predictor instance.

---

## `aggregation.py`

Transforms raw simulation runs into analytical outputs.

Key outputs:

```
team_probabilities
champion_distribution
stage_presence
stage_counts
```

Probabilities are computed by averaging binary stage indicators across simulations.

Example:

```
P(team reaches semifinal)
= mean(semifinal_flag)
```

---

## `reporting.py`

Exports simulation outputs to disk.

Generated artifacts:

```
team_probabilities.parquet
team_probabilities.csv

champion_distribution.parquet
champion_distribution.csv

match_logs.parquet

summary_metadata.json
```

These files serve as inputs for:

* dashboards
* external analysis
* reporting pipelines

---

# Parallel Simulation

The engine supports parallel execution.

Simulations can be distributed across multiple workers:

```
--num-workers 4
```

Each worker runs independent batches of tournament simulations.

Benefits:

* near-linear scaling
* significantly faster large simulation runs
* reproducible results via controlled random seeds

Example scale:

```
100,000 tournament simulations
executed in seconds to minutes
```

depending on hardware.

---

# Reproducibility

All simulations use a deterministic random seed:

```
SimulationConfig.random_seed
```

This guarantees reproducibility across runs when configuration is unchanged.

---

# Output Data Model

The core output structure is `TournamentRunResult`.

Key attributes:

```
simulation_id
group_tables
group_stage_results
qualified_teams
quarterfinalists
semifinalists
finalists
champion
match_results
metadata
```

Aggregation modules operate directly on this object.

---

# Visualization Layer

A Streamlit dashboard (`app/streamlit_app.py`) provides an interactive interface.

The dashboard reads exported artifacts and presents:

* team advancement probabilities
* champion probability rankings
* team-level breakdowns
* match log previews

This layer is intentionally decoupled from the simulation engine.

---

# Current Limitations

The current system assumes:

```
32 teams
8 groups
2 teams advancing per group
standard knockout bracket
```

The real 2026 World Cup format will include:

```
48 teams
12 groups
best third-place teams advancing
round of 32
```

Future work will extend the bracket generator to support this structure.

---

# Future Engineering Improvements

Potential upgrades include:

* distributed simulation execution
* GPU acceleration for match prediction
* scoreline simulation (Poisson / xG models)
* dynamic Elo updates during the tournament
* live tournament updates
* API layer for real-time queries

---

# Summary

The project demonstrates how to build a modular football forecasting system that integrates:

* machine learning prediction
* tournament simulation
* scalable computation
* analytical reporting
* interactive visualization

````

---

# `docs/modeling_notes.md`

```markdown
# Modeling Notes

## Modeling Objective

The modeling component estimates match outcome probabilities for international football matches.

The prediction target is:

````

team_a_win
draw
team_a_loss

```

These probabilities are used as inputs for the tournament simulation engine.

---

# Data Sources

The dataset is constructed from historical international matches.

Typical sources include:

- FIFA international match results
- historical Elo ratings
- match metadata (tournament, neutral venue)

Matches span multiple decades to capture long-term team performance trends.

---

# Feature Engineering

Features describe the relative strength and recent form of the competing teams.

Examples include:

## Elo Rating

```

team_a_elo_before
team_b_elo_before
elo_diff
abs_elo_diff

```

Elo captures long-term team strength and has strong predictive power in football.

---

## Rolling Performance Metrics

Computed over recent matches:

```

rolling_goals_scored
rolling_goals_conceded
rolling_goal_diff
rolling_win_rate
rolling_points

```

These capture short-term form and performance trends.

---

## Derived Features

Additional comparative features include:

```

rolling_goal_diff_diff
rolling_points_diff
rolling_win_rate_diff
rolling_goals_scored_diff
rolling_goals_conceded_diff

```

These represent relative advantages between teams.

---

## Elo-Based Expected Result

A theoretical expected score based on Elo:

```

expected_result = 1 / (1 + 10^((elo_opponent - elo_team) / 400))

```

This approximates the expected win probability under Elo assumptions.

---

# Model Architecture

The current implementation uses a multi-class classifier.

Model:

```

Logistic Regression

```

Predicting:

```

win
draw
loss

```

The model is trained on engineered match-level features.

---

# Training Strategy

Key aspects:

- temporal train/test split
- prevention of future information leakage
- evaluation on modern matches

Example split:

```

Train: historical matches up to 2017
Test: recent matches

```

---

# Model Output

The model outputs a probability vector:

```

P(win)
P(draw)
P(loss)

```

These probabilities are normalized to sum to 1.

Example:

```

Spain vs Brazil

win  = 0.42
draw = 0.29
loss = 0.29

```

---

# Integration with Tournament Simulation

During simulation:

1. Match features are constructed dynamically.
2. The prediction model produces outcome probabilities.
3. Outcomes are sampled from this distribution.

Example sampling:

```

sample from categorical distribution:
[win, draw, loss]

```

This allows stochastic tournament simulations.

---

# Knockout Draw Resolution

Draws cannot occur in knockout rounds.

If a match result is sampled as "draw":

a tie-breaker rule is applied.

Current options:

```

coin_flip
elo_weighted

```

The Elo-weighted method biases penalties toward stronger teams.

---

# Why Monte Carlo Simulation?

A single deterministic tournament path does not capture uncertainty.

Monte Carlo simulation allows estimation of:

```

P(reach quarterfinal)
P(reach semifinal)
P(reach final)
P(win tournament)

```

by repeating the tournament many times.

Typical runs:

```

10,000 – 100,000 simulations

```

---

# Interpretation of Results

Tournament probabilities reflect both:

- team strength
- bracket structure

Two equally strong teams may have different championship probabilities depending on their path through the bracket.

---

# Current Modeling Limitations

The model currently predicts only:

```

win / draw / loss

```

It does not model:

```

exact scorelines
goal distributions
expected goals

```

Therefore:

- match scores are not simulated
- tie-breakers rely on heuristics

---

# Potential Modeling Improvements

Future enhancements could include:

### Scoreline models

```

Poisson goal models
bivariate Poisson
expected goals models

```

### Player-level features

```

squad strength
injuries
club form

### Dynamic team strength


in-tournament Elo updates
form adjustments


### Ensemble modeling

Combining:

Elo
xG models
ML classifiers

---

# Summary

The modeling layer provides probabilistic match outcomes that feed the simulation engine.

This approach allows the system to estimate tournament-level probabilities using a combination of:

- statistical prediction
- stochastic simulation
- aggregation of simulated outcomes