# Modeling Notes

## Modeling Objective

The modeling component estimates match outcome probabilities for international football matches.

The prediction target is:

team_a_win
draw
team_a_loss

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

team_a_elo_before
team_b_elo_before
elo_diff
abs_elo_diff

Elo captures long-term team strength and has strong predictive power in football.

---

## Rolling Performance Metrics

Computed over recent matches:

rolling_goals_scored
rolling_goals_conceded
rolling_goal_diff
rolling_win_rate
rolling_points

These capture short-term form and performance trends.

---

## Derived Features

Additional comparative features include:

rolling_goal_diff_diff
rolling_points_diff
rolling_win_rate_diff
rolling_goals_scored_diff
rolling_goals_conceded_diff

These represent relative advantages between teams.

---

## Elo-Based Expected Result

A theoretical expected score based on Elo:

expected_result = 1 / (1 + 10^((elo_opponent - elo_team) / 400))

This approximates the expected win probability under Elo assumptions.

---

# Model Architecture

The current implementation uses a multi-class classifier.

Model:

Logistic Regression

Predicting:

win
draw
loss

The model is trained on engineered match-level features.

---

# Training Strategy

Key aspects:

- temporal train/test split
- prevention of future information leakage
- evaluation on modern matches

Example split:

Train: historical matches up to 2017
Test: recent matches

---

# Model Output

The model outputs a probability vector:

P(win)
P(draw)
P(loss)

These probabilities are normalized to sum to 1.

Example:

Spain vs Brazil

win  = 0.42
draw = 0.29
loss = 0.29

---

# Integration with Tournament Simulation

During simulation:

1. Match features are constructed dynamically.
2. The prediction model produces outcome probabilities.
3. Outcomes are sampled from this distribution.

Example sampling:

sample from categorical distribution:
[win, draw, loss]

This allows stochastic tournament simulations.

---

# Knockout Draw Resolution

Draws cannot occur in knockout rounds.

If a match result is sampled as "draw":

a tie-breaker rule is applied.

Current options:

`coin_flip`
`elo_weighted`

The Elo-weighted method biases penalties toward stronger teams.

---

# Why Monte Carlo Simulation?

A single deterministic tournament path does not capture uncertainty.

Monte Carlo simulation allows estimation of:

P(reach quarterfinal)
P(reach semifinal)
P(reach final)
P(win tournament)

by repeating the tournament many times.

Typical runs:

10,000 – 100,000 simulations

---

# Interpretation of Results

Tournament probabilities reflect both:

- team strength
- bracket structure

Two equally strong teams may have different championship probabilities depending on their path through the bracket.

---

# Current Modeling Limitations

The model currently predicts only:

win / draw / loss

It does not model:

exact scorelines
goal distributions
expected goals

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