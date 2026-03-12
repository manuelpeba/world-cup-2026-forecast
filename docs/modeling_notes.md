## Model Specification Strategy

During the development of the match outcome prediction system, two different model specifications were evaluated.

The objective was to balance **predictive performance** and **simulation realism** for future tournament forecasting.

### 1. Full Model Specification

The full model includes all available engineered features, including variables derived from the Elo rating update mechanism.

Features included:

- team_a_elo_before
- team_b_elo_before
- elo_diff
- abs_elo_diff
- rolling performance features
- team_a_expected_result
- team_b_expected_result
- match_k_factor
- match_goal_diff_multiplier
- tournament
- neutral_venue

This specification produced the best historical predictive performance.

Typical results:

| Metric | Logistic Regression |
|---|---|
| Accuracy | ~0.61 |
| Log Loss | ~0.76 |
| Brier Score | ~0.46 |

However, some of these variables are **not observable before the match**, which makes them less appropriate for forward simulation.

---

### 2. Pre-Match Model Specification

A second specification was tested using only features available before kickoff.

Removed features:

- match_k_factor
- match_goal_diff_multiplier

These variables are part of the Elo update mechanism and cannot be observed for future matches.

The resulting feature set included:

- Elo ratings
- Elo difference
- expected result from Elo
- rolling team performance
- tournament context
- neutral venue indicator

Results:

| Metric | Logistic Regression |
|---|---|
| Accuracy | ~0.60 |
| Log Loss | ~0.87 |
| Brier Score | ~0.51 |

Predictive performance degraded compared with the full model.

predictive accuracy
vs
simulation realism


---

### Decision

Both model specifications are kept in the repository.

- **full_model** → used as performance benchmark
- **prematch_model** → candidate model for tournament simulation

Further work will focus on improving the pre-match specification by adding richer contextual features rather than relying on Elo update parameters.

---

### Next Steps

Future modeling improvements may include:

- squad strength features
- FIFA ranking
- confederation indicators
- rest days between matches
- travel distance effects
- player availability (future extension)

These features could improve the pre-match model without introducing post-match information leakage.

---

### Interpretation

The removed features contain predictive information derived from historical match dynamics.

Although conceptually cleaner, the pre-match specification loses useful signal.

This creates a trade-off between:


## Experiment: Adding abs_elo_diff

Hypothesis:
Absolute Elo difference may help capture match balance and improve draw prediction.

Changes:
- Added feature `abs_elo_diff`
- Added `neutral_venue` contextual feature

Results:

Logistic Regression
Accuracy: 0.608
Log Loss: 0.765

Random Forest
Accuracy: 0.614
Log Loss: 0.767

Conclusion:
No major performance change but improves model interpretability and simulation realism.