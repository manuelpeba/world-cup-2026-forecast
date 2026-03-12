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