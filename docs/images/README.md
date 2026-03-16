# Images and Visual Assets

This directory contains visual assets used throughout the project documentation.

Images stored here are primarily used in:

- The main project `README`
- Analytical documentation
- Demonstrations of model and simulation outputs

---

## Current Visualizations

### Champion Probability Forecast

Shows the probability of each team winning the FIFA World Cup 2026 based on Monte Carlo simulation results.

Generated from:

`notebooks/02_simulation_analysis.ipynb`

File: 

`champion_probabilities.png`

---

### Team Progression Probabilities

Displays the probability of each team reaching different tournament stages.

Stages include:

- Group stage qualification
- Round of 16
- Quarterfinal
- Semifinal
- Final
- Champion

Generated from:

`notebooks/02_simulation_analysis.ipynb`

File:

`team_progression.png`

---

### Champion Distribution

Distribution of simulated tournament champions across all Monte Carlo simulation runs.

This visualization helps illustrate the **uncertainty inherent in tournament forecasts**.

Generated from:

`notebooks/03_world_cup_forecast_story.ipynb`

File: 

`champion_distribution.png`

---

## How Images Are Generated

Figures are produced from the analytical notebooks and saved programmatically using `matplotlib`.

Example:

```python
plt.savefig("docs/images/champion_probabilities.png", dpi=300)
```

This ensures that documentation images remain reproducible and synchronized with the latest simulation results.

## Guidelines

When adding new images:

1. Generate them from notebooks or scripts.

2. Save them programmatically to ensure reproducibility.

3. Use clear and descriptive filenames.

4. Prefer high resolution (dpi=300) for documentation quality.

Example naming pattern:

`team_progression.png`
`champion_probabilities.png`
`simulation_distribution.png`