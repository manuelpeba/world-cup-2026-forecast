# Notebooks

This directory contains exploratory and analytical notebooks used during the development of the World Cup forecasting project.

These notebooks are **not required to run the simulation engine**.  
The production pipeline lives under `src/` and should be executed through the CLI or Python modules.

## Notebook Overview

### 00_eda_match_dataset.ipynb

Exploratory data analysis of the historical international match dataset used to build the modeling features.

Focus areas include:
- dataset structure
- target distribution
- temporal coverage
- feature sanity checks

### 01_match_model_experiments.ipynb

Research notebook for match outcome modeling experiments.

Includes:

- dataset inspection and feature validation
- temporal train/test evaluation
- baseline multinomial logistic regression model
- candidate model comparisons
- probability calibration diagnostics
- evaluation metrics (accuracy, log loss)

### 02_simulation_analysis.ipynb

Analysis of Monte Carlo tournament simulation outputs.

Explores:
- advancement probabilities
- champion distribution
- upset frequency
- simulation stability.

### 03_world_cup_forecast_story.ipynb

Storytelling notebook presenting the main insights from the World Cup forecast.

Designed for:
- portfolio presentation
- stakeholder communication
- interview walkthroughs.
