# World Cup 2026 Forecast

A football analytics project that forecasts the FIFA World Cup 2026 using a multi-layer predictive modeling framework.

The project combines player-level data, team performance metrics, Bayesian modeling, machine learning, and tournament simulations.

## Modeling Framework

The system is composed of three main modeling layers:

1. **Player Impact Model**
   - Estimates player contribution using club performance and international experience.

2. **Team Strength Model**
   - Bayesian estimation of team attacking and defensive strength.

3. **Match Outcome Model**
   - Machine learning model predicting win/draw/loss probabilities.

The framework is used to simulate **100,000 World Cup tournaments** to estimate:

- probability of reaching each stage
- championship probabilities
- upset likelihood

## Project Structure

data/
raw
interim
processed

src/
ingestion
features
models
simulation

notebooks/
exploratory analysis


## Tech Stack

- Python
- Pandas
- PyMC
- XGBoost / LightGBM
- Monte Carlo simulation
- Streamlit dashboard

## Goal

Build a **recruitment-grade football analytics project** that demonstrates:

- data engineering
- machine learning
- probabilistic modeling
- sports analytics expertise
