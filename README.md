# ⚽ World Cup 2026 Forecasting Engine

A production-style football forecasting system that combines machine learning match prediction with Monte Carlo tournament simulation to estimate advancement and championship probabilities for international tournaments.

This project demonstrates an end-to-end sports analytics pipeline including:

- feature engineering from historical match data
- probabilistic match outcome modeling
- tournament simulation engine
- large-scale Monte Carlo forecasting
- analytics reporting and visualization

The system is designed to replicate the type of forecasting methodology used by organizations such as FiveThirtyEight, Opta, and sports betting analytics teams.

---

# 🧠 Project Objective

Estimate the probability that each national team:

- advances from the group stage
- reaches the quarterfinals
- reaches the semifinals
- reaches the final
- wins the tournament

This is achieved by simulating thousands of full tournaments using a trained match outcome model.

---

# 🏗 System Architecture

The project follows a modular, production-style architecture.

match dataset
↓
feature engineering pipeline
↓
ML match outcome model
↓
predict_match(team_a, team_b)
↓
Monte Carlo tournament simulation
↓
N simulated tournaments
↓
probability aggregation
↓
forecast outputs

---

# 📊 Data Pipeline

The system uses historical international match data to construct team strength features.

### Input Data

Historical international matches including:

- match results
- teams
- match dates
- tournaments
- goals scored

### Feature Engineering

For each national team the pipeline builds rolling metrics such as:

- Elo rating
- rolling goals scored
- rolling goals conceded
- rolling goal difference
- rolling win rate
- rolling points

These features are stored in:

`data/processed/latest_team_features.parquet`

which represents the most recent team strength snapshot used for simulation.

---

# 🤖 Match Outcome Model

The match prediction model estimates:

P(win)
P(draw)
P(loss)

from the perspective of team A.

Current baseline model:

Logistic Regression


Input features include:

- Elo difference
- rolling performance metrics
- recent goal differential
- win rate indicators

The model outputs calibrated probabilities used directly by the simulation engine.

---

# 🎲 Tournament Simulation Engine

The tournament simulator transforms match probabilities into full tournament forecasts.

Core idea:

predict_match(team_a, team_b)
→ probabilities

↓

sample outcome

↓

simulate entire tournament

↓

repeat N times


The system simulates:

- group stage
- knockout rounds
- tournament champion

---

# 🏆 Monte Carlo Forecasting

The engine runs thousands of simulated tournaments:

10,000 – 100,000 tournament simulations

Each simulation produces:

- group standings
- qualified teams
- knockout progression
- finalists
- champion

Results are aggregated into probabilities for each team.

Example output:

| Team | Group Adv | QF | SF | Final | Champion |
|-----|-----|-----|-----|-----|-----|
| Spain | 89% | 43% | 33% | 21% | 11% |
| Argentina | 88% | 41% | 31% | 20% | 10% |
| France | 83% | 37% | 25% | 16% | 8% |

---

# 📁 Project Structure

```bash
world-cup-2026-forecast
│
├─ app
│ └─ streamlit_app.py
│
├─ configs
│ └─ world_cup_groups.json
│
├─ artifacts
│ ├─ models
│ └─ simulations
│
├─ data
│ ├─ raw
│ ├─ interim
│ ├─ processed
│ └─ outputs
│
├─ src
│ ├─ models
│ │ └─ match_outcome
│ │
│ └─ simulation
│ ├─ config.py
│ ├─ structures.py
│ ├─ predictor_adapter.py
│ ├─ sampling.py
│ ├─ group_stage.py
│ ├─ knockout_stage.py
│ ├─ tournament.py
│ ├─ aggregation.py
│ └─ reporting.py
│
└─ notebooks
```

---

# ▶ Running the Simulation

### Run tournament simulations

`py -m src.simulation.run_simulation`
`--groups-path configs/world_cup_groups.json`
`--num-simulations 10000`

Example output:

TOP TEAMS BY CHAMPION PROBABILITY

Spain 22.5%
Argentina 21.0%
Colombia 9.4%
France 6.9%
England 6.8%

Simulation artifacts are exported to:

`data/outputs/simulation`

including:

`team_probabilities.csv`
`champion_distribution.csv`
`match_logs.parquet`
`summary_metadata.json`

---

# 📈 Dashboard

A Streamlit dashboard is included for exploring the forecast.

Run: `streamlit run app/streamlit_app.py`


The dashboard shows:

- champion probabilities
- advancement probabilities
- team strength comparisons
- simulation outputs

---

# ⚠ Current Limitations

This version intentionally simplifies several aspects of real tournaments.

### 1. No explicit goal model

Matches simulate win/draw/loss outcomes only.

Future version: Poisson goal model

---

### 2. Simplified knockout tie resolution

Draws in knockout rounds are resolved using simplified rules rather than explicit extra time modeling.

---

### 3. Simplified tournament format

The current implementation uses a classic:

8 groups × 4 teams


rather than the full 48-team format planned for the 2026 World Cup.

---

# 🚀 Future Improvements

Planned improvements include:

- Poisson goal scoring model
- expected goals (xG) features
- player-level strength features
- parallelized simulation engine
- real 2026 tournament format
- scenario simulation (injuries, squad changes)

---

# 🎯 Why This Project

This project demonstrates the ability to build **end-to-end sports analytics systems**, including:

- feature engineering pipelines
- probabilistic modeling
- simulation architecture
- scalable forecasting
- analytics dashboards

These techniques are directly applicable to:

- football club analytics departments
- sports data companies
- betting analytics teams
- performance analysis groups

---

# 👤 Author

Manuel Pérez Bañuls — Data Scientist focused on football analytics, predictive modeling, and performance analysis.

---

# 📜 License

MIT License

