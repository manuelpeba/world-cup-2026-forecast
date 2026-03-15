# ⚽ World Cup 2026 Forecasting Engine

A production-style football forecasting system that combines **machine
learning match prediction** with **Monte Carlo tournament simulation**
to estimate advancement and championship probabilities for international
tournaments.

This project demonstrates a full **sports analytics forecasting
pipeline**, including:

-   feature engineering from historical match data
-   probabilistic match outcome modeling
-   tournament simulation engine
-   large-scale Monte Carlo forecasting
-   analytics reporting and visualization
-   interactive dashboard

The system is inspired by forecasting methodologies used by
organizations such as **FiveThirtyEight, Opta, and sports betting
analytics teams**.

---

# 🧠 Project Objective

Estimate the probability that each national team:

-   advances from the group stage
-   reaches the Round of 16
-   reaches the quarterfinals
-   reaches the semifinals
-   reaches the final
-   wins the tournament

This is achieved by **simulating thousands of complete tournaments**
using a trained match outcome model.

---

# 🏗 System Architecture

The forecasting pipeline follows a modular architecture:

historical match data\
↓\
feature engineering pipeline\
↓\
ML match outcome model\
↓\
predict_match(team_a, team_b)\
↓\
Monte Carlo tournament simulation\
↓\
N simulated tournaments\
↓\
probability aggregation\
↓\
forecast outputs

The system separates **modeling, simulation, and reporting** components
to maintain a clean and extensible architecture.

---

# 🧩 Forecasting System Architecture

The project is organized as a modular forecasting pipeline combining
**data engineering, machine learning, and simulation components**.

``` mermaid
flowchart TD
A[Historical Match Data] --> B[Data Ingestion]
B --> C[Feature Engineering]
C --> D[Team Strength Features]
D --> E[Match Outcome Model]
E --> F[Match Prediction API]
F --> G[Monte Carlo Tournament Simulation]
G --> H[Group Stage Simulation]
H --> I[Knockout Stage Simulation]
I --> J[Tournament Results]
J --> K[Aggregation Layer]
K --> L[Team Advancement Probabilities]
K --> M[Champion Distribution]
K --> N[Match Logs]
L --> O[Simulation Artifacts]
M --> O
N --> O
O --> P[Streamlit Dashboard]
```

The architecture follows a **production-style separation between data
processing, predictive modeling, simulation, and reporting layers**.

---

# ⚙ Tournament Simulation Flow

``` mermaid
flowchart TD
A[Load Team Strength Snapshot] --> B[Initialize Tournament]
B --> C[Simulate Group Matches]
C --> D[Generate Group Tables]
D --> E[Select Qualified Teams]
E --> F[Simulate Knockout Bracket]
F --> G[Quarterfinalists]
G --> H[Semifinalists]
H --> I[Finalists]
I --> J[Champion]
J --> K[Store Simulation Result]
K --> L{More Simulations?}
L -->|Yes| B
L -->|No| M[Aggregate Tournament Statistics]
M --> N[Team Advancement Probabilities]
M --> O[Champion Probability Distribution]
```

---

# 🧠 Component Responsibilities

| Component | Responsibility |
|---|---|
| Data ingestion | Load historical international match results |
| Feature engineering | Build Elo and rolling performance features |
| Match outcome model | Predict win/draw/loss probabilities |
| Simulation engine | Simulate tournaments using probabilistic match outcomes |
| Aggregation layer | Convert simulation results into advancement probabilities |
| Reporting layer | Export artifacts and power dashboard visualizations |
| Dashboard | Interactive exploration of forecast results |

---

# ⚙ Simulation Engine Internals

The tournament simulation engine is composed of several modules:

    run_simulation.py
            ↓
    tournament.py
            ↓
    group_stage.py
    knockout_stage.py
            ↓
    aggregation.py
            ↓
    reporting outputs

Key outputs include:

-   team advancement probabilities
-   champion probability distribution
-   simulated match logs
-   simulation metadata

---

# 📊 Data Pipeline

The system uses historical international football match data to
construct **team strength features**.

## Input Data

Historical international matches including:

-   match results
-   teams
-   match dates
-   tournaments
-   goals scored

## Feature Engineering

For each national team the pipeline builds rolling metrics such as:

-   Elo rating
-   rolling goals scored
-   rolling goals conceded
-   rolling goal difference
-   rolling win rate
-   rolling points

These features represent current team strength and are stored in:

    data/processed/latest_team_features.parquet

This snapshot is used as the starting point for tournament simulation.

---

# 🤖 Match Outcome Model

The match prediction model estimates:

    P(win)
    P(draw)
    P(loss)

from the perspective of **team A**.

## Baseline Model

Current implementation:

**Multiclass Logistic Regression**

## Input Features

Examples include:

-   Elo difference
-   rolling performance metrics
-   recent goal differential
-   win rate indicators

The model outputs probability distributions that feed directly into the
simulation engine.

---

# 🎲 Tournament Simulation Engine

The tournament simulator transforms match probabilities into full
tournament forecasts.

Core logic:

    predict_match(team_a, team_b)
            ↓
    probability distribution
            ↓
    sample match outcome
            ↓
    simulate tournament
            ↓
    repeat N times

Each simulation generates:

-   group standings
-   qualified teams
-   knockout progression
-   finalists
-   champion

---

# 🏆 Monte Carlo Forecasting

The engine runs thousands of simulated tournaments.

Typical runs:

    10,000 – 100,000 simulations

Simulation outputs are aggregated into probabilities.

### Example Forecast Output

| Team | Advance from Group | Round of 16 | Quarterfinal | Semifinal | Final | Champion |
|---|---:|---:|---:|---:|---:|---:|
| Spain | 88.8% | 57.7% | 43.0% | 33.4% | 23.1% | 23.1% |
| Argentina | 88.5% | 56.8% | 41.2% | 30.7% | 20.8% | 20.8% |
| France | 85.2% | 52.3% | 37.5% | 26.9% | 15.6% | 9.4% |


These probabilities are produced by aggregating thousands of simulated
tournaments.

---

# 📁 Project Structure

```bash
    world-cup-2026-forecast
    │
    ├── app
    │   └── streamlit_app.py
    │
    ├── configs
    │
    ├── data
    │   ├── raw
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   └── outputs
    │
    ├── docs
    │
    ├── notebooks
    │
    ├── src
    │   ├── dashboard
    │   ├── evaluation
    │   ├── features
    │   ├── ingestion
    │   ├── models
    │   ├── pipelines
    │   ├── simulation
    │   └── utils
    │
    ├── tests
    │
    ├── pyproject.toml
    ├── requirements.txt
    └── README.md
```

---

# ▶ Running the Simulation

### Classic Tournament (32 teams)

``` bash
py -m src.simulation.run_simulation \
  --groups-path configs/world_cup_groups.json \
  --num-simulations 10000
```

### World Cup 2026 (48 teams)

``` bash
py -m src.simulation.run_simulation \
  --groups-path configs/world_cup_groups_48.json \
  --bracket-config-path configs/world_cup_2026_bracket.json \
  --num-simulations 10000 \
  --simulation-format v2
```

### Parameters

| Parameter | Description |
|---|---|
| `--groups-path` | JSON group stage configuration |
| `--bracket-config-path` | Knockout bracket configuration |
| `--num-simulations` | Number of Monte Carlo tournament simulations |
| `--simulation-format` | `v1` (32 teams) or `v2` (48 teams) |


# 📦 Simulation Outputs

Simulation artifacts are exported to:

    data/outputs/simulation

Generated files:

-   team_probabilities.csv
-   champion_distribution.csv
-   match_logs.parquet
-   summary_metadata.json

These artifacts are later consumed by the **Streamlit dashboard**.

---

# 📈 Dashboard

A Streamlit dashboard allows interactive exploration of forecast
results.

Run:

``` bash
streamlit run app/streamlit_app.py
```

The dashboard provides:

-   champion probability rankings
-   advancement probability tables
-   team explorer
-   probability charts
-   simulation match logs

----

# 📓 Research Notebooks

The project includes research notebooks used during development.

| Notebook | Purpose |
|---|---|
| `00_eda_match_dataset.ipynb` | Exploratory data analysis |
| `01_match_model_experiments.ipynb` | Model experimentation |
| `02_simulation_analysis.ipynb` | Simulation analysis |
| `03_world_cup_forecast_story.ipynb` | Forecast storytelling |


---

# ⚠ Current Limitations

This version simplifies several aspects of real tournaments.

### No explicit goal model

Matches simulate win/draw/loss outcomes only.

Future improvement: **Poisson goal model**

### Simplified knockout tie resolution

Draws in knockout rounds are resolved using simplified rules rather than
modeling extra time.

---

# 🚀 Future Improvements

Potential extensions include:

-   Poisson goal scoring models
-   expected goals (xG) features
-   player-level strength models
-   scenario simulations (injuries, squad changes)
-   distributed simulation engine

---

# 🎯 Why This Project

This project demonstrates the ability to build **end-to-end sports
analytics systems**, including:

-   feature engineering pipelines
-   probabilistic modeling
-   simulation architecture
-   large-scale forecasting
-   analytics dashboards

These techniques are directly applicable to:

-   football club analytics departments
-   sports data companies
-   betting analytics teams
-   performance analysis groups

## Skills Demonstrated

| Area | What the project shows |
|---|---|
| Data Engineering | Structured ingestion, feature pipelines, reproducible outputs |
| Machine Learning | Probabilistic classification for football match outcomes |
| Simulation | Monte Carlo tournament engine with configurable formats |
| Analytics Engineering | Aggregation layers, artifact exports, forecast tables |
| Product Thinking | Interactive Streamlit dashboard for stakeholder exploration |
| Sports Analytics | Football-specific modeling, tournament logic, forecasting use cases |

---

# 👤 Author

Manuel Pérez Bañuls\
Data Scientist focused on **football analytics, predictive modeling, and
performance analysis**.

---

# 📜 License

MIT License
