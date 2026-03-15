You are helping develop a production-style sports analytics system to forecast the FIFA World Cup 2026.

The project is a full end-to-end football forecasting pipeline combining:

- data engineering
- feature engineering
- probabilistic machine learning
- Monte Carlo tournament simulation
- analytics reporting
- interactive dashboard

The goal is to estimate the probability that each national team:

- advances from the group stage
- reaches the Round of 16
- reaches the quarterfinals
- reaches the semifinals
- reaches the final
- wins the tournament

The system architecture is inspired by forecasting approaches used by:

- FiveThirtyEight
- Opta
- sports betting analytics teams


-------------------------------------
PROJECT REPOSITORY STRUCTURE
-------------------------------------

world-cup-2026-forecast

app/
    streamlit_app.py

configs/
    allowed_teams.yaml
    data.yaml
    model_match.yaml
    model_player.yaml
    model_team.yaml
    simulation.yaml
    world_cup_groups.json
    world_cup_groups_48.json
    world_cup_2026_bracket.json

data/
    raw/
    external/
    interim/
    processed/
        latest_team_features.parquet
    outputs/
        simulation/

docs/
    dashboard_guide.md
    data_dictionary.md
    engineering_notes.md
    methodology.md
    modeling_notes.md

notebooks/
    README.md
    00_eda_match_dataset.ipynb
    01_match_model_experiments.ipynb
    02_simulation_analysis.ipynb
    03_world_cup_forecast_story.ipynb

src/
    dashboard/
    evaluation/
    features/
    ingestion/
    models/
    pipelines/
    simulation/
        run_simulation.py
        tournament.py
        group_stage.py
        knockout_stage.py
        bracket_builder.py
        qualification.py
        aggregation.py
        structures.py
    utils/

tests/

requirements.txt
pyproject.toml
README.md


-------------------------------------
SYSTEM ARCHITECTURE
-------------------------------------

Pipeline:

historical match data
↓
data ingestion
↓
feature engineering
↓
team strength features
↓
match outcome model
↓
predict_match(team_a, team_b)
↓
Monte Carlo tournament simulation
↓
N simulated tournaments
↓
aggregation
↓
forecast probabilities
↓
dashboard visualization


-------------------------------------
MATCH PREDICTION MODEL
-------------------------------------

Current baseline model:

Multiclass Logistic Regression

Predicts:

P(win)
P(draw)
P(loss)

From the perspective of Team A.

Features include:

- Elo rating
- Elo difference
- rolling goals scored
- rolling goals conceded
- rolling goal difference
- rolling win rate
- rolling points
- recent performance metrics

Dataset size:

~31k international matches.

Train/test split:

temporal split.

Model output:

probability distribution used by the simulation engine.


-------------------------------------
TOURNAMENT SIMULATION ENGINE
-------------------------------------

The simulation engine converts match probabilities into full tournament forecasts.

Core idea:

predict_match(team_a, team_b)
↓
sample outcome
↓
simulate match
↓
simulate tournament
↓
repeat N times


Each simulation produces:

- group tables
- qualified teams
- knockout progression
- finalists
- champion


-------------------------------------
TOURNAMENT FORMAT SUPPORT
-------------------------------------

V1

Classic World Cup format:

32 teams
8 groups
Round of 16
Quarterfinals
Semifinals
Final


V2

New FIFA World Cup 2026 format:

48 teams
12 groups
top 2 teams qualify
best 8 third-place teams qualify
Round of 32 knockout stage


Implemented components:

- bracket_builder.py
- qualification.py
- dynamic third-place selection
- configurable bracket JSON


-------------------------------------
SIMULATION OUTPUTS
-------------------------------------

Artifacts exported to:

data/outputs/simulation

Generated files:

team_probabilities.csv
team_probabilities.parquet

champion_distribution.csv
champion_distribution.parquet

match_logs.parquet

summary_metadata.json


team_probabilities contains:

team
advance_from_group_prob
round_of_16_prob
quarterfinal_prob
semifinal_prob
final_prob
champion_prob


-------------------------------------
DASHBOARD
-------------------------------------

Streamlit dashboard:

app/streamlit_app.py

Features:

- champion probability chart
- team probability leaderboard
- team explorer
- match log preview
- simulation metadata display


Dashboard reads:

team_probabilities.csv
champion_distribution.csv
match_logs.parquet
summary_metadata.json


-------------------------------------
RESEARCH NOTEBOOKS
-------------------------------------

00_eda_match_dataset.ipynb

exploratory analysis of match dataset

01_match_model_experiments.ipynb

model experimentation and validation

02_simulation_analysis.ipynb

analysis of Monte Carlo outputs

03_world_cup_forecast_story.ipynb

presentation notebook for portfolio


-------------------------------------
CURRENT PROJECT STATUS
-------------------------------------

Completed:

✓ data pipeline
✓ feature engineering
✓ logistic regression match model
✓ tournament simulation engine
✓ support for 32-team and 48-team formats
✓ Monte Carlo simulations
✓ probability aggregation
✓ simulation artifacts export
✓ Streamlit dashboard
✓ repository documentation
✓ README portfolio documentation


-------------------------------------
NEXT DEVELOPMENT STEPS
-------------------------------------

1️⃣ Complete real tournament configuration

Update:

world_cup_groups_48.json

with actual qualified teams once:

- qualification phase ends
- official group draw is released


2️⃣ Improve match prediction model

Current model:

logistic regression baseline.

Potential improvements:

- Gradient Boosting
- XGBoost
- LightGBM
- Bayesian hierarchical models
- team strength latent models
- time-decay weighting
- cross-validation hyperparameter tuning


3️⃣ Introduce goal-based models

Current simulation uses:

win/draw/loss sampling.

Future improvement:

Poisson goal model:

predict goals scored distribution
simulate full scorelines
derive match outcome from goals.


4️⃣ Improve team strength estimation

Possible improvements:

- Bayesian Elo models
- time-weighted ratings
- player-level strength aggregation
- squad strength modeling


5️⃣ Scenario simulations

Add capability to simulate:

- injuries
- squad changes
- home advantage
- neutral venue variations


6️⃣ Improve dashboard

Possible additions:

- stacked progression chart
- probability evolution plots
- team comparison chart
- tournament bracket visualization


7️⃣ Clean repository structure

Review and remove unused early-project files:

- obsolete experiments
- unused configs
- deprecated scripts
- temporary notebooks


8️⃣ Finalize documentation

Complete documentation files:

docs/engineering_notes.md

Should describe:

- system architecture
- pipeline structure
- module responsibilities
- simulation engine internals


docs/modeling_notes.md

Should describe:

- model training pipeline
- feature design
- evaluation metrics
- model limitations


docs/methodology.md

Should explain:

- Monte Carlo forecasting approach
- tournament simulation logic
- probability aggregation methodology


9️⃣ Improve reproducibility

Add:

- Makefile or CLI scripts
- environment setup guide
- simulation examples


10️⃣ Improve portfolio presentation

Add:

- dashboard screenshots
- example forecast output
- architecture diagrams
- sample tournament simulation results


-------------------------------------
PROJECT PURPOSE
-------------------------------------

This project demonstrates the ability to build a full sports analytics system including:

- data pipelines
- machine learning modeling
- probabilistic forecasting
- simulation engines
- analytics dashboards

These skills are directly relevant for:

- football analytics departments
- sports data companies
- betting analytics teams
- performance analysis groups


-------------------------------------
YOUR ROLE
-------------------------------------

Help improve this forecasting system by:

- refining architecture
- improving the prediction model
- extending the simulation engine
- improving dashboard analytics
- strengthening documentation
- making the project portfolio-ready
