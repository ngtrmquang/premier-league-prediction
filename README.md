# **Premier League Prediction** (Baseline V1)


## How it works

```
history = 2011/12 → 2025/26 (real results)

for matchday in 2026/27 (1 → 38):
    build features (lag) from history
            ↓
    select the fixtures of this matchday
            ↓
    predict FTHG and FTAG (two XGBoost models)
            ↓
    round to integer scores, derive result and points
            ↓
    write the predictions back in the same format as history
            ↓
    next matchday uses them as part of its history
```

After matchday 38, points and goals are aggregated per team to produce the predicted table.

## Notebooks

| Notebook | Purpose |
|---|---|
| `00_dataset_overview.ipynb` | Downloads the 2011/12 → 2025/26 season CSVs from [datahub.io](https://datahub.io/football/english-premier-league), concatenates them, fixes dtypes and saves to `pl_combined.parquet`. Also loads the 2026/27 fixture list (with `Matchday`) and saves it as `season-2627.parquet`. |
| `01_feature_engineering.ipynb` | Builds rolling-form features per team, one-hot encodes team names, and creates the train/validation/predict splits. |
| `02_model_selection.ipynb` | Tunes 2 XGBoost regressor for home goals and away goals using `RandomizedSearchCV` + `TimeSeriesSplit`. |
| `03_predict_final.ipynb` | Refits both models on train + validation, runs the matchday-by-matchday simulation for 2026/27 and aggregates the predicted table. |

## Data

- Historical results: 15 seasons of Premier League matches (2011/12 → 2025/26) from [datahub.io](https://datahub.io/football/english-premier-league).
- 2026/27: a fixture list with a `Matchday` column, results left empty and filled in by the simulation.

## Features

All features use only information from **before** the match (`shift(1)` before the rolling mean), computed per team over the team's last 5 matches:

- Goals Scored, Goals Conceded and Points (all venues): `*_goals_avg_last_5`, `*_conceded_avg_last_5`, `*_points_avg_last_5`
- Venue-specific form: Home Team's last 5 home matches and Away Team's last 5 away matches (goals scored / conceded)
- One-hot encoded `HomeTeam` and `AwayTeam`

The first 5 matches of each team in the dataset have `NaN` features, which XGBoost handles natively.

## Model and setup

- Two separate `XGBRegressor` models (home goals, away goals), objective: squared error
- Train: 2011/12 → 2024/25 
- Validation: 2025/26 
- Final fit: Train + Validation
- Hyperparameter search: 50 random configurations, 5-fold `TimeSeriesSplit`, scored by MAE
- Predicted goals are rounded to integers to form a scoreline

## Results (validation season 2025/26)

| Target | MAE | RMSE |
|---|---|---|
| Home Goals (`FTHG`) | 0.962 | 1.141 |
| Away Goals (`FTHG`) | 1.034 | 1.28 |

The model's predictions are far less spread out than real scores:

| | Actual mean | Actual std | Predicted mean | Predicted std | Predicted max |
|---|---|---|---|---|---|
| Home Goals | 1.53 | 1.17 | 1.55 | 0.39 | 3.22 |
| Away Goals | 1.22 | 1.08 | 1.26 | 0.28 | 2.31 |

> Predicted goals therefore cluster between 1-2

> Results like 5-0 or 4-2 are never produced.


## Roadmap (V2 ideas)

- Poisson objective (or Dixon-Coles style model) and sampling scorelines instead of rounding
- Monte Carlo simulation of the whole season (title, top 4, relegation probabilities)
- One combined model for home and away goals (long format with an `is_home` flag)
- Extra lag features (shots, shots on target, corners), only if they can be simulated consistently in the loop
- Features that carry across seasons with decay, and explicit handling of promoted teams
- Use real results for matchdays already played in 2026/27 and simulate only the rest
- Probabilistic evaluation (log-loss / RPS) against simple baselines
- Add LaLiga


## Running the notebooks

Run them in order, `00 → 03`. Before running, replace the hard-coded local paths (`C:\Users\...`) with your own.
