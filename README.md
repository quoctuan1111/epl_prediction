# ⚽ EPL Match Predictor

A machine-learning pipeline that predicts English Premier League match outcomes (**Win / Draw / Lose**) using historical season data, rolling team stats, Elo ratings, and advanced engineered features.

---

## 📁 Project Structure

```
epl_prediction/
├── data/
│   ├── raw/                    # Season CSVs (2010-11 → 2024-25)
│   └── processed_data/
│       ├── merged_data.csv     # All seasons merged & cleaned
│       └── featured_data.csv   # Final feature-engineered dataset
│
├── data_processing/
│   ├── data_merge.py           # Merge raw CSVs → merged_data.csv
│   └── feature_engineering.py # Engineer features → featured_data.csv
│
├── models/
│   └── wdl_model.py            # Train, tune & evaluate all models
│
├── output/
│   └── wdl_best_model.pkl      # Saved best model
│
├── reports/                    # Generated evaluation charts (.png)
│
├── demo/
│   └── cli_demo.py             # Interactive CLI predictor
│
└── README.md
```

---

## 🔧 Setup

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib scipy
```

---

## 🚀 Usage

### Step 1 — Merge raw data

```bash
python data_processing/data_merge.py
```

### Step 2 — Engineer features

```bash
python data_processing/feature_engineering.py
```

> Takes ~60–90 seconds. Outputs `featured_data.csv` with **26 engineered features**.

### Step 3 — Train & evaluate models

```bash
python models/wdl_model.py
```

> Runs hyperparameter search + trains 4 models. Saves best model to `output/wdl_best_model.pkl` and 4 evaluation charts to `reports/`.

### Step 4 — Run the CLI demo

```bash
python demo/cli_demo.py
```

> Interactive predictor: pick teams, enter days rest, get probabilities + team stats.

---

## 🧠 Features (26 total)

| Group | Features |
|---|---|
| **Rolling (last 5, venue-split)** | Goals scored/conceded, points, goal diff |
| **Shot-based (proxy xG)** | Shots, shots on target (home/away venue) |
| **Form** | Points per game (last 3), win streak |
| **Strength** | Attack & defence strength vs league average |
| **Head-to-Head** | Home win rate in last 6 H2H meetings |
| **Elo** | Home Elo, Away Elo, Elo difference |
| **Rest** | Days since last match (home & away) |

All features are computed with **strict look-back** — no future data is used when building any feature.

---

## 📊 Model Results — 2023-24 Season (336 test matches)

| Model | Accuracy | ROC-AUC | Brier | Log-Loss |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | **58.6%** | 0.682 | 0.183 | 0.935 |
| Stacking Ensemble | 58.3% | **0.685** | 0.184 | **0.934** |
| XGBoost (tuned) | 58.0% | 0.666 | 0.187 | 0.957 |
| Random Forest | 53.0% | 0.667 | 0.194 | 0.978 |

> **XGBoost** is tuned with `RandomizedSearchCV` (30 iterations, `TimeSeriesSplit` 5-fold).  
> **Stacking Ensemble** combines LR + RF + XGBoost with a Logistic Regression meta-learner.

---

## 📈 Evaluation Charts

| Chart | Description |
|---|---|
| `wdl_model_comparison.png` | Accuracy / F1 / ROC-AUC bar chart |
| `wdl_confusion_matrices.png` | Per-model confusion matrix |
| `wdl_calibration.png` | Predicted vs actual probability (Home class) |
| `wdl_feature_importance.png` | XGBoost feature importances |

---

## 🧠 Phase 2 — Deep Learning & Explainability

Run the advanced models pipeline:

```bash
python models/advanced_models.py
```

### New Models

| Model | Config | Accuracy | ROC-AUC |
|---|---|---|---|
| **LightGBM** | 1000 trees, early stop, balanced | 50.9% | 0.647 |
| **MLP Neural Net** | 256→128→64 ReLU, Adam, early stop | 56.3% | 0.677 |

### SHAP Explainability (LightGBM)

SHAP analysis reveals **why** the model makes each prediction:

| Chart | Description |
|---|---|
| `shap_summary_beeswarm.png` | Feature impact on Home Win probability |
| `shap_bar_chart.png` | Global feature importance (mean \|SHAP\|) |
| `shap_dependence.png` | How top features affect predictions |
| `shap_waterfall.png` | Per-match prediction explanations |

**Top predictive features** (by SHAP): `elo_diff` (0.27) → `home_elo_before` (0.10) → `home_shots_roll5` (0.08)

### Full 6-Model Leaderboard (2023-24 test season)

| Model | Accuracy | ROC-AUC | Brier |
|---|---|---|---|
| **LR (Phase 1)** ⭐ | **58.6%** | 0.682 | 0.183 |
| Stacking (Phase 1) | 58.3% | **0.685** | 0.184 |
| XGB tuned (Phase 1) | 58.0% | 0.666 | 0.187 |
| MLP Neural Net 🆕 | 56.3% | 0.677 | 0.187 |
| Random Forest | 53.0% | 0.667 | 0.194 |
| LightGBM 🆕 | 50.9% | 0.647 | 0.198 |


---

## 🗂 Data Source

Season CSV files sourced from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) covering seasons **2010–11 through 2024–25**.
