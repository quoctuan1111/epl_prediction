"""
wdl_model.py
============
Week 3 — Win / Draw / Lose multiclass classification.

Pipeline:
  1. Load featured_data.csv
  2. Drop NaN rows (first games without rolling history)
  3. Time-series split: train = up to end 2022-23, test = 2023-24
  4. Train three models: Logistic Regression, Random Forest, XGBoost
  5. Evaluate: Accuracy, F1 (weighted), Confusion Matrix, Calibration
  6. Save best model to models/wdl_best_model.pkl

Target   : FTR  →  encoded as 0=A, 1=D, 2=H  (Result_encoded column)
Features : all rolling, Elo, days-rest columns

Usage:
    python models/wdl_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)
from sklearn.calibration     import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

INPUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "featured_data.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "output")
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "reports")

TRAIN_END    = "2023-05-31"   # last day of 2022-23 season
TEST_SEASON  = "2023_2024"    # held-out test season
LABEL_MAP    = {0: "Away", 1: "Draw", 2: "Home"}

FEATURE_COLS = [
    "home_scored_roll5",   "home_conceded_roll5",
    "away_scored_roll5",   "away_conceded_roll5",
    "home_pts_roll5",      "away_pts_roll5",
    "home_gd_roll5",       "away_gd_roll5",
    "home_elo_before",     "away_elo_before",  "elo_diff",
    "home_days_rest",      "away_days_rest",
]
TARGET_COL = "Result_encoded"


# ---------------------------------------------------------------------------
# 1. Load & split
# ---------------------------------------------------------------------------

def load_and_split(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])

    # Drop rows where any feature is NaN (first games per team)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"  Rows after dropping NaN: {len(df):,}")

    train = df[df["Date"] <= TRAIN_END].copy()
    test  = df[df["Season"] == TEST_SEASON].copy()

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values.astype(int)
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET_COL].values.astype(int)

    print(f"  Train: {len(train):,} matches  |  Test: {len(test):,} matches")
    print(f"  Train period: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"  Test  period: {test['Date'].min().date()} → {test['Date'].max().date()}")
    return X_train, y_train, X_test, y_test, train, test


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------

def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=1000, C=1.0, random_state=42
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42, verbosity=0,
            early_stopping_rounds=30,
        ),
    }


# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(f"\n  {'─'*45}")
    print(f"  {name}")
    print(f"  {'─'*45}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (wtd) : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Away','Draw','Home'])}")
    return {"name": name, "accuracy": acc, "f1": f1, "y_pred": y_pred, "y_prob": y_prob}


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(results: list, y_test: np.ndarray, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Confusion Matrices — WDL Models", fontsize=14, fontweight="bold")
    labels = ["Away", "Draw", "Home"]

    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels, linewidths=0.5)
        ax.set_title(f"{r['name']}\nAcc={r['accuracy']:.3f}  F1={r['f1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(out_dir, "wdl_confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  ✓ Confusion matrices saved → {path}")


def save_calibration_plot(results: list, y_test: np.ndarray, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Calibration Plots — WDL Models (Home class)", fontsize=14, fontweight="bold")
    class_idx = 2  # Home win

    for ax, r in zip(axes, results):
        if r["y_prob"] is not None:
            prob_true, prob_pred = calibration_curve(
                y_test == class_idx,
                r["y_prob"][:, class_idx],
                n_bins=10
            )
            ax.plot(prob_pred, prob_true, marker="o", label=r["name"], color="#4C72B0")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(r["name"])
            ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "wdl_calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Calibration plot saved → {path}")


def save_summary_bar(results: list, out_dir: str):
    names  = [r["name"] for r in results]
    accs   = [r["accuracy"] for r in results]
    f1s    = [r["f1"] for r in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_a = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0")
    bars_f = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color="#DD8452")

    ax.set_ylim(0.3, 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("WDL Model Comparison — 2023/24 Test Season", fontweight="bold")
    ax.legend()
    ax.axhline(0.55, color="green",  linestyle="--", linewidth=1, label="Good (55%)")
    ax.axhline(0.40, color="orange", linestyle="--", linewidth=1, label="Random baseline")
    ax.legend(loc="lower right")

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "wdl_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Model comparison chart saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("WDL MODEL TRAINING — WEEK 3")
    print("=" * 55)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_split(INPUT_PATH)

    # ── Train ─────────────────────────────────────────────
    print("\n[2/4] Training models …")
    models  = build_models()
    results = []

    for name, model in models.items():
        print(f"\n  → {name} …")
        if name == "XGBoost":
            # Split a small validation set from end of train for early stopping
            val_cut = int(len(X_train) * 0.85)
            X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
            y_tr, y_val = y_train[:val_cut], y_train[val_cut:]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        results.append(evaluate(name, y_test, y_pred, y_prob))
        results[-1]["model"] = model

    # ── Summary ───────────────────────────────────────────
    print("\n[3/4] Generating evaluation charts …")
    save_confusion_matrix(results, y_test, REPORTS_DIR)
    save_calibration_plot(results, y_test, REPORTS_DIR)
    save_summary_bar(results, REPORTS_DIR)

    # ── Save best model ───────────────────────────────────
    print("\n[4/4] Saving best model …")
    best = max(results, key=lambda r: r["accuracy"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "wdl_best_model.pkl")
    joblib.dump(best["model"], out_path)

    print(f"\n{'='*55}")
    print(f"  🏆 Best model : {best['name']}")
    print(f"     Accuracy   : {best['accuracy']*100:.2f}%")
    print(f"     F1 (wtd)   : {best['f1']:.4f}")
    print(f"  Saved → {out_path}")

    # ── Final leaderboard ─────────────────────────────────
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1 (wtd)':>10}")
    print(f"  {'─'*45}")
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        star = " ⭐" if r["name"] == best["name"] else ""
        print(f"  {r['name']:<25} {r['accuracy']*100:>9.2f}% {r['f1']:>10.4f}{star}")
    print(f"{'='*55}")
