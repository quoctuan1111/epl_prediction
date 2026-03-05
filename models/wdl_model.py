"""
wdl_model.py  (enhanced)
========================
Week 3+ — Win / Draw / Lose multiclass classification with fine-tuning.

Pipeline:
  1.  Load featured_data.csv  (now with ~26 features)
  2.  Drop NaN rows
  3.  Time-series split: train ≤ 2022-23, test = 2023-24
  4.  Hyperparameter search for XGBoost (RandomizedSearchCV, TimeSeriesSplit)
  5.  Train four models: LR, Random Forest, Tuned XGBoost, Stacking Ensemble
  6.  Evaluate: Accuracy, F1, Brier, Log-Loss, ROC-AUC, Confusion Matrix, Calibration
  7.  Save best model → output/wdl_best_model.pkl

Target   : FTR  →  0=A, 1=D, 2=H
Features : rolling, shot-based, form, strength, H2H, Elo, rest

Usage:
    python models/wdl_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, brier_score_loss,
    log_loss, roc_auc_score
)
from sklearn.calibration      import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection  import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

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

# Core features (original 13)
CORE_FEATURES = [
    "home_scored_roll5",   "home_conceded_roll5",
    "away_scored_roll5",   "away_conceded_roll5",
    "home_pts_roll5",      "away_pts_roll5",
    "home_gd_roll5",       "away_gd_roll5",
    "home_elo_before",     "away_elo_before",  "elo_diff",
    "home_days_rest",      "away_days_rest",
]

# New enhanced features (added in feature_engineering.py v2)
ENHANCED_FEATURES = [
    "home_shots_roll5", "home_shots_on_target_roll5",
    "away_shots_roll5", "away_shots_on_target_roll5",
    "home_form3", "away_form3",
    "home_win_streak", "away_win_streak",
    "home_attack_str", "away_attack_str",
    "home_defence_str", "away_defence_str",
    "h2h_home_win_rate",
]

TARGET_COL = "Result_encoded"


# ---------------------------------------------------------------------------
# 1. Load & split
# ---------------------------------------------------------------------------

def load_and_split(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])

    # Use all available features (fall back gracefully if some are missing)
    available  = [c for c in CORE_FEATURES + ENHANCED_FEATURES if c in df.columns]
    missing    = [c for c in CORE_FEATURES + ENHANCED_FEATURES if c not in df.columns]
    if missing:
        print(f"  ⚠  Missing feature columns (skipped): {missing}")

    feature_cols = available
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    print(f"  Rows after dropping NaN : {len(df):,}")
    print(f"  Features used           : {len(feature_cols)}  {feature_cols}")

    train = df[df["Date"] <= TRAIN_END].copy()
    test  = df[df["Season"] == TEST_SEASON].copy()

    X_train = train[feature_cols].values
    y_train = train[TARGET_COL].values.astype(int)
    X_test  = test[feature_cols].values
    y_test  = test[TARGET_COL].values.astype(int)

    print(f"  Train: {len(train):,} matches  |  Test: {len(test):,} matches")
    print(f"  Train period: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"  Test  period: {test['Date'].min().date()} → {test['Date'].max().date()}")
    return X_train, y_train, X_test, y_test, train, test, feature_cols


# ---------------------------------------------------------------------------
# 2. Hyperparameter search for XGBoost
# ---------------------------------------------------------------------------

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 30) -> tuple:
    """
    RandomizedSearchCV with TimeSeriesSplit — finds good XGBoost hyperparameters
    without leaking future data into the CV folds.

    Returns
    -------
    (fitted_xgb, best_params_dict)
      fitted_xgb  : XGBClassifier fitted with early stopping on a val slice
      best_params : dict of raw hyperparameters (no early_stopping_rounds)
                    safe to use inside StackingClassifier
    """
    print(f"\n  Running RandomizedSearch ({n_iter} iterations, 5-fold TimeSeriesCV) …")

    param_dist = {
        "n_estimators":      randint(200, 700),
        "max_depth":         randint(3, 7),
        "learning_rate":     uniform(0.01, 0.09),
        "subsample":         uniform(0.6, 0.4),         # 0.6 – 1.0
        "colsample_bytree":  uniform(0.5, 0.5),         # 0.5 – 1.0
        "reg_alpha":         uniform(0.0, 1.0),
        "reg_lambda":        uniform(0.5, 2.0),
        "min_child_weight":  randint(1, 10),
        "gamma":             uniform(0.0, 0.5),
    }

    base_xgb = XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, verbosity=0, n_jobs=-1,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        base_xgb, param_dist,
        n_iter=n_iter, cv=tscv,
        scoring="neg_log_loss",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(f"  Best params  : {best_params}")
    print(f"  Best CV score (neg-log-loss): {search.best_score_:.4f}")

    # Refit on full train with best params + early stopping on a held-out val slice.
    # early_stopping_rounds is intentionally NOT stored in best_params so the stacking
    # version (which has no eval_set) can work cleanly.
    best_xgb = XGBClassifier(
        **best_params,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", use_label_encoder=False,
        early_stopping_rounds=40,
        random_state=42, verbosity=0,
    )
    val_cut  = int(len(X_train) * 0.90)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]
    best_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    return best_xgb, best_params


# ---------------------------------------------------------------------------
# 3. Build all models
# ---------------------------------------------------------------------------

def build_models(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 30) -> dict:
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=1000, C=1.0, random_state=42
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    print("\n  ── Tuning XGBoost ─────────────────────────────────")
    xgb_tuned, best_params = tune_xgboost(X_train, y_train, n_iter=n_iter)

    # XGBoost version for stacking — same hyper-params but NO early_stopping_rounds
    # (StackingClassifier does its own internal CV and never passes an eval_set)
    xgb_for_stack = XGBClassifier(
        **best_params,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42, verbosity=0,
    )

    # Stacking: base estimators → meta logistic regression
    stacking = Pipeline([
        ("scaler", StandardScaler()),
        ("stack",  StackingClassifier(
            estimators=[
                ("lr",  LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                           max_iter=500, C=1.0, random_state=42)),
                ("rf",  RandomForestClassifier(n_estimators=300, max_depth=7,
                                               min_samples_leaf=10,
                                               class_weight="balanced",
                                               random_state=42, n_jobs=-1)),
                ("xgb", xgb_for_stack),
            ],
            final_estimator=LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=500, C=0.5, random_state=42
            ),
            passthrough=False, cv=5, n_jobs=-1,
        )),
    ])

    return {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "XGBoost (tuned)":    xgb_tuned,
        "Stacking Ensemble":  stacking,
    }


# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")
    ll   = log_loss(y_true, y_prob)
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), y_prob[:, c])
        for c in range(3)
    ])
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (wtd)  : {f1:.4f}")
    print(f"  Log-Loss  : {ll:.4f}   (lower = better)")
    print(f"  Brier     : {brier:.4f}  (lower = better)")
    print(f"  ROC-AUC   : {auc:.4f}   (macro one-vs-rest)")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Away','Draw','Home'])}")

    return {
        "name": name, "accuracy": acc, "f1": f1,
        "log_loss": ll, "brier": brier, "auc": auc,
        "y_pred": y_pred, "y_prob": y_prob
    }


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(results: list, y_test: np.ndarray, out_dir: str):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]
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
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Calibration Plots — WDL Models (Home class)", fontsize=14, fontweight="bold")
    class_idx = 2  # Home win

    for ax, r in zip(axes, results):
        if r["y_prob"] is not None:
            prob_true, prob_pred = calibration_curve(
                y_test == class_idx,
                r["y_prob"][:, class_idx],
                n_bins=8
            )
            ax.plot(prob_pred, prob_true, marker="o", label=r["name"], color="#4C72B0")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Fraction positive")
            ax.set_title(r["name"])
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "wdl_calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Calibration plot saved → {path}")


def save_summary_bar(results: list, out_dir: str):
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] for r in results]
    f1s   = [r["f1"] for r in results]
    aucs  = [r["auc"] for r in results]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w,   accs, w, label="Accuracy",     color="#4C72B0")
    ax.bar(x,       f1s,  w, label="F1 (weighted)", color="#DD8452")
    ax.bar(x + w,   aucs, w, label="ROC-AUC",       color="#55A868")

    ax.set_ylim(0.3, 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=10, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("WDL Model Comparison — 2023/24 Test Season", fontweight="bold")
    ax.legend()
    ax.axhline(0.55, color="green", linestyle="--", linewidth=1, label="Good (55%)")
    ax.legend(loc="lower right")

    for bars, vals in [(x - w, accs), (x, f1s), (x + w, aucs)]:
        for xi, v in zip(bars, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "wdl_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Model comparison chart saved → {path}")


def save_feature_importance(model, feature_cols: list, out_dir: str):
    """Only works for XGBoost (has feature_importances_)."""
    try:
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]

        fig, ax = plt.subplots(figsize=(10, max(5, len(feature_cols) * 0.35)))
        bars = ax.barh(
            [feature_cols[i] for i in idx][::-1],
            imp[idx][::-1],
            color="#4C72B0"
        )
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title("XGBoost Feature Importance", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, "wdl_feature_importance.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✓ Feature importance chart saved → {path}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,   exist_ok=True)

    print("=" * 58)
    print("WDL MODEL TRAINING  (enhanced + fine-tuned)")
    print("=" * 58)

    # ── Load ──────────────────────────────────────────────────
    print("\n[1/4] Loading & splitting data …")
    X_train, y_train, X_test, y_test, train_df, test_df, feat_cols = load_and_split(INPUT_PATH)

    # ── Train ─────────────────────────────────────────────────
    print("\n[2/4] Training & tuning models …")
    models  = build_models(X_train, y_train, n_iter=30)
    results = []

    for name, model in models.items():
        print(f"\n  → Fitting {name} …")
        # XGBoost was already fitted with early stopping inside tune_xgboost()
        if name != "XGBoost (tuned)":
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        results.append(evaluate(name, y_test, y_pred, y_prob))
        results[-1]["model"] = model

    # ── Evaluation charts ─────────────────────────────────────
    print("\n[3/4] Generating evaluation charts …")
    save_confusion_matrix(results, y_test, REPORTS_DIR)
    save_calibration_plot(results, y_test, REPORTS_DIR)
    save_summary_bar(results, REPORTS_DIR)

    # Feature importance for tuned XGBoost
    xgb_result = next((r for r in results if "XGBoost" in r["name"]), None)
    if xgb_result:
        save_feature_importance(xgb_result["model"], feat_cols, REPORTS_DIR)

    # ── Save best model ───────────────────────────────────────
    print("\n[4/4] Saving best model …")
    best = max(results, key=lambda r: r["accuracy"])
    out_path = os.path.join(MODEL_DIR, "wdl_best_model.pkl")
    joblib.dump(best["model"], out_path)

    print(f"\n{'='*58}")
    print(f"  🏆 Best model : {best['name']}")
    print(f"     Accuracy   : {best['accuracy']*100:.2f}%")
    print(f"     F1 (wtd)   : {best['f1']:.4f}")
    print(f"     ROC-AUC    : {best['auc']:.4f}")
    print(f"     Brier      : {best['brier']:.4f}")
    print(f"  Saved → {out_path}")

    # ── Final leaderboard ─────────────────────────────────────
    print(f"\n  {'Model':<25} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Brier':>8} {'LogLoss':>9}")
    print(f"  {'─'*68}")
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        star = " ⭐" if r["name"] == best["name"] else ""
        print(
            f"  {r['name']:<25} "
            f"{r['accuracy']*100:>7.2f}% "
            f"{r['f1']:>8.4f} "
            f"{r['auc']:>8.4f} "
            f"{r['brier']:>8.4f} "
            f"{r['log_loss']:>9.4f}"
            f"{star}"
        )
    print(f"{'='*58}")
