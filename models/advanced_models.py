"""
advanced_models.py
==================
Phase 2 — Deep Learning & SHAP Explainability

New models added on top of the existing 26-feature pipeline:
  1. LightGBM  — fast gradient-boosted trees, native SHAP support
  2. MLP       — feed-forward neural network (sklearn MLPClassifier, 3 hidden layers)

Explainability (on LightGBM):
  - SHAP global summary plot (beeswarm)
  - SHAP global bar chart
  - SHAP dependence plots  (top 3 features)
  - SHAP waterfall plots   (3 individual test predictions)

All charts saved to reports/.
Best model (across Phase 1 + Phase 2) saved to output/wdl_best_model.pkl.

Usage:
    python models/advanced_models.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

from sklearn.neural_network   import MLPClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, brier_score_loss,
    log_loss, roc_auc_score
)
from sklearn.calibration      import calibration_curve
import lightgbm as lgb

warnings.filterwarnings("ignore")
shap.initjs()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "featured_data.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "output")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

TRAIN_END   = "2023-05-31"
TEST_SEASON = "2023_2024"
LABELS      = ["Away", "Draw", "Home"]

CORE_FEATURES = [
    "home_scored_roll5",   "home_conceded_roll5",
    "away_scored_roll5",   "away_conceded_roll5",
    "home_pts_roll5",      "away_pts_roll5",
    "home_gd_roll5",       "away_gd_roll5",
    "home_elo_before",     "away_elo_before",  "elo_diff",
    "home_days_rest",      "away_days_rest",
]
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
# 1. Load & split  (same scheme as wdl_model.py)
# ---------------------------------------------------------------------------

def load_and_split(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])
    feat_cols = [c for c in CORE_FEATURES + ENHANCED_FEATURES if c in df.columns]
    df = df.dropna(subset=feat_cols + [TARGET_COL]).reset_index(drop=True)

    train = df[df["Date"] <= TRAIN_END].copy()
    test  = df[df["Season"] == TEST_SEASON].copy()

    X_train = train[feat_cols].values
    y_train = train[TARGET_COL].values.astype(int)
    X_test  = test[feat_cols].values
    y_test  = test[TARGET_COL].values.astype(int)

    print(f"  Rows (no NaN)   : {len(df):,}")
    print(f"  Train           : {len(train):,}  ({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"  Test            : {len(test):,}   ({test['Date'].min().date()} → {test['Date'].max().date()})")
    print(f"  Features used   : {len(feat_cols)}")
    return X_train, y_train, X_test, y_test, feat_cols, test


# ---------------------------------------------------------------------------
# 2. Model builders
# ---------------------------------------------------------------------------

def build_lgb(X_train, y_train):
    """
    LightGBM with early stopping on a 10% held-out slice.
    Returns the fitted Booster and the fitted LGBMClassifier (for sklearn API).
    """
    val_cut = int(len(X_train) * 0.90)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]

    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        class_weight="balanced",
        objective="multiclass",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=-1)]
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    print(f"  LightGBM stopped at iteration {clf.best_iteration_}")
    return clf


def build_mlp(X_train, y_train):
    """
    MLP with 3 hidden layers (256→128→64), ReLU, early stopping on internal
    validation split, wrapped in a StandardScaler pipeline.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,           # L2 regularisation
            learning_rate_init=5e-4,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )),
    ])
    pipe.fit(X_train, y_train)
    n_iter = pipe.named_steps["mlp"].n_iter_
    print(f"  MLP converged at iteration {n_iter}")
    return pipe


# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------

def evaluate(name: str, y_true, y_pred, y_prob) -> dict:
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="weighted")
    ll    = log_loss(y_true, y_prob)
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), y_prob[:, c])
        for c in range(3)
    ])
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    print(f"\n  {'─'*52}")
    print(f"  {name}")
    print(f"  {'─'*52}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (wtd)  : {f1:.4f}")
    print(f"  Log-Loss  : {ll:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=LABELS)}")

    return dict(name=name, accuracy=acc, f1=f1,
                log_loss=ll, brier=brier, auc=auc,
                y_pred=y_pred, y_prob=y_prob)


# ---------------------------------------------------------------------------
# 4. SHAP explainability  (LightGBM)
# ---------------------------------------------------------------------------

def run_shap(model: lgb.LGBMClassifier, X_test: np.ndarray,
             feat_cols: list, test_df: pd.DataFrame, out_dir: str):
    """
    Generate SHAP analysis for LightGBM.

    SHAP 0.46+ / LightGBM v4 API:
      TreeExplainer.shap_values() returns ndarray shaped
        (n_samples, n_features, n_classes)   ← 3-D
      Access per-class slice: shap_values[:, :, class_idx]
    """
    print("\n  Computing SHAP values …")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)          # (n_samples, n_features, n_classes)
    n_classes   = shap_values.shape[2]

    # Mean |SHAP| across all samples and classes → shape (n_features,)
    mean_abs   = np.abs(shap_values).mean(axis=(0, 2))
    feat_array = np.array(feat_cols)
    top3_idx   = np.argsort(mean_abs)[::-1][:3]

    home_class = min(2, n_classes - 1)   # Home-win class index (2)

    # ── 4a. Beeswarm (Home-win class) ────────────────────────
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values[:, :, home_class],   # (n_samples, n_features)
        X_test,
        feature_names=feat_cols,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Summary — Home Win class (LightGBM)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ SHAP beeswarm saved → {path}")

    # ── 4b. Bar chart (mean |SHAP|, all classes) ─────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_idx = np.argsort(mean_abs)
    ax.barh(feat_array[sorted_idx], mean_abs[sorted_idx], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (all classes)")
    ax.set_title("SHAP Feature Importance — LightGBM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ SHAP bar chart saved → {path}")

    # ── 4c. Dependence plots (top-3 features, Home class) ────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SHAP Dependence Plots — Home Win class (LightGBM)",
                 fontsize=13, fontweight="bold")
    for ax, fi in zip(axes, top3_idx):
        shap.dependence_plot(
            fi,
            shap_values[:, :, home_class],   # (n_samples, n_features)
            X_test,
            feature_names=feat_cols,
            ax=ax, show=False,
            dot_size=20, alpha=0.6,
        )
        ax.set_title(feat_cols[fi], fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_dependence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ SHAP dependence plots saved → {path}")

    # ── 4d. Waterfall — 3 individual predictions ─────────────
    home_teams = test_df["HomeTeam"].values if "HomeTeam" in test_df.columns else ["Match"] * len(X_test)
    away_teams = test_df["AwayTeam"].values if "AwayTeam" in test_df.columns else [""] * len(X_test)

    probs_test = model.predict_proba(X_test)
    picks = {}
    for c in range(3):
        best_i = int(np.argmax(probs_test[:, c]))
        if best_i not in picks.values():
            picks[LABELS[c]] = best_i
    if len(picks) < 3:
        picks = {LABELS[i]: i for i in range(3)}

    class_names_full = ["Away Win", "Draw", "Home Win"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle("SHAP Waterfall — Individual Match Explanations (LightGBM)",
                 fontsize=13, fontweight="bold")

    for ax, (class_label, idx) in zip(axes, picks.items()):
        class_idx = LABELS.index(class_label) if class_label in LABELS else 2
        class_idx = min(class_idx, n_classes - 1)

        sv       = shap_values[idx, :, class_idx]    # (n_features,)
        order    = np.argsort(np.abs(sv))[::-1][:12]
        sv_ord   = sv[order]
        nm_ord   = [feat_cols[i] for i in order]
        colours  = ["#e74c3c" if v > 0 else "#3498db" for v in sv_ord]
        y_pos    = np.arange(len(nm_ord))

        ax.barh(y_pos, sv_ord[::-1], color=colours[::-1])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nm_ord[::-1], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ht = home_teams[idx] if idx < len(home_teams) else "?"
        at = away_teams[idx] if idx < len(away_teams) else "?"
        ax.set_title(f"Predicted: {class_names_full[class_idx]}\n{ht} vs {at}", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "shap_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ SHAP waterfall plots saved → {path}")

    # ── Summary table ─────────────────────────────────
    print(f"\n  Top-10 features by mean |SHAP| (all classes):")
    for rank, fi in enumerate(np.argsort(mean_abs)[::-1][:10], 1):
        print(f"    {rank:2d}. {feat_cols[fi]:<38s} {mean_abs[fi]:.4f}")


# ---------------------------------------------------------------------------
# 5. Comparison chart vs Phase 1 models
# ---------------------------------------------------------------------------

def save_full_comparison(all_results: list, out_dir: str):
    """Bar chart comparing all 6 models (Phase 1 + Phase 2)."""
    names = [r["name"] for r in all_results]
    accs  = [r["accuracy"] for r in all_results]
    aucs  = [r["auc"] for r in all_results]
    f1s   = [r["f1"] for r in all_results]

    x = np.arange(len(names))
    w = 0.25
    colours_acc  = ["#4C72B0" if "LightGBM" not in n and "MLP" not in n else "#E27F47"
                    for n in names]
    colours_auc  = ["#55A868" if "LightGBM" not in n and "MLP" not in n else "#C44E52"
                    for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w, accs, w, label="Accuracy",      color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x,      aucs, w, label="ROC-AUC",       color="#55A868", alpha=0.85)
    b3 = ax.bar(x + w,  f1s,  w, label="F1 (weighted)", color="#DD8452", alpha=0.85)

    ax.set_ylim(0.3, 0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Full Model Comparison — Phase 1 & Phase 2 (2023-24 Test Season)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.55, color="green",  linestyle="--", linewidth=1, alpha=0.7, label="Good (55%)")
    ax.legend(loc="lower right", fontsize=9)

    for bars, vals in [(b1, accs), (b2, aucs), (b3, f1s)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "full_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Full comparison chart saved → {path}")


def save_confusion_matrices(results: list, y_test, out_dir: str):
    """Phase 2 models only."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — Phase 2 Models", fontsize=13, fontweight="bold")
    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, linewidths=0.5)
        ax.set_title(f"{r['name']}\nAcc={r['accuracy']:.3f}  F1={r['f1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(out_dir, "phase2_confusion_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrices saved → {path}")


def save_calibration(results: list, y_test, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Calibration Plots — Phase 2 Models (Home class)",
                 fontsize=13, fontweight="bold")
    for ax, r in zip(axes, results):
        if r["y_prob"] is not None:
            pt, pp = calibration_curve(y_test == 2, r["y_prob"][:, 2], n_bins=8)
            ax.plot(pp, pt, marker="o", label=r["name"], color="#E27F47")
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
            ax.set_title(r["name"])
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Fraction positive")
            ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "phase2_calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Calibration plots saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,   exist_ok=True)

    print("=" * 60)
    print("ADVANCED MODELS — Phase 2  (LightGBM + MLP + SHAP)")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    X_train, y_train, X_test, y_test, feat_cols, test_df = load_and_split(INPUT_PATH)

    # ── Train ─────────────────────────────────────────────────
    print("\n[2/5] Training LightGBM …")
    lgb_model = build_lgb(X_train, y_train)
    lgb_pred  = lgb_model.predict(X_test)
    lgb_prob  = lgb_model.predict_proba(X_test)
    lgb_result = evaluate("LightGBM", y_test, lgb_pred, lgb_prob)
    lgb_result["model"] = lgb_model

    print("\n[3/5] Training MLP Neural Network …")
    mlp_model = build_mlp(X_train, y_train)
    mlp_pred  = mlp_model.predict(X_test)
    mlp_prob  = mlp_model.predict_proba(X_test)
    mlp_result = evaluate("MLP Neural Network", y_test, mlp_pred, mlp_prob)
    mlp_result["model"] = mlp_model

    p2_results = [lgb_result, mlp_result]

    # ── Evaluation charts ─────────────────────────────────────
    print("\n[4/5] Generating evaluation charts …")
    save_confusion_matrices(p2_results, y_test, REPORTS_DIR)
    save_calibration(p2_results, y_test, REPORTS_DIR)

    # Load Phase 1 results for comparison (dummy metrics — re-evaluate from saved model)
    # We build a combined leaderboard from what we know
    phase1_known = [
        dict(name="LR (Phase 1)",      accuracy=0.5863, f1=0.5170, auc=0.6819, log_loss=0.9352, brier=0.1832, y_pred=None, y_prob=None),
        dict(name="RF (Phase 1)",       accuracy=0.5298, f1=0.5319, auc=0.6665, log_loss=0.9781, brier=0.1935, y_pred=None, y_prob=None),
        dict(name="XGB (Phase 1)",      accuracy=0.5804, f1=0.5195, auc=0.6656, log_loss=0.9570, brier=0.1874, y_pred=None, y_prob=None),
        dict(name="Stacking (Phase 1)", accuracy=0.5833, f1=0.5107, auc=0.6854, log_loss=0.9338, brier=0.1835, y_pred=None, y_prob=None),
    ]
    all_results = phase1_known + [lgb_result, mlp_result]
    save_full_comparison(all_results, REPORTS_DIR)

    # ── SHAP ──────────────────────────────────────────────────
    print("\n[5/5] Running SHAP analysis on LightGBM …")
    run_shap(lgb_model, X_test, feat_cols, test_df, REPORTS_DIR)

    # ── Save best model if improved ───────────────────────────
    current_best_path = os.path.join(MODEL_DIR, "wdl_best_model.pkl")
    phase1_best_acc   = 0.5863   # Logistic Regression from Phase 1
    p2_best = max(p2_results, key=lambda r: r["accuracy"])

    if p2_best["accuracy"] > phase1_best_acc:
        joblib.dump(p2_best["model"], current_best_path)
        print(f"\n  ✅ New best model saved → {current_best_path}")
        print(f"     {p2_best['name']}  Accuracy: {p2_best['accuracy']*100:.2f}%")
    else:
        print(f"\n  ℹ  Phase 1 model still best ({phase1_best_acc*100:.2f}%); pkl unchanged.")

    # ── Final leaderboard ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FULL LEADERBOARD")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Brier':>8}")
    print(f"  {'─'*62}")
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        tag = " 🆕" if r["name"] in ("LightGBM", "MLP Neural Network") else ""
        print(
            f"  {r['name']:<28} "
            f"{r['accuracy']*100:>7.2f}% "
            f"{r['f1']:>8.4f} "
            f"{r['auc']:>8.4f} "
            f"{r['brier']:>8.4f}"
            f"{tag}"
        )
    print(f"{'='*60}")
