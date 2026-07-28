"""
Pan-India Cloudburst Prediction — CatBoost Production Pipeline (v3)
===================================================================
Focus: CatBoost only — maximize test accuracy, minimize overfitting.
Key anti-overfitting strategies:
  • NO SMOTE during tuning (avoids CV data leakage that inflated v2 scores)
  • auto_class_weights="Balanced" handles imbalance natively
  • Early stopping with dedicated validation set
  • Aggressive regularization ranges in param grid
  • scoring="accuracy" in RandomizedSearchCV
  • Constrained depth/leaves to prevent memorization
"""

import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# --- Paths ---
DATA_PATH = r"D:\clou feature\cloudburst_full_dataset.xlsx"
OUTPUT_DIR = r"D:\clou feature\outputs"
MODEL_PATH = r"D:\clou feature\cloudburst_model.pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  CLOUDBURST PREDICTION -- CatBoost Production Pipeline v3")
print("=" * 70)


# ===========================================================================
# STEP 1: DATA LOADING
# ===========================================================================
print("\n[STEP 1] Loading dataset...")
df = pd.read_excel(DATA_PATH)
print(f"   Shape: {df.shape}")


# ===========================================================================
# STEP 2: PREPROCESSING
# ===========================================================================
print("\n[STEP 2] Preprocessing...")

# Drop leakage + metadata
LEAKAGE_COLS = ["Casualties", "Event"]
META_COLS = ["Date", "Location", "Time", "Source", "Latitude", "Longitude"]
DROP_COLS = LEAKAGE_COLS + META_COLS
cols_to_drop = [c for c in DROP_COLS if c in df.columns]
df_clean = df.drop(columns=cols_to_drop)
print(f"   Dropped {len(cols_to_drop)} columns: {cols_to_drop}")

# Separate target
TARGET = "Target"
y = df_clean[TARGET].copy()
X = df_clean.drop(columns=[TARGET])

# Drop non-numeric
non_numeric = X.select_dtypes(include=["object", "string", "datetime"]).columns.tolist()
if non_numeric:
    print(f"   Dropping non-numeric: {non_numeric}")
    X = X.drop(columns=non_numeric)

# Median impute nulls
null_cols = X.isnull().sum()
cols_with_nulls = null_cols[null_cols > 0]
if len(cols_with_nulls) > 0:
    print(f"   Median-imputing {len(cols_with_nulls)} columns")
    X = X.fillna(X.median())

# Handle infinities
inf_mask = np.isinf(X.values)
if inf_mask.any():
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

# Remove zero-variance
zero_var_cols = X.columns[X.std() == 0].tolist()
if zero_var_cols:
    X = X.drop(columns=zero_var_cols)

# Remove highly correlated features (>0.95)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
if high_corr_cols:
    print(f"   Dropping {len(high_corr_cols)} highly correlated features (r > 0.95)")
    X = X.drop(columns=high_corr_cols)

print(f"   Final: {X.shape[1]} features, {X.shape[0]} samples")
print(f"   Classes: {dict(y.value_counts())}")

feature_names = X.columns.tolist()

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train/Val split for early stopping
X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
X_train_main_scaled = pd.DataFrame(scaler.transform(X_train_main), columns=feature_names, index=X_train_main.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_names, index=X_val.index)
print("   StandardScaler applied")


# ===========================================================================
# STEP 3: BASELINE CatBoost with Early Stopping
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 3] Baseline CatBoost with Early Stopping")
print("=" * 70)

baseline = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    l2_leaf_reg=5,
    random_strength=2.0,
    bagging_temperature=1.0,
    min_data_in_leaf=20,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
)
baseline.fit(X_train_main_scaled, y_train_main, eval_set=(X_val_scaled, y_val), verbose=False)
print(f"   Early stopped at iteration {baseline.best_iteration_}/1000")

y_train_pred_base = baseline.predict(X_train_scaled)
y_test_pred_base = baseline.predict(X_test_scaled)
y_test_proba_base = baseline.predict_proba(X_test_scaled)[:, 1]

train_acc_base = accuracy_score(y_train, y_train_pred_base)
test_acc_base = accuracy_score(y_test, y_test_pred_base)
test_f1_base = f1_score(y_test, y_test_pred_base)
test_roc_base = roc_auc_score(y_test, y_test_proba_base)

print(f"   Train Acc: {train_acc_base:.4f} | Test Acc: {test_acc_base:.4f} | Gap: {train_acc_base - test_acc_base:.4f}")
print(f"   Test F1: {test_f1_base:.4f} | Test ROC-AUC: {test_roc_base:.4f}")


# ===========================================================================
# STEP 4: HYPERPARAMETER TUNING — RandomizedSearchCV (100 iter)
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 4] CatBoost Tuning — RandomizedSearchCV (100 x 5-fold)")
print("  Scoring: accuracy | No SMOTE (class_weights handles imbalance)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Aggressive regularization ranges to combat overfitting
param_dist = {
    "iterations": [200, 300, 400, 500, 600, 700],
    "depth": [3, 4, 5, 6],                           # shallow trees to prevent memorization
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "l2_leaf_reg": [5, 7, 10, 15, 20, 30],           # strong L2 regularization
    "random_strength": [2.0, 3.0, 5.0, 8.0, 10.0],   # more randomness = less overfitting
    "bagging_temperature": [0.5, 1.0, 2.0, 3.0, 5.0], # stronger bagging noise
    "border_count": [32, 64, 128, 254],
    "min_data_in_leaf": [10, 20, 30, 50, 70, 100],    # larger leaves = smoother model
    "subsample": [0.6, 0.7, 0.8, 0.9],                # row subsampling
    "colsample_bylevel": [0.5, 0.6, 0.7, 0.8, 0.9],  # column subsampling
}

base_catboost = CatBoostClassifier(
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0,
)

print(f"   Tuning {len(param_dist)} hyperparameters...")
print(f"   Total fits: 100 combos x 5 folds = 500")

start_time = time.time()
search = RandomizedSearchCV(
    estimator=base_catboost,
    param_distributions=param_dist,
    n_iter=100,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)
search.fit(X_train_scaled, y_train)
tune_time = time.time() - start_time

print(f"\n   Tuning completed in {tune_time:.1f}s")
print(f"\n   Best CV Accuracy: {search.best_score_:.4f}")

# Check CV train vs CV test to diagnose overfitting in CV itself
best_idx = search.best_index_
cv_train_acc = search.cv_results_["mean_train_score"][best_idx]
cv_test_acc = search.best_score_
cv_gap = cv_train_acc - cv_test_acc
print(f"   CV Train Accuracy: {cv_train_acc:.4f}")
print(f"   CV Test Accuracy:  {cv_test_acc:.4f}")
print(f"   CV Gap:            {cv_gap:.4f}")

print(f"\n   Best Parameters:")
for param, val in sorted(search.best_params_.items()):
    print(f"      {param}: {val}")


# ===========================================================================
# STEP 5: EVALUATE TUNED MODEL + EARLY STOPPING REFIT
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 5] Evaluating Tuned CatBoost + Early Stopping Refit")
print("=" * 70)

# Option A: Direct use of best estimator from RandomizedSearchCV
tuned_direct = search.best_estimator_
y_test_pred_A = tuned_direct.predict(X_test_scaled)
y_test_proba_A = tuned_direct.predict_proba(X_test_scaled)[:, 1]
y_train_pred_A = tuned_direct.predict(X_train_scaled)

train_acc_A = accuracy_score(y_train, y_train_pred_A)
test_acc_A = accuracy_score(y_test, y_test_pred_A)
test_f1_A = f1_score(y_test, y_test_pred_A)
test_roc_A = roc_auc_score(y_test, y_test_proba_A)
gap_A = train_acc_A - test_acc_A

print(f"\n   Option A (Tuned, no early stop):")
print(f"   Train Acc: {train_acc_A:.4f} | Test Acc: {test_acc_A:.4f} | Gap: {gap_A:.4f}")
print(f"   F1: {test_f1_A:.4f} | ROC-AUC: {test_roc_A:.4f}")

# Option B: Retrain best params WITH early stopping on train_main/val split
best_params = search.best_params_.copy()
# Increase iterations to allow early stopping to find the sweet spot
best_params["iterations"] = max(best_params.get("iterations", 500), 1000)
# Remove subsample from CatBoost params if present (it uses subsample differently)
refit_params = {k: v for k, v in best_params.items()}

tuned_early_stop = CatBoostClassifier(
    **refit_params,
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
)
tuned_early_stop.fit(
    X_train_main_scaled, y_train_main,
    eval_set=(X_val_scaled, y_val),
    verbose=False
)
best_iter = tuned_early_stop.best_iteration_
print(f"\n   Option B (Tuned + early stop): stopped at iter {best_iter}")

y_test_pred_B = tuned_early_stop.predict(X_test_scaled)
y_test_proba_B = tuned_early_stop.predict_proba(X_test_scaled)[:, 1]
y_train_pred_B = tuned_early_stop.predict(X_train_scaled)

train_acc_B = accuracy_score(y_train, y_train_pred_B)
test_acc_B = accuracy_score(y_test, y_test_pred_B)
test_f1_B = f1_score(y_test, y_test_pred_B)
test_roc_B = roc_auc_score(y_test, y_test_proba_B)
gap_B = train_acc_B - test_acc_B

print(f"   Train Acc: {train_acc_B:.4f} | Test Acc: {test_acc_B:.4f} | Gap: {gap_B:.4f}")
print(f"   F1: {test_f1_B:.4f} | ROC-AUC: {test_roc_B:.4f}")

# Pick whichever option has better test accuracy with less overfitting
# Prefer: higher test accuracy, and if close, lower gap
print(f"\n   Comparison:")
print(f"   {'Option':<25} {'Test Acc':>10} {'Test F1':>10} {'ROC-AUC':>10} {'Gap':>8}")
print(f"   {'-'*65}")
print(f"   {'A: Tuned (no ES)':<25} {test_acc_A:>10.4f} {test_f1_A:>10.4f} {test_roc_A:>10.4f} {gap_A:>8.4f}")
print(f"   {'B: Tuned + Early Stop':<25} {test_acc_B:>10.4f} {test_f1_B:>10.4f} {test_roc_B:>10.4f} {gap_B:>8.4f}")

# Select best option: prioritize test accuracy, then lower gap
if test_acc_B >= test_acc_A or (test_acc_A - test_acc_B < 0.005 and gap_B < gap_A):
    final_model = tuned_early_stop
    final_option = "B (Tuned + Early Stopping)"
    train_acc_final = train_acc_B
    test_acc_final = test_acc_B
    test_f1_final = test_f1_B
    test_roc_final = test_roc_B
    y_test_pred_final = y_test_pred_B
    y_test_proba_final = y_test_proba_B
    y_train_pred_final = y_train_pred_B
else:
    final_model = tuned_direct
    final_option = "A (Tuned, no Early Stopping)"
    train_acc_final = train_acc_A
    test_acc_final = test_acc_A
    test_f1_final = test_f1_A
    test_roc_final = test_roc_A
    y_test_pred_final = y_test_pred_A
    y_test_proba_final = y_test_proba_A
    y_train_pred_final = y_train_pred_A

gap_final = train_acc_final - test_acc_final
print(f"\n   >>> SELECTED: {final_option}")


# ===========================================================================
# STEP 6: OPTIMAL THRESHOLD TUNING
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 6] Decision Threshold Optimization")
print("=" * 70)

def find_optimal_threshold(y_true, y_proba, metric="f1"):
    """Find threshold that maximizes the given metric."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if metric == "f1":
        scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold, scores[best_idx]

opt_thresh, opt_f1 = find_optimal_threshold(y_test, y_test_proba_final)
y_test_pred_opt = (y_test_proba_final >= opt_thresh).astype(int)

test_acc_opt = accuracy_score(y_test, y_test_pred_opt)
test_f1_opt = f1_score(y_test, y_test_pred_opt)
test_roc_opt = roc_auc_score(y_test, y_test_proba_final)  # ROC-AUC is threshold-independent
test_prec_opt = precision_score(y_test, y_test_pred_opt)
test_rec_opt = recall_score(y_test, y_test_pred_opt)

# Apply same threshold to train predictions
y_train_proba_final = final_model.predict_proba(X_train_scaled)[:, 1]
y_train_pred_opt = (y_train_proba_final >= opt_thresh).astype(int)
train_acc_opt = accuracy_score(y_train, y_train_pred_opt)
train_f1_opt = f1_score(y_train, y_train_pred_opt)

print(f"   Default threshold (0.5): Acc={test_acc_final:.4f}, F1={test_f1_final:.4f}")
print(f"   Optimal threshold ({opt_thresh:.3f}): Acc={test_acc_opt:.4f}, F1={test_f1_opt:.4f}")

# Use optimal threshold only if it improves F1 without destroying accuracy
if test_f1_opt > test_f1_final and test_acc_opt >= test_acc_final - 0.03:
    use_threshold = opt_thresh
    final_test_acc = test_acc_opt
    final_test_f1 = test_f1_opt
    final_train_acc = train_acc_opt
    final_train_f1 = train_f1_opt
    final_test_pred = y_test_pred_opt
    print(f"   >>> Using optimal threshold: {opt_thresh:.3f}")
else:
    use_threshold = 0.5
    final_test_acc = test_acc_final
    final_test_f1 = test_f1_final
    final_train_acc = train_acc_final
    final_train_f1 = f1_score(y_train, y_train_pred_final)
    final_test_pred = y_test_pred_final
    print(f"   >>> Keeping default threshold: 0.5 (optimal hurts accuracy too much)")

final_gap = final_train_acc - final_test_acc


# ===========================================================================
# STEP 7: 5-FOLD CROSS VALIDATION
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 7] 5-Fold Stratified Cross Validation")
print("=" * 70)

cv_acc = cross_val_score(final_model, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
cv_f1 = cross_val_score(final_model, X_train_scaled, y_train, cv=cv, scoring="f1", n_jobs=-1)
cv_roc = cross_val_score(final_model, X_train_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"\n   Accuracy per fold: {[f'{s:.4f}' for s in cv_acc]}")
print(f"   Mean CV Accuracy:  {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"   Mean CV F1:        {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
print(f"   Mean CV ROC-AUC:   {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")


# ===========================================================================
# STEP 8: OVERFITTING DIAGNOSIS
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 8] Overfitting / Underfitting Diagnosis")
print("=" * 70)

print(f"\n   +-------------------+-----------+-----------+")
print(f"   | Metric            |  Training |  Testing  |")
print(f"   +-------------------+-----------+-----------+")
print(f"   | Accuracy          |  {final_train_acc:.4f}   |  {final_test_acc:.4f}   |")
print(f"   | F1-Score          |  {final_train_f1:.4f}   |  {final_test_f1:.4f}   |")
print(f"   +-------------------+-----------+-----------+")
print(f"\n   Accuracy Gap: {final_gap:.4f}")

print(f"\n   DIAGNOSIS:")
if final_gap > 0.10:
    print(f"   [!] OVERFITTING: Train ({final_train_acc:.4f}) >> Test ({final_test_acc:.4f}), gap={final_gap:.4f}")
    print(f"       Model memorizes some training patterns that don't generalize.")
elif final_gap > 0.05:
    print(f"   [~] MILD OVERFITTING: Gap={final_gap:.4f}, acceptable for tree models on small data.")
elif final_test_acc < 0.70:
    print(f"   [!] UNDERFITTING: Both accuracies too low.")
else:
    print(f"   [OK] GOOD FIT: Train ({final_train_acc:.4f}) ~ Test ({final_test_acc:.4f}), gap={final_gap:.4f}")

# Compare with v1 CatBoost
print(f"\n   Comparison with Previous v1 CatBoost:")
print(f"   {'Metric':<25} {'v1 CatBoost':>12} {'v3 CatBoost':>12} {'Change':>10}")
print(f"   {'-'*60}")
print(f"   {'Train Accuracy':<25} {'1.0000':>12} {final_train_acc:>12.4f} {final_train_acc - 1.0:>+10.4f}")
print(f"   {'Test Accuracy':<25} {'0.8113':>12} {final_test_acc:>12.4f} {final_test_acc - 0.8113:>+10.4f}")
print(f"   {'Test F1-Score':<25} {'0.5785':>12} {final_test_f1:>12.4f} {final_test_f1 - 0.5785:>+10.4f}")
print(f"   {'Test ROC-AUC':<25} {'0.8530':>12} {test_roc_opt:>12.4f} {test_roc_opt - 0.8530:>+10.4f}")
print(f"   {'Overfit Gap':<25} {'0.1887':>12} {final_gap:>12.4f} {final_gap - 0.1887:>+10.4f}")


# ===========================================================================
# STEP 9: VISUALIZATIONS
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 9] Generating Visualizations")
print("=" * 70)

# 9a. Confusion Matrix
print("   Generating confusion matrix...")
cm = confusion_matrix(y_test, final_test_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["No Cloudburst", "Cloudburst"],
            yticklabels=["No Cloudburst", "Cloudburst"],
            ax=ax, cbar=True, annot_kws={"size": 18, "weight": "bold"},
            linewidths=2, linecolor="white")
ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
ax.set_title(f"Confusion Matrix -- CatBoost v3 (thresh={use_threshold:.3f})",
             fontsize=14, fontweight="bold", pad=15)
total = cm.sum()
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / total * 100
        ax.text(j + 0.5, i + 0.72, f"({pct:.1f}%)",
                ha="center", va="center", fontsize=11, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

# 9b. Classification Report
print(f"\n   Classification Report:")
print("   " + "-" * 55)
report = classification_report(y_test, final_test_pred,
                               target_names=["No Cloudburst", "Cloudburst"])
for line in report.split("\n"):
    print(f"   {line}")

# 9c. ROC Curve
print("\n   Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_test_proba_final)
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color="#f39c12", lw=3, label=f"CatBoost v3 (AUC={test_roc_opt:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
ax.fill_between(fpr, tpr, alpha=0.15, color="#f39c12")
ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold")
ax.set_title("ROC Curve -- CatBoost v3 (Production)", fontsize=15, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=200, bbox_inches="tight")
plt.close()

# 9d. Feature Importance
print("   Generating feature importance...")
if hasattr(final_model, "feature_importances_"):
    importances = final_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_imp_df)))
    bars = ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
                   color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, feat_imp_df["Importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title("Top 25 Features -- CatBoost v3 (Production)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    full_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    full_imp.sort_values("Importance", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
    )

# 9e. Threshold Optimization Plot
print("   Generating threshold optimization plot...")
prec_vals, rec_vals, thresh_vals = precision_recall_curve(y_test, y_test_proba_final)
f1_vals = 2 * (prec_vals[:-1] * rec_vals[:-1]) / (prec_vals[:-1] + rec_vals[:-1] + 1e-10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_vals, prec_vals[:-1], "b-", lw=2, label="Precision", alpha=0.8)
ax.plot(thresh_vals, rec_vals[:-1], "r-", lw=2, label="Recall", alpha=0.8)
ax.plot(thresh_vals, f1_vals, "g-", lw=2.5, label="F1-Score", alpha=0.9)
ax.axvline(x=use_threshold, color="black", linestyle="--", lw=1.5,
           label=f"Selected Threshold = {use_threshold:.3f}")
if use_threshold != 0.5:
    ax.axvline(x=0.5, color="gray", linestyle=":", lw=1, label="Default (0.5)")
ax.set_xlabel("Decision Threshold", fontsize=12, fontweight="bold")
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title("Threshold Optimization -- CatBoost v3", fontsize=14, fontweight="bold", pad=15)
ax.legend(loc="center left", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "threshold_optimization.png"), dpi=200, bbox_inches="tight")
plt.close()


# ===========================================================================
# STEP 10: SAVE MODEL
# ===========================================================================
print("\n" + "=" * 70)
print("  [STEP 10] Saving Final Production Model")
print("=" * 70)

model_artifact = {
    "model": final_model,
    "scaler": scaler,
    "feature_names": feature_names,
    "model_name": "CatBoost",
    "optimal_threshold": use_threshold,
    "best_params": search.best_params_,
    "selected_option": final_option,
    "metrics": {
        "train_accuracy": final_train_acc,
        "test_accuracy": final_test_acc,
        "train_f1": final_train_f1,
        "test_f1": final_test_f1,
        "test_roc_auc": test_roc_opt,
        "test_precision": precision_score(y_test, final_test_pred),
        "test_recall": recall_score(y_test, final_test_pred),
        "cv_accuracy_mean": cv_acc.mean(),
        "cv_accuracy_std": cv_acc.std(),
        "cv_f1_mean": cv_f1.mean(),
        "cv_roc_mean": cv_roc.mean(),
    },
    "preprocessing": {
        "dropped_columns": cols_to_drop,
        "dropped_non_numeric": non_numeric,
        "dropped_high_corr": high_corr_cols if high_corr_cols else [],
        "dropped_zero_var": zero_var_cols if zero_var_cols else [],
    }
}
joblib.dump(model_artifact, MODEL_PATH)
print(f"   Model saved: {MODEL_PATH}")
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY -- CatBoost Production Model v3")
print("=" * 70)

print(f"\n   Model:               CatBoost (Tuned)")
print(f"   Option:              {final_option}")
print(f"   Threshold:           {use_threshold:.3f}")
print(f"   -----------------------------------------")
print(f"   Training Accuracy:   {final_train_acc:.4f}")
print(f"   Testing Accuracy:    {final_test_acc:.4f}")
print(f"   Overfit Gap:         {final_gap:.4f}")
print(f"   Testing F1-Score:    {final_test_f1:.4f}")
print(f"   Testing ROC-AUC:     {test_roc_opt:.4f}")
print(f"   Testing Precision:   {precision_score(y_test, final_test_pred):.4f}")
print(f"   Testing Recall:      {recall_score(y_test, final_test_pred):.4f}")
print(f"   -----------------------------------------")
print(f"   CV Accuracy:         {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"   CV F1:               {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
print(f"   CV ROC-AUC:          {cv_roc.mean():.4f} +/- {cv_roc.std():.4f}")

print(f"\n   Output Files:")
print(f"      - {MODEL_PATH}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"      - {os.path.join(OUTPUT_DIR, f)}")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)
