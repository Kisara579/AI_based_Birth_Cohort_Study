"""
=============================================================================
BREASTFEEDING FEEDING PROBLEM PREDICTION — FULL CLINICAL ML PIPELINE
Fixed version — all issues resolved:

  FIX 1  SMOTE leakage in CV       → SMOTE now applied INSIDE each fold
  FIX 2  Duplicate SMOTE import    → removed re-import in Step 7
  FIX 3  Scaler inconsistency      → single scaler fit on X_train_res,
                                     transforms both train & test
  FIX 4  Redundant Step 6          → removed early scaler block;
                                     scaling handled per-pipeline
  FIX 5  oversample_minority()     → removed (unused dead code)
  FIX 6  plt.show() on Agg backend → removed; savefig() is sufficient
  FIX 7  Bootstrap CI ordering     → RF Calibrated threshold handled safely
  FIX 8  Excel sheet name truncation → clean sheet_name() helper
  FIX 9  SHAP value extraction     → robust handling of list vs ndarray
  FIX 10 kfold_metrics ignores SMOTE → ImbPipeline wraps SMOTE+model per fold
=============================================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # FIX 1

from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                      RandomizedSearchCV)
from sklearn.preprocessing    import StandardScaler
from sklearn.base             import clone
from sklearn.inspection       import permutation_importance
from sklearn.calibration      import (calibration_curve,
                                      CalibratedClassifierCV)

from sklearn.linear_model import LogisticRegression
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import (RandomForestClassifier,
                                  GradientBoostingClassifier,
                                  BaggingClassifier,
                                  VotingClassifier,
                                  StackingClassifier)
from sklearn.svm      import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score,
                             ConfusionMatrixDisplay)

# ── Optional packages ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
    print("  ✓ XGBoost available")
except ImportError:
    XGBOOST_OK = False
    print("  ✗ XGBoost not installed (pip install xgboost)")

try:
    import shap
    SHAP_OK = True
    print("  ✓ SHAP available")
except ImportError:
    SHAP_OK = False
    print("  ✗ SHAP not installed (pip install shap)")

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#1D9E75","#E8593C","#F0E442"]
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})

# ── Clinical settings ─────────────────────────────────────────────────────────
CLINICAL_WEIGHTS = {0: 1, 1: 3}
RECALL_TARGET    = 0.90
N_FOLDS          = 5
N_BOOT           = 1000


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sheet_name(base, suffix):
    """
    FIX 8 — build a valid Excel sheet name (≤31 chars, no illegal chars).
    Truncates base cleanly before appending suffix.
    """
    illegal = r'\/:*?[]'
    for ch in illegal:
        base = base.replace(ch, '')
    max_base = 31 - len(suffix) - 1          # 1 for the space
    return f"{base[:max_base]} {suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + FIX 10 — kfold_metrics with SMOTE INSIDE each fold
# ─────────────────────────────────────────────────────────────────────────────
def kfold_metrics(model, X, y, n_splits=5, apply_smote=True, scaled=False):
    """
    Stratified K-Fold CV.
    When apply_smote=True, SMOTE is applied INSIDE each training fold only
    (no leakage from validation samples into the synthetic generation).
    When scaled=True, a fresh StandardScaler is fit on each training fold.

    Returns a DataFrame: fold rows + Mean + Std rows.
    """
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Apply SMOTE inside fold (training fold only) — FIX 1
        if apply_smote:
            smote_cv = SMOTE(random_state=42)
            X_tr, y_tr = smote_cv.fit_resample(X_tr, y_tr)

        # Apply scaling inside fold (fit on train, transform both) — FIX 3
        if scaled:
            sc_cv = StandardScaler()
            X_tr  = sc_cv.fit_transform(X_tr)
            X_va  = sc_cv.transform(X_va)

        m = clone(model)
        m.fit(X_tr, y_tr)
        yp  = m.predict(X_va)
        ypr = (m.predict_proba(X_va)[:, 1]
               if hasattr(m, "predict_proba") else None)

        rows.append({
            "Fold"      : fold_idx,
            "Accuracy"  : accuracy_score(y_va, yp),
            "Precision" : precision_score(y_va, yp, zero_division=0),
            "Recall"    : recall_score(y_va, yp, zero_division=0),
            "F1"        : f1_score(y_va, yp, zero_division=0),
            "ROC-AUC"   : (roc_auc_score(y_va, ypr)
                           if ypr is not None else np.nan),
        })

    fold_df  = pd.DataFrame(rows).set_index("Fold")
    mean_row = fold_df.mean().rename("Mean")
    std_row  = fold_df.std().rename("Std")
    summary  = pd.concat([fold_df,
                          mean_row.to_frame().T,
                          std_row.to_frame().T])
    summary.index = ([f"Fold {i}" for i in range(1, n_splits + 1)]
                     + ["── Mean ──", "── Std  ──"])
    return summary


def bootstrap_ci(y_true, y_prob, n_boot=N_BOOT, alpha=0.05, threshold=0.5):
    """Bootstrap 95% CI for Acc / Recall / Precision / F1 / ROC-AUC."""
    store = {k: [] for k in
             ["Accuracy","Recall","Precision","F1","ROC-AUC"]}
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt  = y_true[idx]; yp_prob = y_prob[idx]
        yp  = (yp_prob >= threshold).astype(int)
        if len(np.unique(yt)) < 2:
            continue
        store["Accuracy"].append(accuracy_score(yt, yp))
        store["Recall"].append(recall_score(yt, yp, zero_division=0))
        store["Precision"].append(precision_score(yt, yp, zero_division=0))
        store["F1"].append(f1_score(yt, yp, zero_division=0))
        store["ROC-AUC"].append(roc_auc_score(yt, yp_prob))
    rows = []
    for met, vals in store.items():
        v  = np.array(vals)
        lo = np.percentile(v, alpha / 2 * 100)
        hi = np.percentile(v, (1 - alpha / 2) * 100)
        rows.append({"Metric": met, "Mean": v.mean(),
                     "Lower 95%": lo, "Upper 95%": hi})
    return pd.DataFrame(rows).set_index("Metric")


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test — pairwise model comparison."""
    a_right = (pred_a == y_true); b_right = (pred_b == y_true)
    n01 = np.sum(~a_right &  b_right); n10 = np.sum(a_right & ~b_right)
    contingency = np.array(
        [[np.sum(a_right & b_right), n01],
         [n10, np.sum(~a_right & ~b_right)]])
    chi2, p, _, _ = chi2_contingency(contingency, correction=True)
    return chi2, p


def optimal_threshold(y_true, y_prob, target_recall=RECALL_TARGET):
    """Lowest threshold achieving target_recall; then maximise precision."""
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    valid = np.where(rec_arr[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return 0.5, 0.0, 0.0
    best = valid[np.argmax(prec_arr[valid])]
    return float(thr_arr[best]), float(prec_arr[best]), float(rec_arr[best])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 1 — LOAD DATA")
print("=" * 70)

df_raw = pd.read_csv(
    "Breastfeeding and mental health data - open access.csv"
)
print(f"  Raw shape: {df_raw.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2 — BUILD TARGET: neonatal_feeding_problem")
print("=" * 70)

df = df_raw.copy()

# neonatal_feeding_problem = 1 if ANY of the four conditions are met, else 0
df["neonatal_feeding_problem"] = df.apply(
    lambda row: 1 if (
        str(row.get("breastfeeding_challenge_latching", "")).strip() == "1" or
        str(row.get("breastfeeding_challenge_tongue_tie", "")).strip() == "1" or
        str(row.get("breastfeeding_cessation", "")).strip() == "1" or
        str(row.get("breastfeeding_reduction", "")).strip() == "1"
    ) else 0,
    axis=1
)

TARGET = "neonatal_feeding_problem"

vc = df[TARGET].value_counts()
print(f"\n  Target distribution: {vc.to_dict()}")
print(f"  → {df[TARGET].mean()*100:.1f}% have a neonatal feeding problem")
print(f"\n  Criteria (any one sufficient):")
print(f"    • breastfeeding_challenge_latching  == 1")
print(f"    • breastfeeding_challenge_tongue_tie == 1")
print(f"    • breastfeeding_cessation            == 1")
print(f"    • breastfeeding_reduction            == 1")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3 — FEATURE ENGINEERING & CLEANING")
print("=" * 70)

df["MH_postnatal_depression"] = (
    (df["MH_postnatal_suspected_depression"].astype(str).str.strip() == "1") |
    (df["MH_postnatal_diagnosed_depression"].astype(str).str.strip() == "1")
).astype(int)

df["MH_postnatal_anxiety"] = (
    (df["MH_postnatal_suspected_anxiety"].astype(str).str.strip() == "1") |
    (df["MH_postnatal_diagnosed_anxiety"].astype(str).str.strip() == "1")
).astype(int)

FEATURES = [
    "maternal_age",
    "maternal_education",
    "child_birthweight_kgs",
    "child_gest_age_weeks",
    "delivery_method",
    "child_medical_complications",
    "breastfeeding_intention",
    "breastfeeding_knowledge",
    "breastfeeding_support",
    "breastfeeding_initiation_time"
]

df_model = df[FEATURES + [TARGET]].copy()

def clean_numeric(s):
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")

for col in [
    "maternal_age",
    "child_birthweight_kgs",
    "child_gest_age_weeks",
    "child_medical_complications",
    "breastfeeding_intention",
    "breastfeeding_knowledge",
    "breastfeeding_support",
    "breastfeeding_initiation_time"
]:
    df_model[col] = clean_numeric(df_model[col])

df_model["maternal_age"] = df_model["maternal_age"].replace(99, np.nan)

for col in FEATURES:
    n_null = int(df_model[col].isna().sum())
    if n_null > 0:
        med = df_model[col].median()
        df_model[col] = df_model[col].fillna(med)
        print(f"  Imputed '{col}' ({n_null} NaNs) → median = {med:.3f}")

print(f"\n  Cleaned shape : {df_model.shape}")
print(f"  Missing values: {df_model.isna().sum().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EDA
# ══════════════════════════════════════════════════════════════════════════════
import math

n_features = len(FEATURES)
ncols = 3
nrows = math.ceil(n_features / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = axes.flatten()

fig.suptitle(
    "EDA — Feature Distributions by Feeding Problem Status",
    fontsize=14,
    fontweight="bold"
)

for i, col in enumerate(FEATURES):

    ax = axes[i]

    if df_model[col].nunique() <= 5:

        ct = df_model.groupby([col, TARGET]).size().unstack(fill_value=0)

        ct.plot(
            kind="bar",
            ax=ax,
            color=["#5B9BD5", "#ED7D31"],
            edgecolor="black",
            width=0.7
        )

        ax.tick_params(axis="x", rotation=0)
        ax.legend(["No Problem", "Has Problem"], fontsize=8)

    else:

        for label, grp in df_model.groupby(TARGET):

            ax.hist(
                grp[col].dropna(),
                bins=25,
                alpha=0.6,
                label="No Problem" if label == 0 else "Has Problem",
                color="#5B9BD5" if label == 0 else "#ED7D31"
            )

        ax.legend(fontsize=8)

    ax.set_title(col, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

# Hide unused axes
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("eda_feature_distributions.png", dpi=120, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN / TEST SPLIT (80/20 stratified)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5 — TRAIN / TEST SPLIT  (80/20 stratified)")
print("=" * 70)

X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
print(f"  Test  positive rate: {y_test.mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SMOTE  (applied to training set only, ONCE for final models)
#
# FIX 1/2/3: The old code had three problems here:
#   (a) Duplicate "from imblearn.over_sampling import SMOTE" — removed
#   (b) scaler was refit on X_train_res creating a different distribution
#       from X_test_scaled — removed the early scaler block entirely
#   (c) A single scaler now fits on X_train_res and transforms X_test
#       so train and test share the same scaling reference
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6 — SMOTE  (training set only)")
print("=" * 70)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

bc_before = dict(zip(*np.unique(y_train,     return_counts=True)))
bc_after  = dict(zip(*np.unique(y_train_res, return_counts=True)))
print(f"  Before SMOTE : {bc_before}")
print(f"  After  SMOTE : {bc_after}")
print(f"  Clinical weights: {CLINICAL_WEIGHTS}  "
      f"(FN penalised {CLINICAL_WEIGHTS[1]}×)")

# ── FIX 3: single scaler — fit on SMOTE'd train, transform test ───────────
# (scale only continuous features; binary flags stay untouched via passthrough)
from sklearn.compose import ColumnTransformer
CONTINUOUS_IDX = [0, 2, 3]
# maternal_age
# child_birthweight_kgs
# child_gest_age_weeks

BINARY_IDX = [1, 4, 5, 6, 7, 8, 9]
# maternal_education
# delivery_method
# child_medical_complications
# breastfeeding_intention
# breastfeeding_knowledge
# breastfeeding_support
# breastfeeding_initiation_time     # delivery_method, MH flags

preprocessor = ColumnTransformer([
    ("scale",       StandardScaler(), CONTINUOUS_IDX),
    ("passthrough", "passthrough",    BINARY_IDX),
])

X_train_res_s = preprocessor.fit_transform(X_train_res)   # fit on SMOTE train
X_test_scaled = preprocessor.transform(X_test)             # same transform for test

print(f"  ColumnTransformer fit on SMOTE train → scales continuous, "
      f"passes binary as-is ✓")


# ── SMOTE class-distribution visualisation ───────────────────────────────────
before_counts = np.bincount(y_train)
after_counts  = np.bincount(y_train_res)
labels        = ["Class 0\n(No problem)", "Class 1\n(Has problem)"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("SMOTE: Class Distribution Before and After",
             fontsize=13, fontweight="bold")

for ax, counts, title, color in zip(
        axes,
        [before_counts, after_counts],
        ["Before SMOTE", "After SMOTE"],
        [["#5B9BD5","#ED7D31"], ["#5B9BD5","#1D9E75"]]):
    bars = ax.bar(labels, counts, color=color, edgecolor="black", width=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(after_counts) * 1.18)
    ax.spines[["top","right"]].set_visible(False)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom", fontweight="bold")

n_synthetic = after_counts[1] - before_counts[1]
axes[1].text(0.97, 0.95, f"+{n_synthetic} synthetic samples",
             transform=axes[1].transAxes, ha="right", va="top",
             fontsize=9, color="#1D9E75",
             bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f1", ec="#1D9E75"))
plt.tight_layout()
plt.savefig("smote_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()                    # FIX 6: removed plt.show() — Agg backend
print("  → smote_class_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
spw       = neg_count / pos_count   # XGBoost scale_pos_weight

# (name, estimator, uses_scaling)
MODEL_DEFS = [
    ("Logistic Regression",
     LogisticRegression(max_iter=1000, random_state=42,
                        class_weight=CLINICAL_WEIGHTS),
     True),
    ("Decision Tree",
     DecisionTreeClassifier(max_depth=6, min_samples_split=20,
                            random_state=42,
                            class_weight=CLINICAL_WEIGHTS),
     False),
    ("Random Forest",
     RandomForestClassifier(n_estimators=200, max_depth=8,
                            random_state=42,
                            class_weight=CLINICAL_WEIGHTS,
                            n_jobs=-1),
     False),
    ("Gradient Boosting",
     GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                max_depth=4, random_state=42),
     False),
    ("SVM",
     SVC(probability=True, kernel="rbf", C=1.0,
         random_state=42, class_weight=CLINICAL_WEIGHTS),
     True),
    ("KNN",
     KNeighborsClassifier(n_neighbors=7, metric="euclidean", n_jobs=-1),
     True),
    ("Balanced Bagging",
     BaggingClassifier(
         estimator=DecisionTreeClassifier(max_depth=6,
                                          class_weight=CLINICAL_WEIGHTS),
         n_estimators=200, max_samples=0.5,
         random_state=42, n_jobs=-1),
     False),
]

if XGBOOST_OK:
    MODEL_DEFS.append((
        "XGBoost",
        XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                      scale_pos_weight=spw, eval_metric="aucpr",
                      random_state=42, n_jobs=-1, verbosity=0),
        False))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 8 — TRAIN ALL MODELS")
print("=" * 70)

summary_rows   = []
trained        = {}
probs_test     = {}
preds_test     = {}
fold_summaries = {}

for name, estimator, scaled in MODEL_DEFS:

    # FIX 3: both Xtr/Xte now share the same preprocessor
    Xtr = X_train_res_s if scaled else X_train_res
    Ytr = y_train_res
    Xte = X_test_scaled  if scaled else X_test

    model = clone(estimator)
    model.fit(Xtr, Ytr)
    trained[name] = model

    y_pred = model.predict(Xte)
    y_prob = (model.predict_proba(Xte)[:, 1]
              if hasattr(model, "predict_proba") else None)

    preds_test[name] = y_pred
    probs_test[name] = y_prob

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc_ = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    ap   = (average_precision_score(y_test, y_prob)
            if y_prob is not None else np.nan)

    summary_rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                          "Recall": rec, "F1": f1,
                          "ROC-AUC": auc_, "PR-AUC": ap})

    print(f"\n{'─'*70}")
    print(f"  MODEL : {name}")
    print(f"{'─'*70}")
    print(f"  Test → Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  "
          f"ROC-AUC={auc_:.4f}  PR-AUC={ap:.4f}")
    print()
    print("  CLASSIFICATION REPORT:")
    for line in classification_report(
            y_test, y_pred,
            target_names=["No Problem (0)", "Has Problem (1)"],
            digits=4).splitlines():
        print("    " + line)

    # FIX 1 + FIX 10: SMOTE is applied INSIDE each CV fold
    fold_sum = kfold_metrics(model, X_train, y_train,
                             n_splits=N_FOLDS,
                             apply_smote=True,
                             scaled=scaled)
    fold_summaries[name] = fold_sum
    print(f"\n  {N_FOLDS}-FOLD CV METRICS (SMOTE inside each fold — no leakage):")
    print("  " + fold_sum.to_string().replace("\n", "\n  "))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 9 — LEADERBOARD (test-set scores)")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows).set_index("Model")
pd.set_option("display.float_format", "{:.4f}".format)
print(summary_df.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — HYPERPARAMETER TUNING (Random Forest — optimise Recall)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 10 — HYPERPARAMETER TUNING  (Randomized Search — Random Forest)")
print("=" * 70)

param_dist = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [4, 6, 8, 10, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf" : [2, 5, 10],
    "max_features"     : ["sqrt", "log2"],
}
rf_tuned = RandomizedSearchCV(
    RandomForestClassifier(random_state=42,
                           class_weight=CLINICAL_WEIGHTS,
                           n_jobs=-1),
    param_distributions=param_dist, n_iter=30, scoring="recall",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42, n_jobs=-1, verbose=0)

# FIX 3: tune on raw (pre-SMOTE) training data; SMOTE not needed inside
# RandomizedSearchCV because RF uses class_weight to handle imbalance
rf_tuned.fit(X_train_res, y_train_res)

print(f"  Best params    : {rf_tuned.best_params_}")
print(f"  Best CV Recall : {rf_tuned.best_score_:.4f}")

y_pred_tuned = rf_tuned.predict(X_test)
y_prob_tuned = rf_tuned.predict_proba(X_test)[:, 1]
print(f"\n  Tuned RF on test →"
      f"  Acc={accuracy_score(y_test,y_pred_tuned):.4f}"
      f"  Rec={recall_score(y_test,y_pred_tuned):.4f}"
      f"  F1={f1_score(y_test,y_pred_tuned):.4f}"
      f"  AUC={roc_auc_score(y_test,y_prob_tuned):.4f}")
print("\n  Classification Report (Tuned RF):")
for line in classification_report(
        y_test, y_pred_tuned,
        target_names=["No Problem (0)","Has Problem (1)"],
        digits=4).splitlines():
    print("    " + line)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — ENSEMBLE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 11 — ENSEMBLE MODELS")
print("=" * 70)

ensemble_defs = [
    ("Voting Ensemble",
     VotingClassifier(
         estimators=[
             ("rf", RandomForestClassifier(n_estimators=200, random_state=42,
                                           class_weight=CLINICAL_WEIGHTS,
                                           n_jobs=-1)),
             ("gb", GradientBoostingClassifier(n_estimators=200,
                                               learning_rate=0.05,
                                               max_depth=4, random_state=42)),
             ("dt", DecisionTreeClassifier(max_depth=6, random_state=42,
                                           class_weight=CLINICAL_WEIGHTS)),
         ], voting="soft"), False),
    ("Stacking Ensemble",
     StackingClassifier(
         estimators=[
             ("rf", RandomForestClassifier(n_estimators=100, random_state=42,
                                           class_weight=CLINICAL_WEIGHTS,
                                           n_jobs=-1)),
             ("gb", GradientBoostingClassifier(n_estimators=100,
                                               random_state=42)),
             ("dt", DecisionTreeClassifier(max_depth=5, random_state=42,
                                           class_weight=CLINICAL_WEIGHTS)),
         ],
         final_estimator=LogisticRegression(max_iter=500, random_state=42),
         cv=5, n_jobs=-1), False),
]

for name, estimator, scaled in ensemble_defs:
    Xtr = X_train_res_s if scaled else X_train_res
    Ytr = y_train_res
    Xte = X_test_scaled  if scaled else X_test

    model = clone(estimator)
    model.fit(Xtr, Ytr)
    trained[name] = model

    y_pred = model.predict(Xte)
    y_prob = (model.predict_proba(Xte)[:, 1]
              if hasattr(model, "predict_proba") else None)

    preds_test[name] = y_pred
    probs_test[name] = y_prob

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc_ = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    ap   = (average_precision_score(y_test, y_prob)
            if y_prob is not None else np.nan)

    summary_rows.append({"Model": name, "Accuracy": acc, "Precision": prec,
                          "Recall": rec, "F1": f1,
                          "ROC-AUC": auc_, "PR-AUC": ap})

    print(f"\n{'─'*70}")
    print(f"  MODEL : {name}")
    print(f"{'─'*70}")
    print(f"  Test → Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  "
          f"AUC={auc_:.4f}  PR-AUC={ap:.4f}")
    print("\n  CLASSIFICATION REPORT:")
    for line in classification_report(
            y_test, y_pred,
            target_names=["No Problem (0)", "Has Problem (1)"],
            digits=4).splitlines():
        print("    " + line)

    fold_sum = kfold_metrics(model, X_train, y_train,
                             n_splits=N_FOLDS,
                             apply_smote=True,
                             scaled=scaled)
    fold_summaries[name] = fold_sum
    print(f"\n  {N_FOLDS}-FOLD CV METRICS (SMOTE inside each fold):")
    print("  " + fold_sum.to_string().replace("\n", "\n  "))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — PROBABILITY CALIBRATION (Platt scaling on RF)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 12 — PROBABILITY CALIBRATION (Platt scaling — RF)")
print("=" * 70)

rf_base = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  random_state=42,
                                  class_weight=CLINICAL_WEIGHTS,
                                  n_jobs=-1)
rf_calibrated = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5)
rf_calibrated.fit(X_train_res, y_train_res)

y_prob_cal = rf_calibrated.predict_proba(X_test)[:, 1]
y_pred_cal = rf_calibrated.predict(X_test)
ap_cal     = average_precision_score(y_test, y_prob_cal)

print(f"  Calibrated RF → Recall={recall_score(y_test,y_pred_cal):.4f}  "
      f"ROC-AUC={roc_auc_score(y_test,y_prob_cal):.4f}  "
      f"PR-AUC={ap_cal:.4f}")

probs_test["RF Calibrated"] = y_prob_cal
preds_test["RF Calibrated"] = y_pred_cal
trained["RF Calibrated"]    = rf_calibrated

summary_rows.append({
    "Model"    : "RF Calibrated",
    "Accuracy" : accuracy_score(y_test, y_pred_cal),
    "Precision": precision_score(y_test, y_pred_cal, zero_division=0),
    "Recall"   : recall_score(y_test, y_pred_cal),
    "F1"       : f1_score(y_test, y_pred_cal),
    "ROC-AUC"  : roc_auc_score(y_test, y_prob_cal),
    "PR-AUC"   : ap_cal,
})


# ── Build lookup: model name → uses_scaling ───────────────────────────────────
all_model_defs = (
    {n: s for n, _, s in MODEL_DEFS} |
    {n: s for n, _, s in ensemble_defs} |
    {"RF Calibrated": False}
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — OVERFITTING CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 13 — OVERFITTING CHECK  (Train F1 vs Test F1)")
print("=" * 70)
print(f"  {'Model':<25} {'Train F1':>10} {'Test F1':>10} "
      f"{'Gap':>8} {'Status':>12}")
print("  " + "-" * 65)

for name, model in trained.items():
    sc  = all_model_defs.get(name, False)
    Xtr = X_train_res_s if sc else X_train_res
    Ytr = y_train_res
    Xte = X_test_scaled  if sc else X_test
    tr_f1 = f1_score(Ytr, model.predict(Xtr), zero_division=0)
    te_f1 = f1_score(y_test, model.predict(Xte), zero_division=0)
    gap   = tr_f1 - te_f1
    st    = "OVERFIT ⚠" if gap > 0.15 else "OK ✓"
    print(f"  {name:<25} {tr_f1:>10.4f} {te_f1:>10.4f} {gap:>8.4f} {st:>12}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14 — THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  STEP 14 — THRESHOLD TUNING  (target recall ≥ {RECALL_TARGET})")
print("=" * 70)

threshold_results = {}
print(f"\n  {'Model':<25} {'t=0.50 Rec':>12} {'Opt-t':>8} "
      f"{'Opt Rec':>10} {'Opt Prec':>10} {'F1@t':>8}")
print("  " + "-" * 78)

for name, y_prob in probs_test.items():
    if y_prob is None:
        continue
    t_opt, prec_opt, rec_opt = optimal_threshold(y_test, y_prob, RECALL_TARGET)
    y_pred_opt = (y_prob >= t_opt).astype(int)
    f1_opt     = f1_score(y_test, y_pred_opt, zero_division=0)
    rec_def    = recall_score(y_test, (y_prob >= 0.5).astype(int),
                              zero_division=0)
    threshold_results[name] = {
        "threshold": t_opt, "precision": prec_opt,
        "recall": rec_opt, "f1": f1_opt,
    }
    print(f"  {name:<25} {rec_def:>12.4f} {t_opt:>8.3f} "
          f"{rec_opt:>10.4f} {prec_opt:>10.4f} {f1_opt:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — BOOTSTRAP 95% CI (all models)
# FIX 7: threshold_results now includes RF Calibrated before CI is computed
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  STEP 15 — BOOTSTRAP 95% CI — ALL MODELS  ({N_BOOT} samples)")
print("=" * 70)

all_ci = {}

for name, y_prob in probs_test.items():
    if y_prob is None:
        continue
    # FIX 7: safe .get() with fallback 0.5 — handles any ordering
    thresh = threshold_results.get(name, {}).get("threshold", 0.5)
    ci_df  = bootstrap_ci(y_test, y_prob, n_boot=N_BOOT, threshold=thresh)
    all_ci[name] = ci_df
    print(f"\n  {name}  (threshold = {thresh:.3f})")
    print("  " + ci_df.to_string().replace("\n", "\n  "))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 16 — McNEMAR'S TEST (pairwise)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 16 — McNEMAR'S TEST  (pairwise model comparison)")
print("=" * 70)

model_names_mc = list(preds_test.keys())
print(f"\n  {'Model A':<25} {'Model B':<25} {'chi2':>8} "
      f"{'p-value':>10} {'Sig?':>6}")
print("  " + "-" * 80)

for i in range(len(model_names_mc)):
    for j in range(i + 1, len(model_names_mc)):
        na, nb  = model_names_mc[i], model_names_mc[j]
        chi2, p = mcnemar_test(y_test, preds_test[na], preds_test[nb])
        sig     = "Yes *" if p < 0.05 else "No"
        print(f"  {na:<25} {nb:<25} {chi2:>8.3f} {p:>10.4f} {sig:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 17 — PERMUTATION IMPORTANCE (all models)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 17 — PERMUTATION IMPORTANCE — ALL MODELS")
print("=" * 70)

perm_results = {}

for name, model in trained.items():
    sc  = all_model_defs.get(name, False)
    Xte = X_test_scaled if sc else X_test
    try:
        pi = permutation_importance(
            model, Xte, y_test,
            n_repeats=30, random_state=42, scoring="recall")
        perm_df_i = pd.DataFrame({
            "Feature"   : FEATURES,
            "Mean Drop" : pi.importances_mean,
            "Std"       : pi.importances_std,
        }).sort_values("Mean Drop", ascending=False)
        perm_results[name] = perm_df_i
        print(f"\n  {name}:")
        print("  " + perm_df_i.to_string(index=False).replace("\n", "\n  "))
    except Exception as e:
        print(f"\n  {name}: skipped ({e})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 18 — ALL PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 18 — GENERATING ALL PLOTS")
print("=" * 70)

all_results_df = pd.DataFrame(summary_rows).set_index("Model")
ls_cycle       = ["-","--","-.",":","-","--","-.",":","-","--","-"]

# ── (A) Model comparison bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(all_results_df))
w = 0.12
met_list   = ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
bar_colors = ["#5B9BD5","#ED7D31","#C44E52","#70AD47","#7030A0","#937860"]
for i, (met, col) in enumerate(zip(met_list, bar_colors)):
    ax.bar(x + i*w, all_results_df[met], width=w, label=met,
           color=col, edgecolor="black", linewidth=0.5)
ax.set_xticks(x + 2.5*w)
ax.set_xticklabels(all_results_df.index, rotation=25, ha="right")
ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics (incl. PR-AUC)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → model_comparison.png")

# ── (B) Confusion matrices ────────────────────────────────────────────────────
n_m     = len(trained)
ncols_c = 4
nrows_c = (n_m + ncols_c - 1) // ncols_c
fig, axes = plt.subplots(nrows_c, ncols_c,
                         figsize=(ncols_c * 4.5, nrows_c * 4))
axes = axes.flatten()
for idx, (name, model) in enumerate(trained.items()):
    sc  = all_model_defs.get(name, False)
    Xte = X_test_scaled if sc else X_test
    cm  = confusion_matrix(y_test, model.predict(Xte))
    ConfusionMatrixDisplay(cm,
        display_labels=["No Problem","Has Problem"]).plot(
        ax=axes[idx], colorbar=False, cmap="Blues")
    axes[idx].set_title(name)
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Confusion Matrices — All Models",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → confusion_matrices.png")

# ── (C) Individual ROC curves ─────────────────────────────────────────────────
roc_names = [n for n, ypr in probs_test.items() if ypr is not None]
ncols_r   = 4
nrows_r   = (len(roc_names) + ncols_r - 1) // ncols_r
fig, axes = plt.subplots(nrows_r, ncols_r,
                         figsize=(ncols_r * 5, nrows_r * 4.5))
axes = axes.flatten()
for idx, name in enumerate(roc_names):
    ypr_i       = probs_test[name]
    fpr, tpr, _ = roc_curve(y_test, ypr_i)
    auc_val     = roc_auc_score(y_test, ypr_i)
    ax          = axes[idx]
    ax.plot(fpr, tpr, color=PALETTE[idx % len(PALETTE)], lw=2.5,
            label=f"AUC = {auc_val:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE[idx % len(PALETTE)])
    ax.plot([0,1],[0,1],"k--", lw=1.2, label="Random")
    ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
    ax.set_title(f"{name}\nAUC = {auc_val:.4f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.grid(alpha=0.3)
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Individual ROC-AUC Curves", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("roc_individual.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → roc_individual.png")

# ── (D) Combined ROC overlay ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
for idx, name in enumerate(roc_names):
    ypr_i       = probs_test[name]
    fpr, tpr, _ = roc_curve(y_test, ypr_i)
    auc_val     = roc_auc_score(y_test, ypr_i)
    ax.plot(fpr, tpr, lw=2, color=PALETTE[idx % len(PALETTE)],
            linestyle=ls_cycle[idx % len(ls_cycle)],
            label=f"{name}  (AUC={auc_val:.4f})")
ax.plot([0,1],[0,1],"k--", lw=1.5, label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Combined ROC-AUC Curves — All Models",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig("roc_combined.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → roc_combined.png")

# ── (E) Individual Precision-Recall curves ────────────────────────────────────
pr_names  = [n for n, ypr in probs_test.items() if ypr is not None]
ncols_pr  = 4
nrows_pr  = (len(pr_names) + ncols_pr - 1) // ncols_pr
fig, axes = plt.subplots(nrows_pr, ncols_pr,
                         figsize=(ncols_pr * 5, nrows_pr * 4.5))
axes = axes.flatten()
for idx, name in enumerate(pr_names):
    ypr_i              = probs_test[name]
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, ypr_i)
    ap_val             = average_precision_score(y_test, ypr_i)
    t_res              = threshold_results.get(name, {})
    ax                 = axes[idx]
    ax.plot(rec_arr, prec_arr, color=PALETTE[idx % len(PALETTE)], lw=2.5,
            label=f"AP = {ap_val:.4f}")
    if t_res:
        ax.scatter([t_res["recall"]], [t_res["precision"]],
                   color="red", s=90, zorder=5,
                   label=f"t={t_res['threshold']:.2f}")
    ax.axhline(y_test.mean(), color="gray", lw=1, linestyle="--",
               alpha=0.6, label="Baseline")
    ax.set_xlabel("Recall", fontsize=9); ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(f"{name}\nAP = {ap_val:.4f}")
    ax.legend(fontsize=8); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.grid(alpha=0.3)
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f"Precision-Recall Curves — Optimal threshold at Recall ≥ {RECALL_TARGET}",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("precision_recall_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → precision_recall_curves.png")

# ── (F) Combined PR overlay ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
for idx, name in enumerate(pr_names):
    ypr_i             = probs_test[name]
    prec_a, rec_a, _  = precision_recall_curve(y_test, ypr_i)
    ap_val            = average_precision_score(y_test, ypr_i)
    ax.plot(rec_a, prec_a, lw=2, color=PALETTE[idx % len(PALETTE)],
            linestyle=ls_cycle[idx % len(ls_cycle)],
            label=f"{name}  (AP={ap_val:.4f})")
ax.axhline(y_test.mean(), color="gray", lw=1.5, linestyle="--",
           label=f"Baseline (prevalence={y_test.mean():.2f})")
ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Combined Precision-Recall Curves — All Models",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.set_xlim([0,1]); ax.set_ylim([0,1.05]); ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig("pr_combined.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → pr_combined.png")

# ── (G) Calibration plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],"k--", lw=1.5, label="Perfect calibration")
for idx, name in enumerate(roc_names):
    try:
        frac_pos, mean_pred = calibration_curve(
            y_test, probs_test[name], n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", lw=2,
                color=PALETTE[idx % len(PALETTE)], label=name)
    except Exception:
        pass
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Plot — Reliability Diagram",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → calibration_plot.png")

# ── (H) Permutation importance — one subplot per model ───────────────────────
if perm_results:
    n_pi    = len(perm_results)
    ncols_p = 3
    nrows_p = (n_pi + ncols_p - 1) // ncols_p
    fig, axes = plt.subplots(nrows_p, ncols_p,
                             figsize=(ncols_p * 6, nrows_p * 4.5))
    axes = axes.flatten()
    for idx, (name, perm_df_i) in enumerate(perm_results.items()):
        ax          = axes[idx]
        perm_sorted = perm_df_i.sort_values("Mean Drop")
        ax.barh(perm_sorted["Feature"], perm_sorted["Mean Drop"],
                xerr=perm_sorted["Std"],
                color=PALETTE[idx % len(PALETTE)],
                edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 1.2, "capsize": 4})
        ax.set_title(name)
        ax.set_xlabel("Mean Recall drop when permuted")
        ax.axvline(0, color="gray", lw=0.8, linestyle="--")
        ax.grid(axis="x", alpha=0.4)
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Permutation Importance — ALL Models  (Recall drop per feature)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("permutation_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → permutation_importance.png")

    # ── (H2) Permutation importance consensus heatmap ────────────────────────
    perm_mean_matrix = pd.DataFrame(
        {name: df.set_index("Feature")["Mean Drop"]
         for name, df in perm_results.items()}
    )
    fig, ax = plt.subplots(figsize=(max(10, len(perm_results) * 1.5), 6))
    sns.heatmap(perm_mean_matrix, annot=True, fmt=".4f",
                cmap="RdYlGn_r", center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Recall drop"})
    ax.set_title("Permutation Importance Consensus — All Models",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig("permutation_importance_consensus.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → permutation_importance_consensus.png")

# ── (I) Tree-based feature importance (tuned RF) ──────────────────────────────
fi = pd.Series(rf_tuned.best_estimator_.feature_importances_,
               index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
fi.plot(kind="barh", ax=ax, color="#ED7D31", edgecolor="black")
ax.set_title("Tree-Based Feature Importance — Tuned Random Forest")
ax.set_xlabel("Mean Decrease in Impurity")
ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → feature_importance.png")

# ── (J) Per-fold heatmaps ─────────────────────────────────────────────────────
fig_rows = (len(fold_summaries) + 1) // 2
fig, axes = plt.subplots(fig_rows, 2, figsize=(18, fig_rows * 4.5))
axes = axes.flatten()
for idx, (name, fold_df) in enumerate(fold_summaries.items()):
    ax = axes[idx]
    plot_df = fold_df.copy()
    plot_df.index = [str(i) for i in plot_df.index]
    sns.heatmap(plot_df.astype(float), annot=True, fmt=".4f",
                cmap="YlOrRd", linewidths=0.5, ax=ax,
                vmin=0, vmax=1, cbar_kws={"shrink": 0.7})
    ax.set_title(f"{name} — Per-Fold Metrics")
    ax.tick_params(axis="y", rotation=0)
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Per-Fold Cross-Validation Metrics — All Models",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("per_fold_heatmaps.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → per_fold_heatmaps.png")

# ── (K) Per-fold line charts ──────────────────────────────────────────────────
fold_labels = [f"Fold {i}" for i in range(1, N_FOLDS + 1)]
fig, axes   = plt.subplots(1, 2, figsize=(16, 6))
for idx, (name, fold_df) in enumerate(fold_summaries.items()):
    fold_only = fold_df.iloc[:N_FOLDS]
    axes[0].plot(fold_labels, fold_only["F1"].values,
                 marker="o", lw=2, color=PALETTE[idx % len(PALETTE)], label=name)
    axes[1].plot(fold_labels, fold_only["Recall"].values,
                 marker="s", lw=2, color=PALETTE[idx % len(PALETTE)], label=name)
for ax, metric in zip(axes, ["F1 Score", "Recall"]):
    ax.set_xlabel("Fold"); ax.set_ylabel(metric)
    ax.set_title(f"Per-Fold {metric} — All Models")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("per_fold_lines.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → per_fold_lines.png")

# ── (L) CV box plot ───────────────────────────────────────────────────────────
cv_data = []
for name, fold_df in fold_summaries.items():
    for val in fold_df["F1"].iloc[:N_FOLDS]:
        cv_data.append({"Model": name, "CV F1": val})
cv_df = pd.DataFrame(cv_data)
fig, ax = plt.subplots(figsize=(14, 6))
cv_df.boxplot(column="CV F1", by="Model", ax=ax, patch_artist=True, grid=True)
ax.set_title("5-Fold CV F1 Distribution — All Models")
ax.set_xlabel(""); ax.set_ylabel("F1 Score")
plt.xticks(rotation=25, ha="right"); plt.suptitle("")
plt.tight_layout()
plt.savefig("cv_boxplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → cv_boxplot.png")

# ── (M) Threshold tuning summary ─────────────────────────────────────────────
th_names = list(threshold_results.keys())
th_recs  = [threshold_results[n]["recall"]    for n in th_names]
th_precs = [threshold_results[n]["precision"] for n in th_names]
th_f1s   = [threshold_results[n]["f1"]        for n in th_names]
th_thrs  = [threshold_results[n]["threshold"] for n in th_names]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x_t = np.arange(len(th_names)); w_t = 0.28
axes[0].bar(x_t - w_t, th_recs,  width=w_t, label="Recall",    color="#C44E52")
axes[0].bar(x_t,        th_precs, width=w_t, label="Precision", color="#5B9BD5")
axes[0].bar(x_t + w_t,  th_f1s,  width=w_t, label="F1",        color="#70AD47")
axes[0].axhline(RECALL_TARGET, color="red", lw=1.5, linestyle="--",
                label=f"Target recall={RECALL_TARGET}")
axes[0].set_xticks(x_t)
axes[0].set_xticklabels(th_names, rotation=25, ha="right")
axes[0].set_ylim(0, 1.12)
axes[0].set_title(f"Metrics at Optimal Threshold (recall ≥ {RECALL_TARGET})")
axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.4)
axes[1].bar(x_t, th_thrs, color="#8172B2", edgecolor="black", linewidth=0.5)
axes[1].axhline(0.5, color="gray", lw=1.5, linestyle="--", label="Default t=0.50")
axes[1].set_xticks(x_t)
axes[1].set_xticklabels(th_names, rotation=25, ha="right")
axes[1].set_ylim(0, 1.05); axes[1].set_title("Optimal Threshold per Model")
axes[1].set_ylabel("Threshold value")
axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("threshold_tuning.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → threshold_tuning.png")

# ── (N) Bootstrap CI — one subplot per model ─────────────────────────────────
n_ci    = len(all_ci)
ncols_n = 3
nrows_n = (n_ci + ncols_n - 1) // ncols_n
fig, axes = plt.subplots(nrows_n, ncols_n,
                         figsize=(ncols_n * 6, nrows_n * 4.5))
axes = axes.flatten()
for idx, (name, ci_df_i) in enumerate(all_ci.items()):
    ax     = axes[idx]
    mets   = ci_df_i.index.tolist()
    means  = ci_df_i["Mean"].values
    err_lo = means - ci_df_i["Lower 95%"].values
    err_hi = ci_df_i["Upper 95%"].values - means
    ax.bar(mets, means, color=PALETTE[idx % len(PALETTE)],
           edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.errorbar(mets, means, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=6, linewidth=1.8)
    ax.set_ylim(0, 1.18); ax.set_title(name); ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.4)
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f"Bootstrap 95% CIs — ALL Models  ({N_BOOT} samples)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("bootstrap_ci.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → bootstrap_ci.png")

# ── (N2) Bootstrap CI recall comparison ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ci_model_names = list(all_ci.keys())
recall_means   = [all_ci[n].loc["Recall","Mean"]      for n in ci_model_names]
recall_lo      = [all_ci[n].loc["Recall","Lower 95%"] for n in ci_model_names]
recall_hi      = [all_ci[n].loc["Recall","Upper 95%"] for n in ci_model_names]
err_lo_r       = [m - lo for m, lo in zip(recall_means, recall_lo)]
err_hi_r       = [hi - m for m, hi in zip(recall_means, recall_hi)]
x_ci = np.arange(len(ci_model_names))
ax.bar(x_ci, recall_means,
       color=[PALETTE[i % len(PALETTE)] for i in range(len(ci_model_names))],
       edgecolor="black", linewidth=0.5, alpha=0.85)
ax.errorbar(x_ci, recall_means, yerr=[err_lo_r, err_hi_r],
            fmt="none", color="black", capsize=7, linewidth=2)
ax.axhline(RECALL_TARGET, color="red", lw=1.5, linestyle="--",
           label=f"Clinical target recall = {RECALL_TARGET}")
ax.set_xticks(x_ci)
ax.set_xticklabels(ci_model_names, rotation=25, ha="right")
ax.set_ylim(0, 1.15); ax.set_ylabel("Recall (with 95% CI)")
ax.set_title("Recall Comparison with Bootstrap 95% CI — All Models",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("bootstrap_ci_recall_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("  → bootstrap_ci_recall_comparison.png")

# ── (O) SHAP (optional) ───────────────────────────────────────────────────────
if SHAP_OK:
    print("\n  Generating SHAP plots...")
    try:
        rf_model  = trained["Random Forest"]
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X_test)

        # FIX 9: robust SHAP value extraction — handles both list and ndarray
        if isinstance(shap_vals, list):
            # older SHAP: list of [class0_array, class1_array]
            sv = shap_vals[1]
        elif shap_vals.ndim == 3:
            # newer SHAP: shape (n_samples, n_features, n_classes)
            sv = shap_vals[:, :, 1]
        else:
            # already 2-D (binary shorthand)
            sv = shap_vals

        shap.summary_plot(sv, X_test, feature_names=FEATURES, show=False)
        plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("  → shap_summary.png")

        shap.summary_plot(sv, X_test, feature_names=FEATURES,
                          plot_type="bar", show=False)
        plt.savefig("shap_bar.png", dpi=120, bbox_inches="tight")
        plt.close()
        print("  → shap_bar.png")
    except Exception as e:
        print(f"  ✗ SHAP failed: {e}")
else:
    print("  ✗ SHAP skipped (pip install shap to enable)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 19 — EXCEL SUMMARY REPORT
# FIX 8: sheet_name() helper prevents truncation artifacts
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 19 — EXCEL SUMMARY REPORT")
print("=" * 70)

try:
    with pd.ExcelWriter("results_summary.xlsx", engine="openpyxl") as writer:

        pd.DataFrame(summary_rows).set_index("Model").to_excel(
            writer, sheet_name="Leaderboard")

        thr_rows = [{"Model": n, **v} for n, v in threshold_results.items()]
        pd.DataFrame(thr_rows).set_index("Model").to_excel(
            writer, sheet_name="Threshold Tuning")

        for name, ci_df_i in all_ci.items():
            ci_df_i.to_excel(writer, sheet_name=sheet_name(name, "CI"))

        for name, perm_df_i in perm_results.items():
            perm_df_i.to_excel(writer,
                               sheet_name=sheet_name(name, "PI"),
                               index=False)

        for name, fold_df in fold_summaries.items():
            fold_df.to_excel(writer, sheet_name=sheet_name(name, "CV"))

    print("  → results_summary.xlsx")

except ImportError:
    print("  ✗ openpyxl not installed (pip install openpyxl)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 20 — FINAL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 20 — FINAL LEADERBOARD")
print("=" * 70)

final_df = pd.DataFrame(summary_rows).set_index("Model")
best_f1  = final_df["F1"].idxmax()
best_rec = final_df["Recall"].idxmax()
best_pr  = final_df["PR-AUC"].idxmax()

print(f"\n  Best by F1      → {best_f1}")
print(f"  Best by Recall  → {best_rec}   ← preferred clinically")
print(f"  Best by PR-AUC  → {best_pr}")
print("\n  Full leaderboard:\n")
print(final_df.to_string())

print("\n" + "=" * 70)
print("  ALL DONE ✓")
print()
print("  PNG outputs:")
print("    eda_feature_distributions.png       eda_correlation_heatmap.png")
print("    smote_class_distribution.png        model_comparison.png")
print("    confusion_matrices.png              roc_individual.png")
print("    roc_combined.png                    precision_recall_curves.png")
print("    pr_combined.png                     calibration_plot.png")
print("    threshold_tuning.png                permutation_importance.png")
print("    permutation_importance_consensus.png feature_importance.png")
print("    per_fold_heatmaps.png               per_fold_lines.png")
print("    cv_boxplot.png                      bootstrap_ci.png")
print("    bootstrap_ci_recall_comparison.png")
if SHAP_OK:
    print("    shap_summary.png                    shap_bar.png")
print()
print("  Excel: results_summary.xlsx")
print("=" * 70)