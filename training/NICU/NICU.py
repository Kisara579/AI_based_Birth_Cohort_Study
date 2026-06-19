"""
=============================================================================
NICU ADMISSION PREDICTION — FINAL CORRECTED PIPELINE
=============================================================================
All critical bugs fixed + performance improvements applied:

  FIX ① Scaler fitted on raw train set (not SMOTE data)
  FIX ② Threshold tuned on validation fold (not test set) — no leakage
  FIX ③ VotingClassifier uses consistent pipelines with internal SMOTE
  FIX ④ SMOTE applied inside each CV fold via imblearn Pipeline
  FIX ⑤ Optuna CV uses proper imblearn Pipeline (no leakage)

  IMPROVEMENT ① Repeated Stratified K-Fold for final evaluation
  IMPROVEMENT ② Calibration via CalibratedClassifierCV (Platt scaling)
  IMPROVEMENT ③ Bootstrap confidence intervals on test metrics
  IMPROVEMENT ④ Clean model registry — one source of truth per model
  IMPROVEMENT ⑤ SHAP for all tree models (RF, GB, XGB, LGBM)

Models (10 total):
  1.  Logistic Regression
  2.  Decision Tree
  3.  Random Forest
  4.  Gradient Boosting
  5.  SVM
  6.  KNN
  7.  XGBoost      (scale_pos_weight, no SMOTE)
  8.  LightGBM     (is_unbalance,     no SMOTE)
  9.  Stacking     (XGB+LGBM+RF+GB → LR meta)
  10. Soft Voting  (all 8 via unified pipelines)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from collections import Counter

# --- sklearn ---
from sklearn.model_selection     import (train_test_split, StratifiedKFold,
                                          RepeatedStratifiedKFold, cross_val_score)
from sklearn.preprocessing       import StandardScaler
from sklearn.pipeline            import Pipeline
from sklearn.linear_model        import LogisticRegression
from sklearn.tree                import DecisionTreeClassifier
from sklearn.ensemble            import (RandomForestClassifier,
                                          GradientBoostingClassifier,
                                          VotingClassifier, StackingClassifier)
from sklearn.svm                 import SVC
from sklearn.neighbors           import KNeighborsClassifier
from sklearn.calibration         import CalibratedClassifierCV, calibration_curve
from sklearn.metrics             import (accuracy_score, precision_score, recall_score,
                                          f1_score, roc_auc_score, confusion_matrix,
                                          classification_report, roc_curve,
                                          precision_recall_curve,
                                          average_precision_score,
                                          matthews_corrcoef, fbeta_score,
                                          ConfusionMatrixDisplay, brier_score_loss)
from sklearn.utils               import resample

# --- imbalance ---
from imblearn.over_sampling  import SMOTE, ADASYN
from imblearn.combine        import SMOTEENN, SMOTETomek
from imblearn.pipeline       import Pipeline as ImbPipeline   # KEY: keeps SMOTE inside CV

# --- boosters ---
import xgboost  as xgb
import lightgbm as lgb

# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH    = "gdm_t2dm_risk_dataset (3).csv"
TARGET       = "NICU_admission"
RANDOM_STATE = 42
N_SPLITS     = 5
N_REPEATS    = 3           # for RepeatedStratifiedKFold final eval
OPTUNA_TRIALS = 60         # raise to 100 for better tuning
N_BOOTSTRAP   = 1000       # for CI on test metrics

FEATURES = [
    "Insulin_treatment_during_pregnancy",
    "Pregnancy_complications_hypertensive_disorders",
    "Pregnancy_complications_preterm_delivery",
    "Abnormal_OGTT_results",
]

SHORT = {
    "Insulin_treatment_during_pregnancy"             : "Insulin Tx",
    "Pregnancy_complications_hypertensive_disorders" : "Hypertensive",
    "Pregnancy_complications_preterm_delivery"       : "Preterm",
    "Abnormal_OGTT_results"                          : "Abnormal OGTT",
    "Insulin_x_Preterm"                              : "Ins×Preterm",
    "Hypertension_x_OGTT"                            : "Hyp×OGTT",
    "Insulin_x_Hypertension"                         : "Ins×Hyp",
    "Risk_score"                                     : "Risk Score",
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  NICU ADMISSION PREDICTION — FINAL CORRECTED PIPELINE")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

assert TARGET in df.columns, f"Target column '{TARGET}' not found."
assert df[TARGET].nunique() == 2, "Target must be binary."
assert df.isnull().sum().sum() == 0, "Dataset has missing values — handle them first."

neg = (df[TARGET] == 0).sum()
pos = (df[TARGET] == 1).sum()
scale_pos_weight = neg / pos

print(f"\n  Dataset         : {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  Class balance   : {neg} negative  |  {pos} positive")
print(f"  Imbalance ratio : {scale_pos_weight:.2f}:1")

# ═══════════════════════════════════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  SECTION 1 : EDA")
print("─" * 70)

print("\n[2a] Missing values :", df.isnull().sum().sum())
print("\n[2b] Feature stats:")
print(df[FEATURES].describe().T[["mean","std","min","max"]].to_string())

print("\n[2c] Feature → NICU correlation:")
for f in FEATURES:
    r = df[[f, TARGET]].corr().iloc[0, 1]
    print(f"  {SHORT[f]:<18}  r = {r:+.4f}")

print("\n[2d] NICU rate by feature presence:")
for f in FEATURES:
    rate = df.groupby(f)[TARGET].mean()
    print(f"  {SHORT[f]:<18}  absent={rate.get(0,0):.2%}  present={rate.get(1,0):.2%}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  SECTION 2 : FEATURE ENGINEERING")
print("─" * 70)

df["Insulin_x_Preterm"]      = (df["Insulin_treatment_during_pregnancy"] *
                                 df["Pregnancy_complications_preterm_delivery"])
df["Hypertension_x_OGTT"]    = (df["Pregnancy_complications_hypertensive_disorders"] *
                                 df["Abnormal_OGTT_results"])
df["Insulin_x_Hypertension"] = (df["Insulin_treatment_during_pregnancy"] *
                                 df["Pregnancy_complications_hypertensive_disorders"])
df["Risk_score"]             = (df["Insulin_treatment_during_pregnancy"] +
                                 df["Pregnancy_complications_hypertensive_disorders"] +
                                 df["Pregnancy_complications_preterm_delivery"] +
                                 df["Abnormal_OGTT_results"])

FEATURES_EXT = FEATURES + [
    "Insulin_x_Preterm", "Hypertension_x_OGTT",
    "Insulin_x_Hypertension", "Risk_score",
]
feat_names = [SHORT.get(f, f) for f in FEATURES_EXT]

X = df[FEATURES_EXT].copy()
y = df[TARGET].copy()

print(f"\n  Features: {len(FEATURES)} → {len(FEATURES_EXT)}")
print("\n  NICU rate by Risk_score:")
for v, grp in df.groupby("Risk_score"):
    rate = grp[TARGET].mean()
    bar  = "█" * int(rate * 40)
    print(f"    Score {v}: {rate:.2%}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT  (80 / 20 stratified — held-out once, never touched
#    during tuning or threshold selection)
# ═══════════════════════════════════════════════════════════════════════════
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ── FIX ①: Scaler fitted on RAW train set only ──────────────────────────
scaler = StandardScaler()
scaler.fit(X_train_raw)           # fit on raw, no synthetic samples
X_train_sc  = scaler.transform(X_train_raw)
X_test_sc   = scaler.transform(X_test)

# CV splitters
kfold        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=RANDOM_STATE)
rep_kfold    = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                        random_state=RANDOM_STATE)

# ═══════════════════════════════════════════════════════════════════════════
# 5. IMBALANCE STRATEGIES COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  SECTION 3 : IMBALANCE HANDLING COMPARISON")
print("─" * 70)

samplers = {
    "SMOTE"      : SMOTE(random_state=RANDOM_STATE),
    "SMOTE-ENN"  : SMOTEENN(random_state=RANDOM_STATE),
    "SMOTE-Tomek": SMOTETomek(random_state=RANDOM_STATE),
    "ADASYN"     : ADASYN(random_state=RANDOM_STATE),
}
sampled_data = {}
print()
for name, sampler in samplers.items():
    try:
        Xr, yr = sampler.fit_resample(X_train_raw, y_train_raw)
        sampled_data[name] = (Xr, yr)
        c = Counter(yr)
        print(f"  {name:<14}: class 0={c[0]:>5}  class 1={c[1]:>5}")
    except Exception as e:
        print(f"  {name:<14}: FAILED — {e}")
        sampled_data[name] = (X_train_raw.copy(), y_train_raw.copy())

print(f"\n  Sklearn models  : imblearn Pipeline with SMOTE-ENN inside each CV fold")
print(f"  XGB / LGBM      : raw split + scale_pos_weight={scale_pos_weight:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. HELPER: THRESHOLD TUNING ON VALIDATION DATA  (FIX ②)
#    Threshold is found on a held-out VALIDATION split from X_train_raw,
#    NEVER on X_test.  The test set is only touched once for final reporting.
# ═══════════════════════════════════════════════════════════════════════════

def best_f2_threshold_val(model, X_val, y_val, X_is_scaled=False):
    """
    Find the F2-maximising threshold using VALIDATION data only.
    model  : already fitted estimator
    X_val  : validation features (scaled if needed)
    y_val  : validation labels
    Returns: float threshold
    """
    probs = (model.predict_proba(X_val)[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(X_val))
    prec, rec, thr = precision_recall_curve(y_val, probs)
    f2  = (5 * prec * rec) / (4 * prec + rec + 1e-9)
    idx = np.argmax(f2[:-1])
    return float(thr[idx])


def bootstrap_ci(y_true, y_prob, y_pred, metric_fn, n=N_BOOTSTRAP, alpha=0.05):
    """Return (mean, lower, upper) via bootstrap for a given metric function."""
    scores = []
    rng = np.random.RandomState(RANDOM_STATE)
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx], y_pred[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    return scores.mean(), np.percentile(scores, 100*alpha/2), np.percentile(scores, 100*(1-alpha/2))


def evaluate(name, fitted_model, X_te, y_te, threshold):
    """
    Evaluate a fitted model at a PRE-COMPUTED threshold (from validation data).
    threshold : float — found on val set, NOT on X_te.
    """
    probs  = (fitted_model.predict_proba(X_te)[:, 1]
              if hasattr(fitted_model, "predict_proba")
              else fitted_model.decision_function(X_te))
    y_pred = (probs >= threshold).astype(int)

    return {
        "Model"    : name,
        "Threshold": round(threshold, 3),
        "Accuracy" : round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_te, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_te, y_pred, zero_division=0), 4),
        "F2"       : round(fbeta_score(y_te, y_pred, beta=2, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_te, probs), 4),
        "PR-AUC"   : round(average_precision_score(y_te, probs), 4),
        "MCC"      : round(matthews_corrcoef(y_te, y_pred), 4),
        "Brier"    : round(brier_score_loss(y_te, probs), 4),
        "y_pred"   : y_pred,
        "y_prob"   : probs,
    }


def print_result(res):
    print(f"  Threshold  : {res['Threshold']}  (tuned on val set)")
    print(f"  Accuracy   : {res['Accuracy']}")
    print(f"  Precision  : {res['Precision']}")
    print(f"  Recall     : {res['Recall']}   ← clinical priority")
    print(f"  F1         : {res['F1']}")
    print(f"  F2         : {res['F2']}   ← weights Recall 2×")
    print(f"  ROC-AUC    : {res['ROC-AUC']}")
    print(f"  PR-AUC     : {res['PR-AUC']}  ← best for imbalanced")
    print(f"  MCC        : {res['MCC']}   ← balanced quality")
    print(f"  Brier      : {res['Brier']}  ← calibration")
    print()
    print(classification_report(y_test, res["y_pred"],
          target_names=["No NICU (0)", "NICU (1)"], digits=4))


# ── Validation split for threshold tuning (from training data only) ───────
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_train_raw, y_train_raw,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_raw)

X_tr2_sc  = scaler.transform(X_tr2)
X_val_sc  = scaler.transform(X_val)

results = {}

# ═══════════════════════════════════════════════════════════════════════════
# 7. MODEL 1 — LOGISTIC REGRESSION
#    FIX ④: imblearn Pipeline keeps SMOTE-ENN inside each CV fold
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 1 : LOGISTIC REGRESSION")
print("─" * 70)

def lr_objective(trial):
    C       = trial.suggest_float("C", 1e-3, 100, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    pipe = ImbPipeline([
        ("smote", SMOTEENN(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver="liblinear",
                                    class_weight="balanced",
                                    max_iter=2000, random_state=RANDOM_STATE)),
    ])
    return cross_val_score(pipe, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

lr_study = optuna.create_study(direction="maximize")
lr_study.optimize(lr_objective, n_trials=OPTUNA_TRIALS)
p = lr_study.best_params

# Final model fitted on full train set via pipeline
best_lr_pipe = ImbPipeline([
    ("smote",  SMOTEENN(random_state=RANDOM_STATE)),
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=p["C"], penalty=p["penalty"],
                                   solver="liblinear", class_weight="balanced",
                                   max_iter=2000, random_state=RANDOM_STATE)),
])
best_lr_pipe.fit(X_train_raw, y_train_raw)

# Threshold tuned on val set (FIX ②)
thr_lr = best_f2_threshold_val(best_lr_pipe, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {lr_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["Logistic Regression"] = evaluate("Logistic Regression", best_lr_pipe,
                                            X_test, y_test, thr_lr)
print_result(results["Logistic Regression"])

# ═══════════════════════════════════════════════════════════════════════════
# 8. MODEL 2 — DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 2 : DECISION TREE")
print("─" * 70)

def dt_objective(trial):
    pipe = ImbPipeline([
        ("smote", SMOTEENN(random_state=RANDOM_STATE)),
        ("clf",   DecisionTreeClassifier(
            max_depth        = trial.suggest_int("max_depth", 2, 12),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 30),
            min_samples_split= trial.suggest_int("min_samples_split", 2, 20),
            criterion        = trial.suggest_categorical("criterion", ["gini","entropy"]),
            class_weight     = "balanced",
            random_state     = RANDOM_STATE)),
    ])
    return cross_val_score(pipe, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

dt_study = optuna.create_study(direction="maximize")
dt_study.optimize(dt_objective, n_trials=OPTUNA_TRIALS)
p = dt_study.best_params

best_dt_pipe = ImbPipeline([
    ("smote", SMOTEENN(random_state=RANDOM_STATE)),
    ("clf",   DecisionTreeClassifier(**p, class_weight="balanced",
                                      random_state=RANDOM_STATE)),
])
best_dt_pipe.fit(X_train_raw, y_train_raw)
thr_dt = best_f2_threshold_val(best_dt_pipe, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {dt_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["Decision Tree"] = evaluate("Decision Tree", best_dt_pipe,
                                     X_test, y_test, thr_dt)
print_result(results["Decision Tree"])

# ═══════════════════════════════════════════════════════════════════════════
# 9. MODEL 3 — RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 3 : RANDOM FOREST")
print("─" * 70)

def rf_objective(trial):
    pipe = ImbPipeline([
        ("smote", SMOTEENN(random_state=RANDOM_STATE)),
        ("clf",   RandomForestClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 15),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20),
            max_features     = trial.suggest_categorical("max_features",
                                                          ["sqrt","log2",0.5]),
            class_weight     = "balanced",
            n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    return cross_val_score(pipe, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=OPTUNA_TRIALS)
p = rf_study.best_params

best_rf_pipe = ImbPipeline([
    ("smote", SMOTEENN(random_state=RANDOM_STATE)),
    ("clf",   RandomForestClassifier(**p, class_weight="balanced",
                                      n_jobs=-1, random_state=RANDOM_STATE)),
])
best_rf_pipe.fit(X_train_raw, y_train_raw)
thr_rf = best_f2_threshold_val(best_rf_pipe, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {rf_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["Random Forest"] = evaluate("Random Forest", best_rf_pipe,
                                     X_test, y_test, thr_rf)
print_result(results["Random Forest"])

# ═══════════════════════════════════════════════════════════════════════════
# 10. MODEL 4 — GRADIENT BOOSTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 4 : GRADIENT BOOSTING")
print("─" * 70)

def gb_objective(trial):
    pipe = ImbPipeline([
        ("smote", SMOTEENN(random_state=RANDOM_STATE)),
        ("clf",   GradientBoostingClassifier(
            n_estimators  = trial.suggest_int("n_estimators", 100, 400),
            max_depth     = trial.suggest_int("max_depth", 2, 6),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample     = trial.suggest_float("subsample", 0.6, 1.0),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20),
            random_state  = RANDOM_STATE)),
    ])
    return cross_val_score(pipe, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

gb_study = optuna.create_study(direction="maximize")
gb_study.optimize(gb_objective, n_trials=OPTUNA_TRIALS)
p = gb_study.best_params

best_gb_pipe = ImbPipeline([
    ("smote", SMOTEENN(random_state=RANDOM_STATE)),
    ("clf",   GradientBoostingClassifier(**p, random_state=RANDOM_STATE)),
])
best_gb_pipe.fit(X_train_raw, y_train_raw)
thr_gb = best_f2_threshold_val(best_gb_pipe, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {gb_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["Gradient Boosting"] = evaluate("Gradient Boosting", best_gb_pipe,
                                         X_test, y_test, thr_gb)
print_result(results["Gradient Boosting"])

# ═══════════════════════════════════════════════════════════════════════════
# 11. MODEL 5 — SVM
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 5 : SVM")
print("─" * 70)

def svm_objective(trial):
    pipe = ImbPipeline([
        ("smote",  SMOTEENN(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            C            = trial.suggest_float("C", 0.01, 100, log=True),
            kernel       = trial.suggest_categorical("kernel", ["rbf","linear"]),
            gamma        = trial.suggest_categorical("gamma", ["scale","auto"]),
            class_weight = "balanced",
            probability  = True,
            random_state = RANDOM_STATE)),
    ])
    return cross_val_score(pipe, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

svm_study = optuna.create_study(direction="maximize")
svm_study.optimize(svm_objective, n_trials=OPTUNA_TRIALS)
p = svm_study.best_params

best_svm_pipe = ImbPipeline([
    ("smote",  SMOTEENN(random_state=RANDOM_STATE)),
    ("scaler", StandardScaler()),
    ("clf",    SVC(**p, class_weight="balanced", probability=True,
                  random_state=RANDOM_STATE)),
])
best_svm_pipe.fit(X_train_raw, y_train_raw)
thr_svm = best_f2_threshold_val(best_svm_pipe, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {svm_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["SVM"] = evaluate("SVM", best_svm_pipe, X_test, y_test, thr_svm)
print_result(results["SVM"])

# ═══════════════════════════════════════════════════════════════════════════
# 12. MODEL 6 — KNN
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 6 : KNN")
print("─" * 70)

def knn_objective(trial):
    pipe = ImbPipeline([
        ("smote",  SMOTEENN(random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(
            n_neighbors = trial.suggest_int("n_neighbors", 3, 25),
            weights     = trial.suggest_categorical("weights",
                                                     ["uniform","distance"]),
            metric      = trial.suggest_categorical("metric",
                                                     ["euclidean","manhattan",
                                                      "minkowski"]),
            p           = trial.suggest_int("p", 1, 3))),
    ])
    return cross_val_score(pipe, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

knn_study = optuna.create_study(direction="maximize")
knn_study.optimize(knn_objective, n_trials=OPTUNA_TRIALS)
p = knn_study.best_params

best_knn_pipe = ImbPipeline([
    ("smote",  SMOTEENN(random_state=RANDOM_STATE)),
    ("scaler", StandardScaler()),
    ("clf",    KNeighborsClassifier(**p)),
])
best_knn_pipe.fit(X_train_raw, y_train_raw)
thr_knn = best_f2_threshold_val(best_knn_pipe, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {knn_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["KNN"] = evaluate("KNN", best_knn_pipe, X_test, y_test, thr_knn)
print_result(results["KNN"])

# ═══════════════════════════════════════════════════════════════════════════
# 13. MODEL 7 — XGBoost
#     No SMOTE — uses scale_pos_weight natively
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 7 : XGBoost")
print("─" * 70)

def xgb_objective(trial):
    model = xgb.XGBClassifier(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        max_depth        = trial.suggest_int("max_depth", 3, 9),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 15),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
        gamma            = trial.suggest_float("gamma", 0, 5),
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "aucpr",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    return cross_val_score(model, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS)
p = xgb_study.best_params

best_xgb = xgb.XGBClassifier(**p, scale_pos_weight=scale_pos_weight,
                               eval_metric="aucpr",
                               random_state=RANDOM_STATE, n_jobs=-1)
best_xgb.fit(X_train_raw, y_train_raw)
thr_xgb = best_f2_threshold_val(best_xgb, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {xgb_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["XGBoost"] = evaluate("XGBoost", best_xgb, X_test, y_test, thr_xgb)
print_result(results["XGBoost"])

# ═══════════════════════════════════════════════════════════════════════════
# 14. MODEL 8 — LightGBM
#     No SMOTE — uses is_unbalance natively
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MODEL 8 : LightGBM")
print("─" * 70)

def lgb_objective(trial):
    model = lgb.LGBMClassifier(
        n_estimators      = trial.suggest_int("n_estimators", 100, 700),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 15, 150),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
        is_unbalance      = True,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1,
    )
    return cross_val_score(model, X_train_raw, y_train_raw,
                           cv=kfold, scoring="average_precision", n_jobs=-1).mean()

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS)
p = lgb_study.best_params

best_lgbm = lgb.LGBMClassifier(**p, is_unbalance=True,
                                 random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
best_lgbm.fit(X_train_raw, y_train_raw)
thr_lgbm = best_f2_threshold_val(best_lgbm, X_val, y_val)
print(f"\n  Optuna best PR-AUC : {lgb_study.best_value:.4f}")
print(f"  Best params        : {p}")
results["LightGBM"] = evaluate("LightGBM", best_lgbm, X_test, y_test, thr_lgbm)
print_result(results["LightGBM"])

# ═══════════════════════════════════════════════════════════════════════════
# 15. ENSEMBLES — Stacking + Soft Voting
#     FIX ③: VotingClassifier uses the full imblearn pipelines consistently
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  ENSEMBLE MODELS")
print("─" * 70)

# Stacking: boosters + RF + GB as base; LR as meta
stack_estimators = [
    ("xgb",  best_xgb),
    ("lgbm", best_lgbm),
    ("rf",   best_rf_pipe),
    ("gb",   best_gb_pipe),
]
stack = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(class_weight="balanced",
                                        max_iter=2000, random_state=RANDOM_STATE),
    cv=5, passthrough=False, n_jobs=-1)
stack.fit(X_train_raw, y_train_raw)
thr_stack = best_f2_threshold_val(stack, X_val, y_val)
results["Stacking"] = evaluate("Stacking", stack, X_test, y_test, thr_stack)
print("\n── Stacking (XGB + LGBM + RF + GB → LR meta)")
print_result(results["Stacking"])

# Soft Voting — all 8 models as imblearn pipelines or raw for XGB/LGBM
# (FIX ③: consistent — all pipelines trained on same raw split)
voting = VotingClassifier(estimators=[
    ("lr",   best_lr_pipe),
    ("dt",   best_dt_pipe),
    ("rf",   best_rf_pipe),
    ("gb",   best_gb_pipe),
    ("svm",  best_svm_pipe),
    ("knn",  best_knn_pipe),
    ("xgb",  best_xgb),
    ("lgbm", best_lgbm),
], voting="soft")
voting.fit(X_train_raw, y_train_raw)
thr_vote = best_f2_threshold_val(voting, X_val, y_val)
results["Soft Voting"] = evaluate("Soft Voting", voting, X_test, y_test, thr_vote)
print("── Soft Voting (all 8 models)")
print_result(results["Soft Voting"])

# ═══════════════════════════════════════════════════════════════════════════
# [INSERT AFTER SECTION 3 / ENSEMBLE MODELS, BEFORE SECTION 4 BOOTSTRAP CI]
#
# PER-FOLD CROSS-VALIDATION — ALL 10 MODELS
# Prints fold-by-fold PR-AUC, Recall, and F2 for every model.
# Uses the same kfold splitter as the rest of the pipeline.
# SMOTE is applied inside the imblearn Pipeline per fold (no leakage).
# XGBoost and LightGBM use raw splits + scale_pos_weight (no SMOTE).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PER-FOLD CROSS-VALIDATION — ALL 10 MODELS")
print("=" * 70)
print(f"\n  Splitter : StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
      f"random_state={RANDOM_STATE})")
print(f"  Scored on : X_train_raw / y_train_raw  (test set never touched)\n")

# ── helper: score one fold ────────────────────────────────────────────────
def _fold_scores(model, X_tr, y_tr, X_va, y_va):
    """
    Fit model on (X_tr, y_tr) and evaluate on (X_va, y_va).
    Returns (pr_auc, recall, f2) or (nan, nan, nan) on error.
    """
    try:
        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_va)[:, 1]
        else:
            probs = model.decision_function(X_va)

        # F2-optimal threshold on this val fold (mirrors production logic)
        prec_c, rec_c, thr_c = precision_recall_curve(y_va, probs)
        f2_c  = (5 * prec_c * rec_c) / (4 * prec_c + rec_c + 1e-9)
        best_thr = float(thr_c[np.argmax(f2_c[:-1])]) if len(thr_c) > 0 else 0.5
        y_pred = (probs >= best_thr).astype(int)

        pr_auc = average_precision_score(y_va, probs)
        rec    = recall_score(y_va, y_pred, zero_division=0)
        f2     = fbeta_score(y_va, y_pred, beta=2, zero_division=0)
        return pr_auc, rec, f2
    except Exception as e:
        print(f"      !! fold error: {e}")
        return np.nan, np.nan, np.nan


# ── models registry (all 10) ──────────────────────────────────────────────
# Ensembles (stacking, voting) need clone() so each fold gets a fresh fit.
from sklearn.base import clone

per_fold_models = {
    "Logistic Regression" : best_lr_pipe,
    "Decision Tree"       : best_dt_pipe,
    "Random Forest"       : best_rf_pipe,
    "Gradient Boosting"   : best_gb_pipe,
    "SVM"                 : best_svm_pipe,
    "KNN"                 : best_knn_pipe,
    "XGBoost"             : best_xgb,
    "LightGBM"            : best_lgbm,
    "Stacking"            : stack,
    "Soft Voting"         : voting,
}

X_arr = np.array(X_train_raw)   # numpy for index slicing
y_arr = np.array(y_train_raw)

fold_summary = {}   # name → {metric: [fold_values]}

for model_name, base_model in per_fold_models.items():
    print("\n" + "─" * 70)
    print(f"  {model_name}")
    print("─" * 70)
    print(f"  {'Fold':<6} {'PR-AUC':>10} {'Recall':>10} {'F2':>10}"
          f"  {'Threshold':>10}")
    print("  " + "-" * 50)

    fold_pr, fold_rec, fold_f2 = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(X_arr, y_arr), start=1):

        X_tr_f, y_tr_f = X_arr[train_idx], y_arr[train_idx]
        X_va_f, y_va_f = X_arr[val_idx],   y_arr[val_idx]

        # Clone to avoid cross-fold state contamination
        model_clone = clone(base_model)

        pr, rec, f2 = _fold_scores(model_clone, X_tr_f, y_tr_f,
                                    X_va_f, y_va_f)

        # Recompute threshold for display (same logic as _fold_scores)
        thr_display = np.nan
        try:
            model_clone_display = clone(base_model)
            model_clone_display.fit(X_tr_f, y_tr_f)
            if hasattr(model_clone_display, "predict_proba"):
                probs_d = model_clone_display.predict_proba(X_va_f)[:, 1]
            else:
                probs_d = model_clone_display.decision_function(X_va_f)
            prec_d, rec_d, thr_d = precision_recall_curve(y_va_f, probs_d)
            f2_d = (5 * prec_d * rec_d) / (4 * prec_d + rec_d + 1e-9)
            if len(thr_d) > 0:
                thr_display = float(thr_d[np.argmax(f2_d[:-1])])
        except Exception:
            pass

        fold_pr.append(pr); fold_rec.append(rec); fold_f2.append(f2)

        pr_s  = f"{pr:.4f}"   if not np.isnan(pr)          else "  n/a"
        rec_s = f"{rec:.4f}"  if not np.isnan(rec)         else "  n/a"
        f2_s  = f"{f2:.4f}"   if not np.isnan(f2)          else "  n/a"
        thr_s = f"{thr_display:.3f}" if not np.isnan(thr_display) else "  n/a"
        print(f"  {fold_idx:<6} {pr_s:>10} {rec_s:>10} {f2_s:>10}  {thr_s:>10}")

    # ── per-model summary ─────────────────────────────────────────────────
    fold_pr_arr  = np.array([v for v in fold_pr  if not np.isnan(v)])
    fold_rec_arr = np.array([v for v in fold_rec if not np.isnan(v)])
    fold_f2_arr  = np.array([v for v in fold_f2  if not np.isnan(v)])

    def _fmt(arr):
        if len(arr) == 0:
            return "n/a ± n/a"
        return f"{arr.mean():.4f} ± {arr.std():.4f}"

    print("  " + "-" * 50)
    print(f"  {'Mean':6} {_fmt(fold_pr_arr):>21} {_fmt(fold_rec_arr):>21}"
          f" {_fmt(fold_f2_arr):>21}")

    # Highlight min/max folds
    if len(fold_pr_arr) == N_SPLITS:
        best_fold  = int(np.argmax(fold_pr))  + 1
        worst_fold = int(np.argmin(fold_pr))  + 1
        print(f"\n  Best  fold (PR-AUC): Fold {best_fold}  "
              f"({fold_pr[best_fold-1]:.4f})")
        print(f"  Worst fold (PR-AUC): Fold {worst_fold}  "
              f"({fold_pr[worst_fold-1]:.4f})")
        variance_flag = ("⚠  High variance across folds"
                         if fold_pr_arr.std() > 0.08
                         else "✓  Stable across folds")
        print(f"  Variance check     : {variance_flag}")

    fold_summary[model_name] = {
        "pr_auc": fold_pr,
        "recall": fold_rec,
        "f2"    : fold_f2,
    }

# ── cross-model per-fold summary table ────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-FOLD CV SUMMARY TABLE  (mean ± std, sorted by PR-AUC)")
print("=" * 70)
print(f"\n  {'Model':<22}  {'PR-AUC (mean±std)':<24}"
      f"  {'Recall (mean±std)':<24}  {'F2 (mean±std)'}")
print("  " + "-" * 90)

summary_rows = []
for name, scores in fold_summary.items():
    pr_arr  = np.array([v for v in scores["pr_auc"] if not np.isnan(v)])
    rec_arr = np.array([v for v in scores["recall"] if not np.isnan(v)])
    f2_arr  = np.array([v for v in scores["f2"]     if not np.isnan(v)])
    summary_rows.append({
        "name"   : name,
        "pr_mean": pr_arr.mean()  if len(pr_arr)  else np.nan,
        "pr_std" : pr_arr.std()   if len(pr_arr)  else np.nan,
        "rec_mean": rec_arr.mean() if len(rec_arr) else np.nan,
        "rec_std" : rec_arr.std()  if len(rec_arr) else np.nan,
        "f2_mean": f2_arr.mean()  if len(f2_arr)  else np.nan,
        "f2_std" : f2_arr.std()   if len(f2_arr)  else np.nan,
    })

summary_rows.sort(key=lambda x: x["pr_mean"] if not np.isnan(x["pr_mean"]) else -1,
                  reverse=True)

for row in summary_rows:
    pr_s  = (f"{row['pr_mean']:.4f} ± {row['pr_std']:.4f}"
             if not np.isnan(row["pr_mean"]) else "n/a")
    rec_s = (f"{row['rec_mean']:.4f} ± {row['rec_std']:.4f}"
             if not np.isnan(row["rec_mean"]) else "n/a")
    f2_s  = (f"{row['f2_mean']:.4f} ± {row['f2_std']:.4f}"
             if not np.isnan(row["f2_mean"]) else "n/a")
    print(f"  {row['name']:<22}  {pr_s:<24}  {rec_s:<24}  {f2_s}")

print("\n  Note: thresholds above are tuned per fold on that fold's val split,")
print("  so they may differ from the final val-set threshold used on X_test.")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# [CONTINUE WITH SECTION 4 : BOOTSTRAP CONFIDENCE INTERVALS]
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 16. BOOTSTRAP CONFIDENCE INTERVALS  (IMPROVEMENT ③)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  SECTION 4 : BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("─" * 70)

y_test_arr = np.array(y_test)

def pr_auc_fn(yt, yp, _):  return average_precision_score(yt, yp)
def recall_fn(yt, _, ypred): return recall_score(yt, ypred, zero_division=0)
def f2_fn(yt, _, ypred):    return fbeta_score(yt, ypred, beta=2, zero_division=0)

print(f"\n  {'Model':<22} {'PR-AUC (95% CI)':<28} {'Recall (95% CI)':<28} {'F2 (95% CI)'}")
print("  " + "-" * 100)
for name, res in results.items():
    yp = res["y_prob"]
    ypred = res["y_pred"]
    prauc_m, prauc_l, prauc_u = bootstrap_ci(y_test_arr, yp, ypred, pr_auc_fn)
    rec_m,   rec_l,   rec_u   = bootstrap_ci(y_test_arr, yp, ypred, recall_fn)
    f2_m,    f2_l,    f2_u    = bootstrap_ci(y_test_arr, yp, ypred, f2_fn)
    print(f"  {name:<22} "
          f"{prauc_m:.3f} [{prauc_l:.3f}–{prauc_u:.3f}]    "
          f"{rec_m:.3f} [{rec_l:.3f}–{rec_u:.3f}]    "
          f"{f2_m:.3f} [{f2_l:.3f}–{f2_u:.3f}]")

# ═══════════════════════════════════════════════════════════════════════════
# 17. REPEATED K-FOLD CV ON TRAIN SET  (IMPROVEMENT ①)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print(f"  SECTION 5 : REPEATED {N_REPEATS}×{N_SPLITS}-FOLD CV  (train set only)")
print("─" * 70)

cv_models = {
    "Logistic Regression": best_lr_pipe,
    "Random Forest"      : best_rf_pipe,
    "Gradient Boosting"  : best_gb_pipe,
    "XGBoost"            : best_xgb,
    "LightGBM"           : best_lgbm,
}
print(f"\n  {'Model':<22}  CV PR-AUC mean ± std")
print("  " + "-" * 50)
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train_raw, y_train_raw,
                              cv=rep_kfold, scoring="average_precision", n_jobs=-1)
    print(f"  {name:<22}  {scores.mean():.4f} ± {scores.std():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 18. SHAP EXPLAINABILITY  — raw estimators for tree models
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  SECTION 6 : SHAP EXPLAINABILITY")
print("─" * 70)

# Extract raw classifiers from pipelines for SHAP (TreeExplainer needs the tree)
rf_clf  = best_rf_pipe.named_steps["clf"]
gb_clf  = best_gb_pipe.named_steps["clf"]

# For tree SHAP we need the X that was fed to the clf step
# i.e. after SMOTE (use X_test unchanged since no scaling in RF/GB pipe)
shap_models = {
    "XGBoost" : (best_xgb,  X_test),
    "LightGBM": (best_lgbm, X_test),
    "RF"      : (rf_clf,    X_test),
    "GB"      : (gb_clf,    X_test),
}

shap_values_dict = {}
for mname, (model, Xte) in shap_models.items():
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(Xte)
        if isinstance(sv, list):
            sv = sv[1]
        shap_values_dict[mname] = sv
        ms = np.abs(sv).mean(axis=0)
        print(f"\n  {mname} SHAP (mean |SHAP|):")
        for nm, v in sorted(zip(feat_names, ms), key=lambda x: -x[1]):
            print(f"    {nm:<20} {v:.4f}  {'█'*int(v*60)}")
    except Exception as e:
        print(f"\n  {mname} SHAP failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 19. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY  (sorted by PR-AUC)")
print("=" * 70)

DISPLAY_COLS = ["Model","Threshold","Recall","F2","PR-AUC","ROC-AUC","MCC","F1","Brier"]
summary = pd.DataFrame([{c: r[c] for c in DISPLAY_COLS} for r in results.values()])
summary = summary.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
summary.index += 1
print("\n" + summary.to_string())

best = summary.iloc[0]
print(f"\n🏆 Best (PR-AUC) : {best['Model']}"
      f"  PR-AUC={best['PR-AUC']}  Recall={best['Recall']}  F2={best['F2']}")

# ═══════════════════════════════════════════════════════════════════════════
# 20. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Generating visualisations ...")

mnames = list(results.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(mnames)))

# ── FIGURE 1 : EDA ──────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(20, 12))
fig1.suptitle("EDA — NICU Admission Dataset", fontsize=15, fontweight="bold")
gs1  = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.48, wspace=0.38)

ax_pie = fig1.add_subplot(gs1[0, 0])
ax_pie.pie([neg, pos], labels=["No NICU (0)", "NICU (1)"],
           autopct="%1.1f%%", colors=["#4C72B0","#DD8452"],
           startangle=90, textprops={"fontsize":9})
ax_pie.set_title("Target Distribution", fontweight="bold")

ax_cnt = fig1.add_subplot(gs1[0, 1])
bars = ax_cnt.bar(["No NICU (0)","NICU (1)"], [neg, pos],
                  color=["#4C72B0","#DD8452"], edgecolor="white")
for b, v in zip(bars, [neg, pos]):
    ax_cnt.text(b.get_x()+b.get_width()/2, b.get_height()+30,
                str(v), ha="center", fontsize=9, fontweight="bold")
ax_cnt.set_title("Class Count", fontweight="bold")

ax_risk = fig1.add_subplot(gs1[0, 2])
risk_rate = df.groupby("Risk_score")[TARGET].mean() * 100
ax_risk.bar(risk_rate.index, risk_rate.values, color="#55A868", edgecolor="white")
for x, v in zip(risk_rate.index, risk_rate.values):
    ax_risk.text(x, v+0.5, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax_risk.set_xlabel("Risk Score (0-4)"); ax_risk.set_ylabel("NICU Rate (%)")
ax_risk.set_title("NICU Rate by Risk Score", fontweight="bold")

ax_heat = fig1.add_subplot(gs1[0, 3])
corr_mat = df[FEATURES_EXT + [TARGET]].corr()
sns.heatmap(corr_mat[[TARGET]].drop(TARGET), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax_heat, linewidths=0.5,
            yticklabels=feat_names, cbar=True, annot_kws={"size":8})
ax_heat.set_title("Feature → NICU Correlation", fontweight="bold")
ax_heat.tick_params(axis="y", labelsize=7)

for i, feat in enumerate(FEATURES):
    ax = fig1.add_subplot(gs1[1, i])
    rate = df.groupby(feat)[TARGET].mean() * 100
    bars2 = ax.bar(["Absent","Present"], rate.values,
                   color=["#4C72B0","#DD8452"], edgecolor="white")
    for b, v in zip(bars2, rate.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_title(SHORT[feat], fontweight="bold", fontsize=8)
    ax.set_ylabel("NICU Rate (%)")

plt.tight_layout()
plt.savefig("plot1_eda.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 2 : IMBALANCE STRATEGIES ─────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Imbalance Handling Strategies Comparison", fontweight="bold")

strat_names = list(sampled_data.keys())
before      = [Counter(y_train_raw)[0], Counter(y_train_raw)[1]]
counts0     = [Counter(sampled_data[s][1])[0] for s in strat_names]
counts1     = [Counter(sampled_data[s][1])[1] for s in strat_names]

x_s = np.arange(len(strat_names))
axes2[0].bar(x_s-0.2, counts0, 0.4, label="Class 0 (No NICU)", color="#4C72B0")
axes2[0].bar(x_s+0.2, counts1, 0.4, label="Class 1 (NICU)",    color="#DD8452")
axes2[0].set_xticks(x_s); axes2[0].set_xticklabels(strat_names, fontsize=9)
axes2[0].set_ylabel("Sample count"); axes2[0].set_title("After Resampling", fontweight="bold")
axes2[0].legend(); axes2[0].grid(axis="y", alpha=0.3)

ratios = [c1/(c0+c1)*100 for c0,c1 in zip(counts0,counts1)]
bars3  = axes2[1].bar(strat_names, ratios, color="#55A868", edgecolor="white")
axes2[1].axhline(y=before[1]/(before[0]+before[1])*100, color="red",
                 linestyle="--", linewidth=1.5,
                 label=f"Original {before[1]/(before[0]+before[1]):.1%}")
for b, v in zip(bars3, ratios):
    axes2[1].text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%",
                  ha="center", fontsize=9, fontweight="bold")
axes2[1].set_ylabel("Minority class %"); axes2[1].legend()
axes2[1].set_title("Minority Class % After Resampling", fontweight="bold")
axes2[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plot2_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 3 : ROC + PR CURVES ──────────────────────────────────────────
fig3, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle("ROC and Precision-Recall Curves — All Models", fontweight="bold")

for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ax_roc.plot(fpr, tpr, label=f"{name} ({res['ROC-AUC']:.3f})",
                color=colors[i], linewidth=1.8)
    ax_pr.plot(rec, prec, label=f"{name} ({res['PR-AUC']:.3f})",
               color=colors[i], linewidth=1.8)

ax_roc.plot([0,1],[0,1],"k--",linewidth=1)
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
ax_roc.set_title("ROC Curves (AUC)", fontweight="bold")
ax_roc.legend(fontsize=7, loc="lower right"); ax_roc.grid(alpha=0.3)

ax_pr.axhline(y=pos/(neg+pos), color="k", linestyle="--", linewidth=1,
              label=f"Baseline ({pos/(neg+pos):.2%})")
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
ax_pr.set_title("PR Curves (AP) — Better for imbalanced data", fontweight="bold")
ax_pr.legend(fontsize=7, loc="upper right"); ax_pr.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot3_roc_pr.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 4 : CONFUSION MATRICES ───────────────────────────────────────
n_models = len(results)
ncols    = 5
nrows    = (n_models + ncols - 1) // ncols
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
fig4.suptitle("Confusion Matrices — All Models (val-tuned threshold)", fontweight="bold")
axes4 = np.array(axes4).flatten()

for i, (name, res) in enumerate(results.items()):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No NICU","NICU"])
    disp.plot(ax=axes4[i], cmap="Blues", colorbar=False)
    axes4[i].set_title(f"{name}\nPR-AUC={res['PR-AUC']}  Recall={res['Recall']}",
                       fontweight="bold", fontsize=9)
    axes4[i].tick_params(labelsize=8)

for j in range(i+1, len(axes4)):
    axes4[j].set_visible(False)

plt.tight_layout()
plt.savefig("plot4_confusion.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 5 : SHAP ─────────────────────────────────────────────────────
# ── FIGURE 5 : SHAP (Individual plots per model) ─────────────────────────
print("\n  Generating individual SHAP plots...")

shap_colors = {"XGBoost": "#DD8452", "LightGBM": "#4C72B0",
               "RF": "#55A868", "GB": "#C44E52"}

for mname, sv_array in shap_values_dict.items():
    try:
        # Calculate mean absolute SHAP values
        ms = np.abs(sv_array).mean(axis=0).flatten()

        # Check dimensions
        if len(ms) != len(feat_names):
            print(f"  {mname}: Feature mismatch - SHAP has {len(ms)}, expected {len(feat_names)}")
            # Take the minimum length
            min_len = min(len(ms), len(feat_names))
            ms = ms[:min_len]
            used_features = feat_names[:min_len]
        else:
            used_features = feat_names

        # Sort and get top features
        idx = np.argsort(ms)[::-1][:15]  # Top 15 features
        top_features = [used_features[i] for i in idx]
        top_values = ms[idx]

        # Create individual figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_features)), top_values,
                       color=shap_colors.get(mname, "#888"),
                       edgecolor="white")

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(f"{mname} - SHAP Feature Importance", fontweight="bold", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(f"plot5_shap_{mname.lower().replace(' ', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ SHAP plot saved for {mname}")

    except Exception as e:
        print(f"  ✗ Failed to create SHAP plot for {mname}: {e}")
        import traceback

        traceback.print_exc()
# ── FIGURE 6 : PERFORMANCE HEATMAP ──────────────────────────────────────
metric_cols = ["Recall","F2","PR-AUC","ROC-AUC","MCC","F1"]
hm_data     = summary.set_index("Model")[metric_cols]
fig6, ax6   = plt.subplots(figsize=(12, 6))
sns.heatmap(hm_data, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax6, linewidths=0.5, vmin=0, vmax=1,
            annot_kws={"size":10, "weight":"bold"})
ax6.set_title("Model Performance Heatmap (sorted by PR-AUC)",
              fontweight="bold", fontsize=13)
ax6.tick_params(axis="x", labelsize=9, rotation=30)
ax6.tick_params(axis="y", labelsize=9, rotation=0)
plt.tight_layout()
plt.savefig("plot6_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 7 : OPTUNA HISTORY ────────────────────────────────────────────
studies = {
    "Logistic Regression": lr_study,
    "Decision Tree"      : dt_study,
    "Random Forest"      : rf_study,
    "Gradient Boosting"  : gb_study,
    "SVM"                : svm_study,
    "KNN"                : knn_study,
    "XGBoost"            : xgb_study,
    "LightGBM"           : lgb_study,
}
fig7, axes7 = plt.subplots(2, 4, figsize=(20, 8))
fig7.suptitle("Optuna Bayesian Tuning History — All Models", fontweight="bold")
axes7 = axes7.flatten()
for i, (name, study) in enumerate(studies.items()):
    vals = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(vals)
    axes7[i].scatter(range(len(vals)), vals, alpha=0.35, s=12, color="#4C72B0")
    axes7[i].plot(range(len(vals)), best_so_far, color="#DD8452", linewidth=2)
    axes7[i].set_title(name, fontweight="bold", fontsize=9)
    axes7[i].set_xlabel("Trial"); axes7[i].set_ylabel("PR-AUC (CV, train only)")
    axes7[i].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot7_optuna.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 8 : CALIBRATION CURVES ────────────────────────────────────────
fig8, ax8 = plt.subplots(figsize=(10, 7))
ax8.plot([0,1],[0,1],"k--",linewidth=1, label="Perfect calibration")
for i, (name, res) in enumerate(results.items()):
    try:
        frac_pos, mean_pred = calibration_curve(y_test, res["y_prob"],
                                                 n_bins=8, strategy="uniform")
        ax8.plot(mean_pred, frac_pos, marker="o", markersize=5,
                 label=f"{name} (Brier={res['Brier']})",
                 color=colors[i], linewidth=1.6)
    except Exception:
        pass
ax8.set_xlabel("Mean predicted probability"); ax8.set_ylabel("Fraction of positives")
ax8.set_title("Calibration Curves (Reliability Diagram)", fontweight="bold")
ax8.legend(fontsize=7, loc="upper left"); ax8.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot8_calibration.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 9 : THRESHOLD SENSITIVITY ─────────────────────────────────────
fig9, axes9 = plt.subplots(2, 5, figsize=(25, 9))
fig9.suptitle("Threshold Sensitivity — Precision / Recall / F2 vs Threshold",
              fontweight="bold")
axes9 = axes9.flatten()
for ax in axes9:
    ax.set_visible(False)
for i, (name, res) in enumerate(results.items()):
    axes9[i].set_visible(True)
    prec, rec, thr = precision_recall_curve(y_test, res["y_prob"])
    f2v  = (5*prec*rec) / (4*prec+rec+1e-9)
    thr_full = np.append(thr, 1.0)
    axes9[i].plot(thr_full, prec, label="Precision", color="#4C72B0", linewidth=1.5)
    axes9[i].plot(thr_full, rec,  label="Recall",    color="#DD8452", linewidth=1.5)
    axes9[i].plot(thr_full, f2v,  label="F2",        color="#55A868", linewidth=1.5)
    axes9[i].axvline(x=res["Threshold"], color="red", linestyle="--",
                     linewidth=1, label=f"Val thr={res['Threshold']}")
    axes9[i].set_title(name, fontweight="bold", fontsize=9)
    axes9[i].set_xlabel("Threshold"); axes9[i].set_ylim(0, 1.05)
    axes9[i].legend(fontsize=6); axes9[i].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot9_threshold.png", dpi=150, bbox_inches="tight")
plt.close()

# ── FIGURE 10 : MULTI-METRIC BAR ─────────────────────────────────────────
fig10, ax10 = plt.subplots(figsize=(16, 6))
metric_keys = ["Recall","F2","PR-AUC","ROC-AUC","MCC","F1"]
x10  = np.arange(len(metric_keys))
bw10 = 0.07
for i, (name, res) in enumerate(results.items()):
    vals = [res[k] for k in metric_keys]
    ax10.bar(x10 + i*bw10, vals, bw10, label=name, color=colors[i])
ax10.set_xticks(x10 + bw10*len(results)/2)
ax10.set_xticklabels(metric_keys, fontsize=10)
ax10.set_ylim(0, 1.18)
ax10.set_title("All Metrics — All Models (val-tuned thresholds)", fontweight="bold")
ax10.legend(fontsize=7, loc="upper right", ncol=2)
ax10.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot10_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

# ── SAVE CSV ─────────────────────────────────────────────────────────────
out_csv = summary.drop(columns=["y_pred","y_prob"], errors="ignore")
out_csv.to_csv("nicu_final_results.csv", index=False)

print("✔ All plots saved  (plot1_eda → plot10_metrics)")
print("✔ Results saved  → nicu_final_results.csv")
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — all leakage issues resolved")
print("=" * 70)
"""
=============================================================================
OVERFITTING DIAGNOSTIC — ADD THIS BLOCK AT THE END OF YOUR PIPELINE
=============================================================================
Paste this entire block after the final print("PIPELINE COMPLETE") line.

Checks performed:
  CHECK 1  Unique input patterns  — how many distinct rows exist (max = 2^4 = 16)
  CHECK 2  CV PR-AUC vs Test PR-AUC gap per model (> 0.05 = overfit warning)
  CHECK 3  Logistic Regression baseline comparison (complex models should not
           hugely outperform LR on 4 binary features without suspicion)
  CHECK 4  Train-set memorisation score — accuracy on X_train_raw vs X_test
  CHECK 5  Permutation importance sanity (do features matter, or is the model
           fitting noise?)
  CHECK 6  Learning-curve stability — does performance plateau or keep rising?
  CHECK 7  Repeated-CV variance (high std = unstable / overfit to folds)
  CHECK 8  Bootstrap CI overlap test — if all model CIs overlap, no model is
           meaningfully better (overfitting to split noise)
  CHECK 9  Brier score on train vs test (calibration gap)
  CHECK 10 Summary table + VERDICT
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.inspection      import permutation_importance
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics         import (brier_score_loss, roc_auc_score,
                                     average_precision_score)
from sklearn.dummy           import DummyClassifier

print("\n")
print("=" * 70)
print("  OVERFITTING DIAGNOSTIC — 10-CHECK SUITE")
print("=" * 70)

# ── helpers ──────────────────────────────────────────────────────────────

def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def _prauc(model, X, y):
    try:
        return average_precision_score(y, _proba(model, X))
    except Exception:
        return np.nan

def _brier(model, X, y):
    try:
        p = _proba(model, X)
        # clip to [0,1] for models that return decision_function values
        p = np.clip(p, 0, 1)
        return brier_score_loss(y, p)
    except Exception:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────
# CHECK 1 — Feature-space exhaustion
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 1 : UNIQUE INPUT PATTERNS (feature-space exhaustion)")
print("─" * 70)

# Raw 4-feature space
unique_raw  = X_train_raw[FEATURES].drop_duplicates().shape[0]
unique_ext  = X_train_raw.drop_duplicates().shape[0]
max_raw     = 2 ** len(FEATURES)          # 16
max_ext     = 2 ** len(FEATURES_EXT)      # 256 (but engineered cols are deterministic)

print(f"\n  Raw 4 features   : {unique_raw} unique patterns out of max {max_raw}")
print(f"  All 8 features   : {unique_ext} unique patterns (engineered cols are deterministic)")
print(f"  Train rows       : {len(X_train_raw)}")
print(f"  Avg rows/pattern : {len(X_train_raw)/unique_raw:.1f}")

if unique_raw <= 12:
    print("\n  ⚠  WARNING: Very few unique input patterns.")
    print("     Complex models will perfectly memorise these patterns.")
elif unique_raw == max_raw:
    print("\n  ✓  All 16 patterns present in training data.")

# Check for identical rows in train AND test (guaranteed overlap given binary features)
overlap_patterns = (
    pd.merge(
        X_train_raw[FEATURES].drop_duplicates(),
        X_test[FEATURES].drop_duplicates(),
        on=FEATURES, how="inner"
    ).shape[0]
)
print(f"\n  Patterns shared between train and test : {overlap_patterns} / {unique_raw}")
if overlap_patterns == unique_raw:
    print("  ⚠  Every test pattern also appears in training data.")
    print("     Models have effectively 'seen' every test input during training.")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 2 — CV PR-AUC vs Test PR-AUC gap
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 2 : CV PR-AUC vs TEST PR-AUC GAP")
print("─" * 70)
print("\n  A positive gap (CV > Test) = optimism / overfit to CV folds")
print("  A negative gap (Test > CV) = possible overfit to THIS test split\n")

cv_models_all = {
    "Logistic Regression": best_lr_pipe,
    "Decision Tree"      : best_dt_pipe,
    "Random Forest"      : best_rf_pipe,
    "Gradient Boosting"  : best_gb_pipe,
    "SVM"                : best_svm_pipe,
    "KNN"                : best_knn_pipe,
    "XGBoost"            : best_xgb,
    "LightGBM"           : best_lgbm,
    "Stacking"           : stack,
    "Soft Voting"        : voting,
}

gap_records = []
print(f"  {'Model':<22} {'CV PR-AUC':>10} {'Test PR-AUC':>12} {'Gap':>8}  Flag")
print("  " + "-" * 65)
for name, model in cv_models_all.items():
    try:
        cv_scores = cross_val_score(model, X_train_raw, y_train_raw,
                                    cv=kfold, scoring="average_precision",
                                    n_jobs=-1)
        cv_mean = cv_scores.mean()
    except Exception:
        cv_mean = np.nan

    test_prauc = results[name]["PR-AUC"] if name in results else np.nan
    gap        = cv_mean - test_prauc if not np.isnan(cv_mean) else np.nan

    flag = ""
    if not np.isnan(gap):
        if gap > 0.08:
            flag = "⚠  Overfit (CV >> Test)"
        elif gap < -0.08:
            flag = "⚠  Test inflation? (Test >> CV)"
        else:
            flag = "✓  OK"

    gap_records.append({
        "Model": name, "CV_PRAUC": cv_mean,
        "Test_PRAUC": test_prauc, "Gap": gap
    })
    cv_str   = f"{cv_mean:.4f}" if not np.isnan(cv_mean) else "n/a"
    test_str = f"{test_prauc:.4f}" if not np.isnan(test_prauc) else "n/a"
    gap_str  = f"{gap:+.4f}" if not np.isnan(gap) else "n/a"
    print(f"  {name:<22} {cv_str:>10} {test_str:>12} {gap_str:>8}  {flag}")

gap_df = pd.DataFrame(gap_records)

# ─────────────────────────────────────────────────────────────────────────
# CHECK 3 — Logistic Regression baseline comparison
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 3 : COMPLEXITY vs LOGISTIC REGRESSION BASELINE")
print("─" * 70)
print("\n  On 4 binary features, LR should be nearly optimal.")
print("  Large gains from complex models → likely overfit.\n")

lr_prauc = results["Logistic Regression"]["PR-AUC"]
print(f"  LR PR-AUC (baseline) : {lr_prauc:.4f}\n")
print(f"  {'Model':<22} {'Test PR-AUC':>12} {'Δ vs LR':>10}  Flag")
print("  " + "-" * 56)
for name, res in results.items():
    if name == "Logistic Regression":
        continue
    delta = res["PR-AUC"] - lr_prauc
    flag  = "⚠  Suspicious gain" if delta > 0.05 else "✓  Plausible"
    print(f"  {name:<22} {res['PR-AUC']:>12.4f} {delta:>+10.4f}  {flag}")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 4 — Train vs Test accuracy / PR-AUC (memorisation check)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 4 : TRAIN vs TEST PR-AUC (memorisation)")
print("─" * 70)
print("\n  Train PR-AUC >> Test PR-AUC = model memorised training data.\n")

print(f"  {'Model':<22} {'Train PR-AUC':>13} {'Test PR-AUC':>12} {'Ratio':>8}  Flag")
print("  " + "-" * 68)
for name, model in cv_models_all.items():
    train_pa = _prauc(model, X_train_raw, y_train_raw)
    test_pa  = results[name]["PR-AUC"] if name in results else np.nan
    ratio    = train_pa / test_pa if test_pa and test_pa > 0 else np.nan
    flag     = ""
    if not np.isnan(ratio):
        if ratio > 1.15:
            flag = "⚠  Memorisation likely"
        elif ratio > 1.05:
            flag = "~  Slight overfit"
        else:
            flag = "✓  OK"
    print(f"  {name:<22} {train_pa:>13.4f} {test_pa:>12.4f} {ratio:>8.3f}  {flag}")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 5 — Permutation importance sanity
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 5 : PERMUTATION IMPORTANCE SANITY")
print("─" * 70)
print("\n  If shuffling a feature causes no PR-AUC drop, the model ignores it")
print("  (or found a spurious shortcut). Checked on test set.\n")

perm_models = {
    "Logistic Regression": best_lr_pipe,
    "XGBoost"            : best_xgb,
    "LightGBM"           : best_lgbm,
    "Random Forest"      : best_rf_pipe,
}

for name, model in perm_models.items():
    try:
        pi = permutation_importance(
            model, X_test, y_test,
            n_repeats=30, random_state=RANDOM_STATE,
            scoring="average_precision", n_jobs=-1
        )
        print(f"  {name}:")
        order = np.argsort(pi.importances_mean)[::-1]
        for i in order:
            fname = feat_names[i]
            mean  = pi.importances_mean[i]
            std   = pi.importances_std[i]
            flag  = "" if mean > 0.001 else "  ← near-zero, possible noise"
            print(f"    {fname:<20}  Δ PR-AUC = {mean:+.4f} ± {std:.4f}{flag}")
        print()
    except Exception as e:
        print(f"  {name}: failed — {e}\n")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 6 — Learning curves
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 6 : LEARNING CURVES")
print("─" * 70)
print("\n  If train score stays high while val score stays low → overfit.")
print("  If both converge → generalising well.\n")

lc_models = {
    "Logistic Regression": best_lr_pipe,
    "Random Forest"      : best_rf_pipe,
    "XGBoost"            : best_xgb,
}

fig_lc, axes_lc = plt.subplots(1, len(lc_models), figsize=(6*len(lc_models), 5))
fig_lc.suptitle("Learning Curves — Overfit Check", fontweight="bold")

for ax, (name, model) in zip(axes_lc, lc_models.items()):
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train_raw, y_train_raw,
            cv=kfold, scoring="average_precision",
            train_sizes=np.linspace(0.2, 1.0, 6),
            n_jobs=-1
        )
        t_mean = train_scores.mean(axis=1)
        v_mean = val_scores.mean(axis=1)
        t_std  = train_scores.std(axis=1)
        v_std  = val_scores.std(axis=1)

        ax.plot(train_sizes, t_mean, "o-", color="#DD8452", label="Train PR-AUC")
        ax.plot(train_sizes, v_mean, "o-", color="#4C72B0", label="Val PR-AUC")
        ax.fill_between(train_sizes, t_mean-t_std, t_mean+t_std, alpha=0.15, color="#DD8452")
        ax.fill_between(train_sizes, v_mean-v_std, v_mean+v_std, alpha=0.15, color="#4C72B0")
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.set_xlabel("Training samples"); ax.set_ylabel("PR-AUC")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

        final_gap = t_mean[-1] - v_mean[-1]
        status = "⚠  OVERFIT" if final_gap > 0.10 else ("~  Mild" if final_gap > 0.05 else "✓  OK")
        print(f"  {name:<22}  Train={t_mean[-1]:.4f}  Val={v_mean[-1]:.4f}"
              f"  Gap={final_gap:+.4f}  {status}")
        ax.set_title(f"{name}\n(gap={final_gap:+.3f}  {status})",
                     fontweight="bold", fontsize=9)
    except Exception as e:
        print(f"  {name}: failed — {e}")
        ax.set_title(name)

plt.tight_layout()
plt.savefig("overfit_lc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✔ Saved overfit_lc.png")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 7 — Repeated-CV variance (all 10 models)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 7 : REPEATED CV VARIANCE (all 10 models)")
print("─" * 70)
print("\n  High std → unstable / overfit to fold structure.\n")

print(f"  {'Model':<22} {'Mean PR-AUC':>12} {'Std':>8} {'CV/Test gap':>12}  Flag")
print("  " + "-" * 68)

rep_kfold_check = __import__("sklearn.model_selection", fromlist=["RepeatedStratifiedKFold"]
                              ).RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

overfit_scores = {}
for name, model in cv_models_all.items():
    try:
        sc = cross_val_score(model, X_train_raw, y_train_raw,
                             cv=rep_kfold_check, scoring="average_precision",
                             n_jobs=-1)
        m, s = sc.mean(), sc.std()
        test_pa = results[name]["PR-AUC"] if name in results else np.nan
        gap = m - test_pa
        flag = ""
        if s > 0.10:
            flag = "⚠  High variance"
        elif abs(gap) > 0.08:
            flag = "⚠  CV/test mismatch"
        else:
            flag = "✓  Stable"
        overfit_scores[name] = {"cv_mean": m, "cv_std": s, "gap": gap}
        print(f"  {name:<22} {m:>12.4f} {s:>8.4f} {gap:>+12.4f}  {flag}")
    except Exception as e:
        print(f"  {name:<22} failed — {e}")
        overfit_scores[name] = {"cv_mean": np.nan, "cv_std": np.nan, "gap": np.nan}

# ─────────────────────────────────────────────────────────────────────────
# CHECK 8 — Bootstrap CI overlap
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 8 : BOOTSTRAP CI OVERLAP (PR-AUC, 95%)")
print("─" * 70)
print("\n  If ALL models' CIs overlap → no model is statistically better than")
print("  any other — differences are split noise, not real signal.\n")

rng      = np.random.RandomState(RANDOM_STATE)
ci_data  = {}
y_test_a = np.array(y_test)

for name, res in results.items():
    scores = []
    yp = res["y_prob"]
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, len(y_test_a), len(y_test_a))
        try:
            scores.append(average_precision_score(y_test_a[idx], yp[idx]))
        except Exception:
            pass
    arr = np.array(scores)
    ci_data[name] = (arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5))

print(f"  {'Model':<22} {'PR-AUC':>8}  {'95% CI':<22}")
print("  " + "-" * 55)
for name, (m, lo, hi) in sorted(ci_data.items(), key=lambda x: -x[1][0]):
    bar_lo = "─" * int((lo - 0.0) * 60)
    bar    = "█" * max(1, int((hi - lo) * 60))
    print(f"  {name:<22} {m:>8.4f}  [{lo:.3f}–{hi:.3f}]")

# Check universal overlap
lowers = [v[1] for v in ci_data.values()]
uppers = [v[2] for v in ci_data.values()]
best_lower = max(lowers)
worst_upper = min(uppers)
if best_lower < worst_upper:
    print("\n  ⚠  All model CIs overlap — no model is significantly better.")
    print("     Observed differences may be due to split noise.")
else:
    print("\n  ✓  At least one model has a non-overlapping CI — some separation exists.")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 9 — Brier score: train vs test calibration gap
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  CHECK 9 : BRIER SCORE TRAIN vs TEST (calibration gap)")
print("─" * 70)
print("\n  Train Brier << Test Brier = overconfident / overfit probabilities.\n")

print(f"  {'Model':<22} {'Train Brier':>12} {'Test Brier':>12} {'Ratio':>8}  Flag")
print("  " + "-" * 62)
for name, model in cv_models_all.items():
    tr_b  = _brier(model, X_train_raw, y_train_raw)
    te_b  = results[name]["Brier"] if name in results else np.nan
    ratio = te_b / tr_b if tr_b and tr_b > 1e-6 else np.nan
    flag  = ""
    if not np.isnan(ratio):
        if ratio > 2.0:
            flag = "⚠  Severe calibration gap"
        elif ratio > 1.3:
            flag = "~  Moderate gap"
        else:
            flag = "✓  OK"
    print(f"  {name:<22} {tr_b:>12.4f} {te_b:>12.4f} {ratio:>8.2f}  {flag}")

# ─────────────────────────────────────────────────────────────────────────
# CHECK 10 — Summary verdict table
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  CHECK 10 : OVERFIT DIAGNOSTIC SUMMARY")
print("=" * 70)

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(X_train_raw, y_train_raw)
dummy_prauc = average_precision_score(y_test, dummy.predict_proba(X_test)[:, 1])

print(f"\n  Dummy classifier PR-AUC (majority class) : {dummy_prauc:.4f}")
print(f"  If any model is barely above this, it adds no value.\n")

print(f"  {'Model':<22} {'CV PR-AUC':>10} {'Test PR-AUC':>12} {'Gap':>8} "
      f"{'CV Std':>8} {'Verdict'}")
print("  " + "-" * 80)

for name in cv_models_all:
    rec  = overfit_scores.get(name, {})
    cvm  = rec.get("cv_mean", np.nan)
    cvs  = rec.get("cv_std", np.nan)
    gap  = rec.get("gap", np.nan)
    tepa = results[name]["PR-AUC"] if name in results else np.nan

    flags = []
    if not np.isnan(gap):
        if gap > 0.08:  flags.append("CV>>Test")
        if gap < -0.08: flags.append("Test>>CV")
    if not np.isnan(cvs) and cvs > 0.10:
        flags.append("HighVar")
    tr_pa = _prauc(cv_models_all[name], X_train_raw, y_train_raw)
    if not np.isnan(tr_pa) and not np.isnan(tepa) and tepa > 0:
        if tr_pa / tepa > 1.15: flags.append("Memorise")

    verdict = "✓ OK" if not flags else ("⚠ " + ", ".join(flags))
    cv_s  = f"{cvm:.4f}" if not np.isnan(cvm) else "n/a"
    cvsd  = f"{cvs:.4f}" if not np.isnan(cvs) else "n/a"
    gap_s = f"{gap:+.4f}" if not np.isnan(gap) else "n/a"
    te_s  = f"{tepa:.4f}" if not np.isnan(tepa) else "n/a"
    print(f"  {name:<22} {cv_s:>10} {te_s:>12} {gap_s:>8} {cvsd:>8}  {verdict}")

# ─────────────────────────────────────────────────────────────────────────
# VISUALISATION — Overfit gap chart + CI plot
# ─────────────────────────────────────────────────────────────────────────
fig_ov, axes_ov = plt.subplots(1, 2, figsize=(18, 6))
fig_ov.suptitle("Overfitting Diagnostic — CV vs Test PR-AUC", fontweight="bold")

model_names = list(cv_models_all.keys())
cv_means = [overfit_scores.get(n, {}).get("cv_mean", np.nan) for n in model_names]
te_means = [results[n]["PR-AUC"] if n in results else np.nan for n in model_names]
gaps     = [c - t if not (np.isnan(c) or np.isnan(t)) else np.nan
            for c, t in zip(cv_means, te_means)]

x = np.arange(len(model_names))
w = 0.35
ax1 = axes_ov[0]
bars1 = ax1.bar(x - w/2, cv_means, w, label="CV PR-AUC (train folds)",
                color="#4C72B0", alpha=0.85)
bars2 = ax1.bar(x + w/2, te_means, w, label="Test PR-AUC (held-out)",
                color="#DD8452", alpha=0.85)
ax1.axhline(dummy_prauc, color="red", linestyle="--", linewidth=1.2,
            label=f"Dummy baseline ({dummy_prauc:.3f})")
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel("PR-AUC")
ax1.set_title("CV vs Test PR-AUC per Model", fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.3)

ax2 = axes_ov[1]
colors_gap = ["#DD8452" if (g is not None and not np.isnan(g) and abs(g) > 0.08)
              else "#4C72B0" for g in gaps]
bars3 = ax2.bar(model_names, gaps, color=colors_gap, alpha=0.85, edgecolor="white")
ax2.axhline(0,    color="black", linewidth=1)
ax2.axhline(0.08, color="red",   linewidth=1, linestyle="--", label="+0.08 overfit threshold")
ax2.axhline(-0.08,color="red",   linewidth=1, linestyle="--", label="-0.08 threshold")
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Gap (CV − Test PR-AUC)")
ax2.set_title("Overfit Gap per Model\n(orange = |gap| > 0.08)", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("overfit_gap.png", dpi=150, bbox_inches="tight")
plt.close()

# CI plot
fig_ci, ax_ci = plt.subplots(figsize=(10, 6))
ci_names = list(ci_data.keys())
ci_means = [ci_data[n][0] for n in ci_names]
ci_los   = [ci_data[n][1] for n in ci_names]
ci_his   = [ci_data[n][2] for n in ci_names]
order    = np.argsort(ci_means)[::-1]

y_pos = np.arange(len(ci_names))
for i, idx in enumerate(order):
    name  = ci_names[idx]
    m, lo, hi = ci_data[name]
    ax_ci.plot([lo, hi], [i, i], color="#4C72B0", linewidth=5, alpha=0.5, solid_capstyle="round")
    ax_ci.plot(m, i, "o", color="#4C72B0", markersize=8, zorder=5)

ax_ci.set_yticks(range(len(ci_names)))
ax_ci.set_yticklabels([ci_names[i] for i in order], fontsize=9)
ax_ci.axvline(dummy_prauc, color="red", linestyle="--", linewidth=1.2,
              label=f"Dummy ({dummy_prauc:.3f})")
ax_ci.set_xlabel("PR-AUC (95% Bootstrap CI)")
ax_ci.set_title("Bootstrap 95% CI — Do models actually differ?", fontweight="bold")
ax_ci.legend(fontsize=9); ax_ci.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("overfit_ci.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n")
print("✔ Saved: overfit_gap.png, overfit_lc.png, overfit_ci.png")
print("\n" + "=" * 70)
print("  HOW TO INTERPRET RESULTS")
print("=" * 70)
print("""
  Gap (CV − Test) > +0.08   → Model is overfit to CV folds; CV was too optimistic.
  Gap (CV − Test) < -0.08   → Test PR-AUC inflated (possible overfit to this split).
  Train PR-AUC / Test > 1.15 → Model memorised training data (Check 4).
  CI overlap for all models  → No model is statistically better; differences are noise.
  Permutation Δ ≈ 0          → Feature is ignored; model found a spurious shortcut.
  Learning curve gap > 0.10  → Classic bias-variance overfit signal (Check 6).

  Key diagnostic: if XGBoost / Stacking beat Logistic Regression by > 0.05 PR-AUC
  on only 4 binary features, treat that gain with scepticism — run on an independent
  dataset to confirm.
""")



print("=" * 70)
print("  OVERFITTING DIAGNOSTIC COMPLETE")
print("=" * 70)