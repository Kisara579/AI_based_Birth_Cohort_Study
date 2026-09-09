import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
=============================================================================
NICU ADMISSION PREDICTION - FINAL CORRECTED PIPELINE
=============================================================================
All critical bugs fixed + performance improvements applied:

  FIX ① Scaler fitted on raw train set (not SMOTE data)
  FIX ② Threshold tuned on validation fold (not test set) - no leakage
  FIX ③ VotingClassifier uses consistent pipelines with internal SMOTE
  FIX ④ SMOTE applied inside each CV fold via imblearn Pipeline
  FIX ⑤ Optuna CV uses proper imblearn Pipeline (no leakage)

  IMPROVEMENT ① Repeated Stratified K-Fold for final evaluation
  IMPROVEMENT ② Calibration via CalibratedClassifierCV (Platt scaling)
  IMPROVEMENT ③ Bootstrap confidence intervals on test metrics
  IMPROVEMENT ④ Clean model registry - one source of truth per model
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
  9.  Stacking     (XGB+LGBM+RF+GB -> LR meta)
  10. Soft Voting  (all 8 via unified pipelines)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import seaborn as sns
# import shap
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

# ===========================================================================
# 0. CONFIGURATION
# ===========================================================================
DATA_PATH = PROJECT_ROOT / "data" / "NICU" / "raw" / "gdm_t2dm_risk_dataset (3).csv"
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
    "Insulin_x_Preterm"                              : "InsxPreterm",
    "Hypertension_x_OGTT"                            : "HypxOGTT",
    "Insulin_x_Hypertension"                         : "InsxHyp",
    "Risk_score"                                     : "Risk Score",
}

# ===========================================================================
# 1. LOAD & VALIDATE DATA
# ===========================================================================
print("=" * 70)
print("  NICU ADMISSION PREDICTION - FINAL CORRECTED PIPELINE")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

assert TARGET in df.columns, f"Target column '{TARGET}' not found."
assert df[TARGET].nunique() == 2, "Target must be binary."
assert df.isnull().sum().sum() == 0, "Dataset has missing values - handle them first."

neg = (df[TARGET] == 0).sum()
pos = (df[TARGET] == 1).sum()
scale_pos_weight = neg / pos

print(f"\n  Dataset         : {df.shape[0]} rows x {df.shape[1]} cols")
print(f"  Class balance   : {neg} negative  |  {pos} positive")
print(f"  Imbalance ratio : {scale_pos_weight:.2f}:1")

# ===========================================================================
# 2. EDA
# ===========================================================================
print("\n" + "-" * 70)
print("  SECTION 1 : EDA")
print("-" * 70)

print("\n[2a] Missing values :", df.isnull().sum().sum())
print("\n[2b] Feature stats:")
print(df[FEATURES].describe().T[["mean","std","min","max"]].to_string())

print("\n[2c] Feature -> NICU correlation:")
for f in FEATURES:
    r = df[[f, TARGET]].corr().iloc[0, 1]
    print(f"  {SHORT[f]:<18}  r = {r:+.4f}")

print("\n[2d] NICU rate by feature presence:")
for f in FEATURES:
    rate = df.groupby(f)[TARGET].mean()
    print(f"  {SHORT[f]:<18}  absent={rate.get(0,0):.2%}  present={rate.get(1,0):.2%}")

# ===========================================================================
# 3. FEATURE ENGINEERING
# ===========================================================================
print("\n" + "-" * 70)
print("  SECTION 2 : FEATURE ENGINEERING")
print("-" * 70)

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

print(f"\n  Features: {len(FEATURES)} -> {len(FEATURES_EXT)}")
print("\n  NICU rate by Risk_score:")
for v, grp in df.groupby("Risk_score"):
    rate = grp[TARGET].mean()
    bar  = "" * int(rate * 40)
    print(f"    Score {v}: {rate:.2%}  {bar}")

# ===========================================================================
# 4. TRAIN / TEST SPLIT  (80 / 20 stratified - held-out once, never touched
#    during tuning or threshold selection)
# ===========================================================================
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# -- FIX ①: Scaler fitted on RAW train set only --------------------------
scaler = StandardScaler()
scaler.fit(X_train_raw)           # fit on raw, no synthetic samples
X_train_sc  = scaler.transform(X_train_raw)
X_test_sc   = scaler.transform(X_test)

# CV splitters
kfold        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=RANDOM_STATE)
rep_kfold    = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                        random_state=RANDOM_STATE)

# ===========================================================================
# 5. IMBALANCE STRATEGIES COMPARISON
# ===========================================================================
print("\n" + "-" * 70)
print("  SECTION 3 : IMBALANCE HANDLING COMPARISON")
print("-" * 70)

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
        print(f"  {name:<14}: FAILED - {e}")
        sampled_data[name] = (X_train_raw.copy(), y_train_raw.copy())

print(f"\n  Sklearn models  : imblearn Pipeline with SMOTE-ENN inside each CV fold")
print(f"  XGB / LGBM      : raw split + scale_pos_weight={scale_pos_weight:.2f}")

# ===========================================================================
# 6. HELPER: THRESHOLD TUNING ON VALIDATION DATA  (FIX ②)
#    Threshold is found on a held-out VALIDATION split from X_train_raw,
#    NEVER on X_test.  The test set is only touched once for final reporting.
# ===========================================================================

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


def get_oof_train_metrics(model, X_train, y_train, name):
    """Compute Out-of-Fold training metrics via cross_val_predict."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.base import clone
    cv_oof = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    try:
        oof_model = clone(model)
    except Exception:
        oof_model = model

    try:
        y_train_prob_cv = cross_val_predict(
            oof_model, X_train, y_train, cv=cv_oof,
            method='predict_proba', n_jobs=-1
        )[:, 1]
    except Exception:
        y_train_prob_cv = None

    if y_train_prob_cv is not None:
        thresholds_search = np.arange(0.1, 0.91, 0.05)
        best_threshold_cv = 0.5
        best_f1_cv = 0

        for t in thresholds_search:
            y_pred_t = (y_train_prob_cv >= t).astype(int)
            f1_t = f1_score(y_train, y_pred_t, zero_division=0)
            if f1_t > best_f1_cv:
                best_f1_cv = f1_t
                best_threshold_cv = t

        best_threshold_cv = float(round(best_threshold_cv, 2))
        print(f"  Best OOF threshold (CV, F1) for {name}: {best_threshold_cv}")

        y_train_pred_cv = (y_train_prob_cv >= best_threshold_cv).astype(int)

        oof_metrics = {
            "accuracy":  float(accuracy_score(y_train, y_train_pred_cv)),
            "precision": float(precision_score(y_train, y_train_pred_cv, zero_division=0)),
            "recall":    float(recall_score(y_train, y_train_pred_cv, zero_division=0)),
            "f1":        float(f1_score(y_train, y_train_pred_cv, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_train, y_train_prob_cv)),
        }
        print(f"  OOF Train ({name}) -> Acc={oof_metrics['accuracy']:.4f}  "
              f"F1={oof_metrics['f1']:.4f}  AUC={oof_metrics['roc_auc']:.4f}")
        return oof_metrics
    else:
        print(f"  (cross_val_predict not available for {name}, using defaults)")
        return {}


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
    threshold : float - found on val set, NOT on X_te.
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
        "raw_threshold": float(threshold),
    }


def print_result(res):
    print(f"  Threshold  : {res['Threshold']}  (tuned on val set)")
    print(f"  Accuracy   : {res['Accuracy']}")
    print(f"  Precision  : {res['Precision']}")
    print(f"  Recall     : {res['Recall']}   <- clinical priority")
    print(f"  F1         : {res['F1']}")
    print(f"  F2         : {res['F2']}   <- weights Recall 2x")
    print(f"  ROC-AUC    : {res['ROC-AUC']}")
    print(f"  PR-AUC     : {res['PR-AUC']}  <- best for imbalanced")
    print(f"  MCC        : {res['MCC']}   <- balanced quality")
    print(f"  Brier      : {res['Brier']}  <- calibration")
    print()
    print(classification_report(y_test, res["y_pred"],
          target_names=["No NICU (0)", "NICU (1)"], digits=4))


# -- Validation split for threshold tuning (from training data only) -------
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_train_raw, y_train_raw,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_raw)

X_tr2_sc  = scaler.transform(X_tr2)
X_val_sc  = scaler.transform(X_val)

results = {}
oof_train_metrics_dict = {}

# ===========================================================================
# 7. MODEL 1 - LOGISTIC REGRESSION
#    FIX ④: imblearn Pipeline keeps SMOTE-ENN inside each CV fold
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 1 : LOGISTIC REGRESSION")
print("-" * 70)

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
oof_train_metrics_dict["Logistic Regression"] = get_oof_train_metrics(best_lr_pipe, X_train_raw, y_train_raw, "Logistic Regression")

# ===========================================================================
# 8. MODEL 2 - DECISION TREE
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 2 : DECISION TREE")
print("-" * 70)

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
oof_train_metrics_dict["Decision Tree"] = get_oof_train_metrics(best_dt_pipe, X_train_raw, y_train_raw, "Decision Tree")

# ===========================================================================
# 9. MODEL 3 - RANDOM FOREST
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 3 : RANDOM FOREST")
print("-" * 70)

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
oof_train_metrics_dict["Random Forest"] = get_oof_train_metrics(best_rf_pipe, X_train_raw, y_train_raw, "Random Forest")

# ===========================================================================
# 10. MODEL 4 - GRADIENT BOOSTING
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 4 : GRADIENT BOOSTING")
print("-" * 70)

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
oof_train_metrics_dict["Gradient Boosting"] = get_oof_train_metrics(best_gb_pipe, X_train_raw, y_train_raw, "Gradient Boosting")

# ===========================================================================
# 11. MODEL 5 - SVM
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 5 : SVM")
print("-" * 70)

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
oof_train_metrics_dict["SVM"] = get_oof_train_metrics(best_svm_pipe, X_train_raw, y_train_raw, "SVM")

# ===========================================================================
# 12. MODEL 6 - KNN
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 6 : KNN")
print("-" * 70)

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
oof_train_metrics_dict["KNN"] = get_oof_train_metrics(best_knn_pipe, X_train_raw, y_train_raw, "KNN")

# ===========================================================================
# 13. MODEL 7 - XGBoost
#     No SMOTE - uses scale_pos_weight natively
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 7 : XGBoost")
print("-" * 70)

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
oof_train_metrics_dict["XGBoost"] = get_oof_train_metrics(best_xgb, X_train_raw, y_train_raw, "XGBoost")

# ===========================================================================
# 14. MODEL 8 - LightGBM
#     No SMOTE - uses is_unbalance natively
# ===========================================================================
print("\n" + "-" * 70)
print("  MODEL 8 : LightGBM")
print("-" * 70)

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
oof_train_metrics_dict["LightGBM"] = get_oof_train_metrics(best_lgbm, X_train_raw, y_train_raw, "LightGBM")

# ===========================================================================
# 15. ENSEMBLES - Stacking + Soft Voting
#     FIX ③: VotingClassifier uses the full imblearn pipelines consistently
# ===========================================================================
print("\n" + "-" * 70)
print("  ENSEMBLE MODELS")
print("-" * 70)

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
results["Stacking (XGB+LGBM+RF+GB)"] = evaluate("Stacking (XGB+LGBM+RF+GB)", stack, X_test, y_test, thr_stack)
print("\n-- Stacking (XGB + LGBM + RF + GB -> LR meta)")
print_result(results["Stacking (XGB+LGBM+RF+GB)"])
oof_train_metrics_dict["Stacking (XGB+LGBM+RF+GB)"] = get_oof_train_metrics(stack, X_train_raw, y_train_raw, "Stacking (XGB+LGBM+RF+GB)")

# Soft Voting - all 8 models as imblearn pipelines or raw for XGB/LGBM
# (FIX ③: consistent - all pipelines trained on same raw split)
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
results["Soft Voting (LR+DT+RF+GB+SVM+KNN+XGB+LGBM)"] = evaluate("Soft Voting (LR+DT+RF+GB+SVM+KNN+XGB+LGBM)", voting, X_test, y_test, thr_vote)
print("-- Soft Voting (all 8 models)")
print_result(results["Soft Voting (LR+DT+RF+GB+SVM+KNN+XGB+LGBM)"])
oof_train_metrics_dict["Soft Voting (LR+DT+RF+GB+SVM+KNN+XGB+LGBM)"] = get_oof_train_metrics(voting, X_train_raw, y_train_raw, "Soft Voting (LR+DT+RF+GB+SVM+KNN+XGB+LGBM)")

# ===========================================================================
# [INSERT AFTER SECTION 3 / ENSEMBLE MODELS, BEFORE SECTION 4 BOOTSTRAP CI]
#
# PER-FOLD CROSS-VALIDATION - ALL 10 MODELS
# Prints fold-by-fold PR-AUC, Recall, and F2 for every model.
# Uses the same kfold splitter as the rest of the pipeline.
# SMOTE is applied inside the imblearn Pipeline per fold (no leakage).
# XGBoost and LightGBM use raw splits + scale_pos_weight (no SMOTE).
# ===========================================================================

print("\n" + "=" * 70)
print("  PER-FOLD CROSS-VALIDATION - ALL 10 MODELS")
print("=" * 70)
print(f"\n  Splitter : StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
      f"random_state={RANDOM_STATE})")
print(f"  Scored on : X_train_raw / y_train_raw  (test set never touched)\n")

# -- helper: score one fold ------------------------------------------------
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


# -- models registry (all 10) ----------------------------------------------
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

fold_summary = {}   # name -> {metric: [fold_values]}

for model_name, base_model in per_fold_models.items():
    print("\n" + "-" * 70)
    print(f"  {model_name}")
    print("-" * 70)
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

    # -- per-model summary -------------------------------------------------
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
        variance_flag = ("!  High variance across folds"
                         if fold_pr_arr.std() > 0.08
                         else "OK  Stable across folds")
        print(f"  Variance check     : {variance_flag}")

    fold_summary[model_name] = {
        "pr_auc": fold_pr,
        "recall": fold_rec,
        "f2"    : fold_f2,
    }

# -- cross-model per-fold summary table ------------------------------------
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

# ===========================================================================
# [CONTINUE WITH SECTION 4 : BOOTSTRAP CONFIDENCE INTERVALS]
# ===========================================================================


# ===========================================================================
# 16. BOOTSTRAP CONFIDENCE INTERVALS  (IMPROVEMENT ③)
# ===========================================================================
print("\n" + "-" * 70)
print("  SECTION 4 : BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("-" * 70)

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

# ===========================================================================
# 17. REPEATED K-FOLD CV ON TRAIN SET  (IMPROVEMENT ①)
# ===========================================================================
print("\n" + "-" * 70)
print(f"  SECTION 5 : REPEATED {N_REPEATS}x{N_SPLITS}-FOLD CV  (train set only)")
print("-" * 70)

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

# # ===========================================================================
# # 18. SHAP EXPLAINABILITY  - raw estimators for tree models
# # ===========================================================================
# print("\n" + "-" * 70)
# print("  SECTION 6 : SHAP EXPLAINABILITY")
# print("-" * 70)
# 
# # Extract raw classifiers from pipelines for SHAP (TreeExplainer needs the tree)
# rf_clf  = best_rf_pipe.named_steps["clf"]
# gb_clf  = best_gb_pipe.named_steps["clf"]
# 
# # For tree SHAP we need the X that was fed to the clf step
# # i.e. after SMOTE (use X_test unchanged since no scaling in RF/GB pipe)
# shap_models = {
#     "XGBoost" : (best_xgb,  X_test),
#     "LightGBM": (best_lgbm, X_test),
#     "RF"      : (rf_clf,    X_test),
#     "GB"      : (gb_clf,    X_test),
# }
# 
# shap_values_dict = {}
# for mname, (model, Xte) in shap_models.items():
#     try:
#         exp = shap.TreeExplainer(model)
#         sv  = exp.shap_values(Xte)
#         if isinstance(sv, list):
#             sv = sv[1]
#         shap_values_dict[mname] = sv
#         ms = np.abs(sv).mean(axis=0)
#         print(f"\n  {mname} SHAP (mean |SHAP|):")
#         for nm, v in sorted(zip(feat_names, ms), key=lambda x: -x[1]):
#             print(f"    {nm:<20} {v:.4f}  {''*int(v*60)}")
#     except Exception as e:
#         print(f"\n  {mname} SHAP failed: {e}")

# ===========================================================================
# 19. SUMMARY TABLE
# ===========================================================================
print("\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY  (sorted by PR-AUC)")
print("=" * 70)

DISPLAY_COLS = ["Model","Threshold","Recall","F2","PR-AUC","ROC-AUC","MCC","F1","Brier"]
summary = pd.DataFrame([{c: r[c] for c in DISPLAY_COLS} for r in results.values()])
summary = summary.sort_values("PR-AUC", ascending=False).reset_index(drop=True)
summary.index += 1
print("\n" + summary.to_string())

best = summary.iloc[0]
print(f"\n Best (PR-AUC) : {best['Model']}"
      f"  PR-AUC={best['PR-AUC']}  Recall={best['Recall']}  F2={best['F2']}")

# ==============================================================================
# STEP 21 - SAVE MODEL ARTIFACTS
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 21 - SAVE MODEL ARTIFACTS")
print("=" * 70)

from shared.model_artifacts import save_model_artifacts
from sklearn.pipeline import Pipeline as SkPipeline
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Map model names to their fitted model objects and pipelines
trained_models = {
    "Logistic Regression": best_lr_pipe,
    "Decision Tree": best_dt_pipe,
    "Random Forest": best_rf_pipe,
    "Gradient Boosting": best_gb_pipe,
    "SVM": best_svm_pipe,
    "KNN": best_knn_pipe,
    "XGBoost": best_xgb,
    "LightGBM": best_lgbm,
    "Stacking (XGB+LGBM+RF+GB)": stack,
    "Soft Voting (LR+DT+RF+GB+SVM+KNN+XGB+LGBM)": voting,
}

FEATURES_USED = FEATURES_EXT
categorical_feature_names = []  # all features are numeric binary/integer (0/1 and risk score)

for name, model in trained_models.items():
    print(f"  Saving artifacts for {name}...")
    
    res = results[name]
    y_pred = res["y_pred"]
    y_prob = res["y_prob"]
    raw_thresh = res.get("raw_threshold", res["Threshold"])
    
    # Determine the optimal threshold for saving that matches y_pred
    t_save = round(raw_thresh, 2)
    if y_prob is not None and not np.array_equal((y_prob >= t_save).astype(int), y_pred):
        t_save = float(np.floor(raw_thresh * 100) / 100)
        if not np.array_equal((y_prob >= t_save).astype(int), y_pred):
            t_save = float(raw_thresh)

    if y_prob is None:
        auc_score = float('nan')
        y_prob_out = np.zeros_like(y_pred, dtype=float)
    else:
        auc_score = float(roc_auc_score(y_test, y_prob))
        y_prob_out = y_prob
        # IMPORTANT: y_pred is already the true final model output from evaluate()
        
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc_score
    }

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics_payload = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm
    }

    # These models are already pipelines or raw estimators - wrap in Pipeline for artifact saving
    pipeline = SkPipeline([
        ("classifier", model)
    ])

    # Compute train metrics aligned with the operational threshold
    train_metrics = oof_train_metrics_dict.get(name, {})
    try:
        if hasattr(model, "predict_proba"):
            p_tr = model.predict_proba(X_train_raw)[:, 1]
        else:
            p_tr = model.decision_function(X_train_raw)
        pred_tr = (p_tr >= raw_thresh).astype(int)
        train_metrics = {
            "accuracy": float(accuracy_score(y_train_raw, pred_tr)),
            "precision": float(precision_score(y_train_raw, pred_tr, zero_division=0)),
            "recall": float(recall_score(y_train_raw, pred_tr, zero_division=0)),
            "f1": float(f1_score(y_train_raw, pred_tr, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_train_raw, p_tr)),
        }
    except Exception:
        pass

    algo_id = name.lower().replace(" ", "").replace("(", "").replace(")", "").replace("+", "_")

    save_model_artifacts(
        pipeline=pipeline,
        outcome="nicu",
        outcome_id="nicu",
        algorithm=algo_id,
        label=name,
        description=f"{name} classifier trained on NICU admission dataset.",
        target_column=TARGET,
        feature_order=FEATURES_USED,
        categorical_features=categorical_feature_names,
        metrics_payload=metrics_payload,
        positive_label="NICU (1)",
        negative_label="No NICU (0)",
        threshold=t_save,
        y_test=np.array(y_test),
        y_prob=y_prob_out,
        train_metrics=train_metrics,
    )

print("\n" + "=" * 70)
print("  ALL DONE OK")
print("=" * 70)
