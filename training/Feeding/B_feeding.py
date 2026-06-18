import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
=============================================================================
BREASTFEEDING FEEDING PROBLEM PREDICTION - FULL CLINICAL ML PIPELINE
Adapted to run without imbalanced-learn, xgboost, or shap.
SMOTE is implemented from scratch using sklearn's NearestNeighbors.
All fixes from the original script are preserved.
=============================================================================
"""

# -- 0. Imports ----------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from sklearn.neighbors import NearestNeighbors  # used by manual SMOTE

from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                      RandomizedSearchCV)
from sklearn.preprocessing    import StandardScaler
from sklearn.base             import clone
from sklearn.inspection       import permutation_importance
from sklearn.calibration      import (calibration_curve,
                                      CalibratedClassifierCV)
from sklearn.compose          import ColumnTransformer

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

# -- Clinical settings ---------------------------------------------------------
CLINICAL_WEIGHTS = {0: 1, 1: 3}
RECALL_TARGET    = 0.90
N_FOLDS          = 5
N_BOOT           = 1000


# ==============================================================================
# HELPERS
# ==============================================================================

def sheet_name(base, suffix):
    """Build a valid Excel sheet name (≤31 chars, no illegal chars)."""
    illegal = r'\/:*?[]'
    for ch in illegal:
        base = base.replace(ch, '')
    max_base = 31 - len(suffix) - 1
    return f"{base[:max_base]} {suffix}"


# -- Manual SMOTE (replaces imbalanced-learn) ----------------------------------
def smote_resample(X, y, random_state=42, k_neighbors=5):
    """
    Synthetic Minority Over-sampling TEchnique.
    Generates synthetic samples for the minority class so both classes
    are equal in size.  Uses sklearn NearestNeighbors internally.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    majority_class  = classes[np.argmax(counts)]
    minority_class  = classes[np.argmin(counts)]
    X_min = X[y == minority_class]
    X_maj = X[y == majority_class]
    n_needed = len(X_maj) - len(X_min)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    nn.fit(X_min)
    _, indices = nn.kneighbors(X_min)          # shape (n_min, k+1)

    synthetic = []
    for _ in range(n_needed):
        idx  = rng.integers(0, len(X_min))
        nn_idx = rng.integers(1, k_neighbors + 1)  # skip self (col 0)
        neighbor = X_min[indices[idx, nn_idx]]
        alpha    = rng.random()
        synthetic.append(X_min[idx] + alpha * (neighbor - X_min[idx]))

    X_syn = np.array(synthetic)
    X_res = np.vstack([X, X_syn])
    y_res = np.concatenate([y, np.full(n_needed, minority_class)])
    # Shuffle
    perm  = rng.permutation(len(X_res))
    return X_res[perm], y_res[perm]


# -- K-Fold CV with SMOTE inside each fold -------------------------------------
def kfold_metrics(model, X, y, n_splits=5, apply_smote=True, scaled=False):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if apply_smote:
            X_tr, y_tr = smote_resample(X_tr, y_tr, random_state=42)

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
                     + ["-- Mean --", "-- Std  --"])
    return summary


def bootstrap_ci(y_true, y_prob, n_boot=N_BOOT, alpha=0.05, threshold=0.5):
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
    a_right = (pred_a == y_true); b_right = (pred_b == y_true)
    n01 = np.sum(~a_right &  b_right); n10 = np.sum(a_right & ~b_right)
    contingency = np.array(
        [[np.sum(a_right & b_right), n01],
         [n10, np.sum(~a_right & ~b_right)]])
    chi2, p, _, _ = chi2_contingency(contingency, correction=True)
    return chi2, p


def optimal_threshold(y_true, y_prob, target_recall=RECALL_TARGET):
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    valid = np.where(rec_arr[:-1] >= target_recall)[0]
    if len(valid) == 0:
        return 0.5, 0.0, 0.0
    best = valid[np.argmax(prec_arr[valid])]
    return float(thr_arr[best]), float(prec_arr[best]), float(rec_arr[best])


# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 1 - LOAD DATA")
print("=" * 70)

DATA_PATH = Path("data/Feeding/raw/Breastfeeding and mental health data - open access.csv")

df_raw = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df_raw.shape}")


# ==============================================================================
# STEP 2 - BUILD TARGET
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 2 - BUILD TARGET: has_feeding_problem")
print("=" * 70)

challenge_cols = [
    "breastfeeding_challenge_sore_nipples",
    "breastfeeding_challenge_latching",
    "breastfeeding_challenge_tongue_tie",
    "breastfeeding_challenge_mastitis",
    "breastfeeding_challenge_pain",
    "breastfeeding_challenge_engorgement",
    "breastfeeding_challenge_low_milk_supply",
    "breastfeeding_challenge_high_milk_supply",
    "breastfeeding_challenge_other",
]

df = df_raw.copy()
df["has_feeding_problem"] = df[challenge_cols].apply(
    lambda row: 1 if any(str(v).strip() == "1" for v in row) else 0, axis=1
)
vc = df["has_feeding_problem"].value_counts()
print(f"\n  Target distribution: {vc.to_dict()}")
print(f"  -> {df['has_feeding_problem'].mean()*100:.1f}% have a feeding problem")


# ==============================================================================
# STEP 3 - FEATURE ENGINEERING & CLEANING
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 3 - FEATURE ENGINEERING & CLEANING")
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
    "child_medical_complications",
    "delivery_method",
    "breastfeeding_duration_months",
    "MH_postnatal_depression",
    "MH_postnatal_anxiety",
]
TARGET = "has_feeding_problem"

df_model = df[FEATURES + [TARGET]].copy()

def clean_numeric(s):
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")

for col in ["maternal_age", "child_birthweight_kgs",
            "breastfeeding_duration_months", "child_medical_complications"]:
    df_model[col] = clean_numeric(df_model[col])

df_model["maternal_age"] = df_model["maternal_age"].replace(99, np.nan)
df_model["breastfeeding_duration_months"] = df_model[
    "breastfeeding_duration_months"].apply(
    lambda x: 0 if (pd.notna(x) and x < 0.01) else x)

for col in FEATURES:
    n_null = int(df_model[col].isna().sum())
    if n_null > 0:
        med = df_model[col].median()
        df_model[col] = df_model[col].fillna(med)
        print(f"  Imputed '{col}' ({n_null} NaNs) -> median = {med:.3f}")

print(f"\n  Cleaned shape : {df_model.shape}")
print(f"  Missing values: {df_model.isna().sum().sum()}")



# ==============================================================================
# STEP 5 - TRAIN / TEST SPLIT
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 5 - TRAIN / TEST SPLIT  (80/20 stratified)")
print("=" * 70)

X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")
print(f"  Train positive rate: {y_train.mean()*100:.1f}%")
print(f"  Test  positive rate: {y_test.mean()*100:.1f}%")


# ==============================================================================
# STEP 6 - SMOTE  (manual implementation - no imbalanced-learn needed)
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 6 - SMOTE  (training set only)")
print("=" * 70)

X_train_res, y_train_res = smote_resample(X_train, y_train, random_state=42)

bc_before = dict(zip(*np.unique(y_train,     return_counts=True)))
bc_after  = dict(zip(*np.unique(y_train_res, return_counts=True)))
print(f"  Before SMOTE : {bc_before}")
print(f"  After  SMOTE : {bc_after}")
print(f"  Clinical weights: {CLINICAL_WEIGHTS}  "
      f"(FN penalised {CLINICAL_WEIGHTS[1]}×)")

CONTINUOUS_IDX = [0, 1, 2, 3, 5]
BINARY_IDX     = [4, 6, 7]

preprocessor = ColumnTransformer([
    ("scale",       StandardScaler(), CONTINUOUS_IDX),
    ("passthrough", "passthrough",    BINARY_IDX),
])

X_train_res_s = preprocessor.fit_transform(X_train_res)
X_test_scaled = preprocessor.transform(X_test)

print(f"  ColumnTransformer: scales continuous, passes binary as-is OK")

before_counts = np.bincount(y_train)
after_counts  = np.bincount(y_train_res)
labels        = ["Class 0\n(No problem)", "Class 1\n(Has problem)"]


# ==============================================================================
# STEP 7 - MODEL DEFINITIONS
# ==============================================================================
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())

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


# ==============================================================================
# STEP 8 - TRAIN ALL MODELS
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 8 - TRAIN ALL MODELS")
print("=" * 70)

summary_rows   = []
trained        = {}
probs_test     = {}
preds_test     = {}
fold_summaries = {}

for name, estimator, scaled in MODEL_DEFS:
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

    # -- Artifact code (mirrored from gradient boosting reference) -------------
    # 1. Out-of-Fold probabilities via cross_val_predict (no data leakage)
    from sklearn.model_selection import cross_val_predict
    cv_oof = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    if scaled:
        from sklearn.pipeline import Pipeline as SkPipeline
        oof_pipeline = SkPipeline([
            ("scaler", StandardScaler()),
            ("model", clone(estimator))
        ])
    else:
        oof_pipeline = clone(estimator)

    try:
        y_train_prob_cv = cross_val_predict(
            oof_pipeline, X_train, y_train, cv=cv_oof,
            method='predict_proba', n_jobs=-1
        )[:, 1]
    except Exception:
        # Fallback: use training-set predictions if cross_val_predict fails
        y_train_prob_cv = None

    if y_train_prob_cv is not None:
        # 2. Dynamic best-threshold search on OOF predictions (maximise F1)
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
        print(f"  Best OOF threshold (CV, F1): {best_threshold_cv}")

        # 3. Train metrics computed purely on OOF cross-val predictions
        y_train_pred_cv = (y_train_prob_cv >= best_threshold_cv).astype(int)

        oof_train_metrics = {
            "accuracy":  float(accuracy_score(y_train, y_train_pred_cv)),
            "precision": float(precision_score(y_train, y_train_pred_cv, zero_division=0)),
            "recall":    float(recall_score(y_train, y_train_pred_cv, zero_division=0)),
            "f1":        float(f1_score(y_train, y_train_pred_cv, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_train, y_train_prob_cv)),
        }
        print(f"  OOF Train -> Acc={oof_train_metrics['accuracy']:.4f}  "
              f"F1={oof_train_metrics['f1']:.4f}  "
              f"AUC={oof_train_metrics['roc_auc']:.4f}")
    else:
        best_threshold_cv = 0.5
        oof_train_metrics = {}
        print(f"  (cross_val_predict not available for {name}, using defaults)")
    # -- End artifact code -----------------------------------------------------

    print(f"\n{'-'*70}")
    print(f"  MODEL : {name}")
    print(f"{'-'*70}")
    print(f"  Test -> Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Rec={rec:.4f}  F1={f1:.4f}  "
          f"ROC-AUC={auc_:.4f}  PR-AUC={ap:.4f}")
    print()
    print("  CLASSIFICATION REPORT:")
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
    print(f"\n  {N_FOLDS}-FOLD CV METRICS (SMOTE inside each fold - no leakage):")
    print("  " + fold_sum.to_string().replace("\n", "\n  "))


# ==============================================================================
# STEP 9 - LEADERBOARD
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 9 - LEADERBOARD (test-set scores)")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows).set_index("Model")
pd.set_option("display.float_format", "{:.4f}".format)
print(summary_df.to_string())


# ==============================================================================
# STEP 10 - HYPERPARAMETER TUNING (Random Forest)
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 10 - HYPERPARAMETER TUNING  (Randomized Search - Random Forest)")
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

rf_tuned.fit(X_train_res, y_train_res)

print(f"  Best params    : {rf_tuned.best_params_}")
print(f"  Best CV Recall : {rf_tuned.best_score_:.4f}")

y_pred_tuned = rf_tuned.predict(X_test)
y_prob_tuned = rf_tuned.predict_proba(X_test)[:, 1]

trained["RF Tuned"] = rf_tuned.best_estimator_
preds_test["RF Tuned"] = y_pred_tuned
probs_test["RF Tuned"] = y_prob_tuned

print(f"\n  Tuned RF on test ->"
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


# ==============================================================================
# STEP 12 - PROBABILITY CALIBRATION
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 12 - PROBABILITY CALIBRATION (Platt scaling - RF)")
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

print(f"  Calibrated RF -> Recall={recall_score(y_test,y_pred_cal):.4f}  "
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

all_model_defs = (
    {n: s for n, _, s in MODEL_DEFS} |
    {"RF Calibrated": False, "RF Tuned": False}
)


# ==============================================================================
# STEP 13 - OVERFITTING CHECK
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 13 - OVERFITTING CHECK  (Train F1 vs Test F1)")
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
    st    = "OVERFIT !" if gap > 0.15 else "OK OK"
    print(f"  {name:<25} {tr_f1:>10.4f} {te_f1:>10.4f} {gap:>8.4f} {st:>12}")


# ==============================================================================
# STEP 14 - THRESHOLD TUNING
# ==============================================================================
print("\n" + "=" * 70)
print(f"  STEP 14 - THRESHOLD TUNING  (target recall >= {RECALL_TARGET})")
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




# ==============================================================================
# STEP 20 - FINAL LEADERBOARD
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 20 - FINAL LEADERBOARD")
print("=" * 70)

final_df = pd.DataFrame(summary_rows).set_index("Model")
best_f1  = final_df["F1"].idxmax()
best_rec = final_df["Recall"].idxmax()
best_pr  = final_df["PR-AUC"].idxmax()

print(f"\n  Best by F1      -> {best_f1}")
print(f"  Best by Recall  -> {best_rec}   <- preferred clinically")
print(f"  Best by PR-AUC  -> {best_pr}")
print("\n  Full leaderboard:\n")
print(final_df.to_string())

# ==============================================================================
# STEP 21 - SAVE MODEL ARTIFACTS
# ==============================================================================
print("\n" + "=" * 70)
print("  STEP 21 - SAVE MODEL ARTIFACTS")
print("=" * 70)

from shared.model_artifacts import save_model_artifacts
from sklearn.pipeline import Pipeline

categorical_feature_names = [FEATURES[i] for i in BINARY_IDX]

for name, model in trained.items():
    print(f"  Saving artifacts for {name}...")
    
    y_pred = preds_test[name]
    y_prob = probs_test[name]
    
    if y_prob is None:
        auc_score = float('nan')
        y_prob_out = np.zeros_like(y_pred, dtype=float)
    else:
        auc_score = float(roc_auc_score(y_test, y_prob))
        y_prob_out = y_prob
        
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

    scaled = all_model_defs.get(name, False)
    if scaled:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    else:
        pipeline = Pipeline([
            ("classifier", model)
        ])

    best_threshold = threshold_results.get(name, {}).get("threshold", 0.5)

    train_metrics = {}
    if name in fold_summaries:
        mean_row = fold_summaries[name].loc["-- Mean --"]
        train_metrics = {
            "accuracy": float(mean_row.get("Accuracy", 0)),
            "precision": float(mean_row.get("Precision", 0)),
            "recall": float(mean_row.get("Recall", 0)),
            "f1": float(mean_row.get("F1", 0)),
            "roc_auc": float(mean_row.get("ROC-AUC", 0))
        }

    algo_id = name.lower().replace(" ", "")

    save_model_artifacts(
        pipeline=pipeline,
        outcome="feeding",
        outcome_id="feeding",
        algorithm=algo_id,
        label=name,
        description=f"{name} classifier trained on breastfeeding dataset.",
        target_column=TARGET,
        feature_order=FEATURES,
        categorical_features=categorical_feature_names,
        metrics_payload=metrics_payload,
        positive_label="Has Problem (1)",
        negative_label="No Problem (0)",
        threshold=best_threshold,
        y_test=y_test,
        y_prob=y_prob_out,
        train_metrics=train_metrics,
    )

print("\n" + "=" * 70)
print("  ALL DONE OK")
print("=" * 70)