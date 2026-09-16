"""
run_preeclampsia_xgb.py

Updated: adds
 - Accuracy vs dataset size (summary)
 - ROC curves (per n + combined)
 - Precision-Recall curves (per n + combined)
 - Learning curve (sklearn) per experiment
 - Confusion matrix (per experiment)
 - Feature importance (XGBoost importance or permutation importance fallback) per experiment

Usage: same as before. Save/inspect outputs in PLOTS_DIR and models in OUT_DIR.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# require xgboost
import xgboost as xgb

# -----------------------
# Config - EDIT IF NEEDED
# -----------------------
CSV_PATH = Path("data/preeclampsia/preeclampsia_sri_lanka_synthetic.csv")   # update the path if your file name differs
OUT_DIR = Path("models_xgb")
PLOTS_DIR = Path("plots_xgb")
RESULTS_CSV = Path("results_summary_xgb_preeclampsia.csv")
RANDOM_STATE = 42
SIZES = [300, 600, 1000, 1500]

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Load dataset
# -----------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("Dataset loaded:", df.shape)
print("Columns:", list(df.columns)[:30])

# -----------------------
# Column detection / mapping (common names)
# -----------------------
col_candidates = {
    'previous_pih': ['Previous_PIH_Preeclampsia','previous_pih','History_of_Preeclampsia','History_of_PIH'],
    'pre_preg_weight': ['Prepreg_weight_kg','prepreg_weight_kg','Prepreg_weight','pre_preg_weight'],
    'height': ['Height_cm','height_cm','Height','height'],
    'booking_bmi': ['Booking_BMI','booking_bmi','Prepreg_BMI','Prepreg_BMI'],
    'family_hbp': ['Family_history_high_BP','family_history_hbp','family_history_high_bp'],
    'age_years': ['Age_years','Age','age_years','age'],
    'age_over_35': ['Age_more_than_35','age_over_35','age_gt_35','age>35'],
    'prenatal_smoking': ['Prenatal_smoking','prenatal_smoking','Smoking_in_pregnancy','Smoking'],
    'outcome': ['Preeclampsia_outcome','preeclampsia_outcome','Preeclampsia','Outcome','preeclampsia']
}

def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

mapped = {k: find_col(df.columns, v) for k, v in col_candidates.items()}
print("\nAuto-detected mapping:")
for k, v in mapped.items():
    print(f"  {k:20} -> {v}")

# Check for essential columns
essentials = ['pre_preg_weight', 'booking_bmi', 'outcome']
missing = [e for e in essentials if mapped.get(e) is None]
if missing:
    raise ValueError(f"Essential columns not found (auto-detection failed): {missing}. Rename CSV headers or update mapping.")

# rename to short names
rename_map = {}
if mapped.get('previous_pih'): rename_map[mapped['previous_pih']] = 'previous_pih'
if mapped.get('pre_preg_weight'): rename_map[mapped['pre_preg_weight']] = 'pre_preg_weight'
if mapped.get('height'): rename_map[mapped['height']] = 'height_cm'
if mapped.get('booking_bmi'): rename_map[mapped['booking_bmi']] = 'booking_bmi'
if mapped.get('family_hbp'): rename_map[mapped['family_hbp']] = 'family_hbp'
if mapped.get('age_years'): rename_map[mapped['age_years']] = 'age_years'
if mapped.get('age_over_35'): rename_map[mapped['age_over_35']] = 'age_over_35'
if mapped.get('prenatal_smoking'): rename_map[mapped['prenatal_smoking']] = 'prenatal_smoking'
if mapped.get('outcome'): rename_map[mapped['outcome']] = 'Preeclampsia'

df = df.rename(columns=rename_map)
print("\nColumns after rename preview:", list(df.columns)[:30])

# -----------------------
# Convert yes/no -> 0/1 for binary columns
# -----------------------
def map_yesno_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).str.strip().str.lower().map(
        {'yes':1,'y':1,'true':1,'1':1,'no':0,'n':0,'false':0,'0':0}
    )

for col in ['previous_pih','family_hbp','age_over_35','prenatal_smoking']:
    if col in df.columns:
        df[col] = map_yesno_series(df[col])

if 'Preeclampsia' in df.columns:
    df['Preeclampsia'] = map_yesno_series(df['Preeclampsia'])
else:
    raise ValueError("Target column not found in data after mapping. Ensure Preeclampsia_outcome exists.")

# -----------------------
# Coerce numeric columns & fill NA
# -----------------------
for col in ['booking_bmi','pre_preg_weight','age_years','height_cm']:
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill numeric NA with median
df = df.fillna(df.median(numeric_only=True))

# If age_over_35 missing but age_years present, derive it
if 'age_over_35' not in df.columns and 'age_years' in df.columns:
    df['age_over_35'] = (df['age_years'] > 35).astype(int)

# -----------------------
# Final feature list (only available ones)
# -----------------------
feature_candidates = ['previous_pih','pre_preg_weight','height_cm','booking_bmi',
                      'family_hbp','age_years','age_over_35','prenatal_smoking']
features = [f for f in feature_candidates if f in df.columns]

print("\nFinal features used:", features)
print("Target distribution:\n", df['Preeclampsia'].value_counts())

# -----------------------
# Helper function for probabilities
# -----------------------
def safe_predict_proba(model, X):
    try:
        probs = model.predict_proba(X)
        # return probability of positive class (assume col 1 is positive)
        return probs[:, 1]
    except Exception:
        try:
            scores = model.decision_function(X)
            # map to probabilities via sigmoid
            return 1 / (1 + np.exp(-scores))
        except Exception:
            preds = model.predict(X)
            return preds.astype(float)

# plotting helpers
def plot_confusion_matrix_save(y_true, y_pred, outpath, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_feature_importance(model, X_train, y_train, feature_names, outpath, topk=20, random_state=RANDOM_STATE):
    """
    Prefer model.feature_importances_ (XGBoost), otherwise compute permutation importance.
    """
    # try native importance first
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp_series = pd.Series(imp, index=feature_names).sort_values(ascending=False)
            imp_series = imp_series.head(topk)
        elif hasattr(model, "get_booster"):
            # xgboost native
            booster = model.get_booster()
            fmap = booster.get_score(importance_type='weight')
            # booster.get_score returns dict of feature->score like 'f0':val; map to names
            # map f0 -> feature_names[0]
            mapping = {}
            for k, v in fmap.items():
                # key like 'f0', 'f10'
                try:
                    idx = int(k[1:])
                    mapping[feature_names[idx]] = v
                except Exception:
                    mapping[k] = v
            imp_series = pd.Series(mapping).sort_values(ascending=False).head(topk)
        else:
            raise AttributeError("No feature_importances_ available")
    except Exception:
        # fallback to permutation importance (slower)
        r = permutation_importance(clone(model), X_train, y_train, n_repeats=10, random_state=random_state, n_jobs=-1)
        imp_series = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False).head(topk)

    plt.figure(figsize=(8, max(4, 0.25*len(imp_series))))
    sns.barplot(x=imp_series.values, y=imp_series.index)
    plt.xlabel("Importance")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_learning_curve_save(estimator, X, y, outpath, cv=5, scoring='accuracy', n_jobs=-1):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 6), shuffle=True, random_state=RANDOM_STATE
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', label='Train score')
    plt.plot(train_sizes, test_mean, 'o-', label='CV score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# store results and curves
results = []
roc_curves = {}
pr_curves = {}

# -----------------------
# Main experiment loop (XGBoost only)
# -----------------------
for n in SIZES:
    print("\n" + "="*60)
    print(f"Experiment with n = {n}")
    print("="*60)

    if n >= len(df):
        df_n = df.copy()
        print("Requested n >= dataset size. Using full dataset.")
    else:
        df_n = df.sample(n=n, random_state=RANDOM_STATE)

    X_n = df_n[features].reset_index(drop=True)
    y_n = df_n['Preeclampsia'].astype(int).reset_index(drop=True)

    if y_n.nunique() < 2:
        print(f"Skipping n={n}: only one class present in target.")
        continue

    # 80/20 stratified split -> THIS is where train/test split happens
    X_train, X_test, y_train, y_test = train_test_split(
        X_n, y_n, test_size=0.2, stratify=y_n, random_state=RANDOM_STATE
    )

    print("Train samples:", len(X_train), "Test samples:", len(X_test))
    print("Train class distribution:", y_train.value_counts().to_dict())
    print("Test class distribution:", y_test.value_counts().to_dict())

    # compute scale_pos_weight (for imbalance)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print("scale_pos_weight:", scale_pos_weight)

    # build xgboost model
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # train
    model.fit(X_train, y_train)

    # predict & evaluate
    preds = model.predict(X_test)
    probs = safe_predict_proba(model, X_test)

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
    pr = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    print(f"\nXGBoost (n={n}) Accuracy: {acc*100:.2f}%")
    print(f"ROC AUC: {roc:.4f}    PR AUC: {pr:.4f}")
    print("Precision / Recall / F1:", prec, rec, f1)
    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))

    # store results
    results.append({
        'model': 'XGBoost',
        'n': n,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'pr_auc': pr
    })

    # store ROC/PR points
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_curves[n] = (fpr, tpr)
    except Exception:
        roc_curves[n] = (np.array([]), np.array([]))
    try:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_curves[n] = (precision, recall)
    except Exception:
        pr_curves[n] = (np.array([]), np.array([]))

    # save model
    out_path = OUT_DIR / f"xgb_preeclampsia_n{n}.joblib"
    joblib.dump(model, out_path)
    print("Saved XGBoost model to:", out_path)

    # ---- save confusion matrix (raw + normalized) (Commented out) ----
    # cm_path = PLOTS_DIR / f"confusion_matrix_n{n}.png"
    # plot_confusion_matrix_save(y_test, preds, cm_path, normalize=False)
    # cm_norm_path = PLOTS_DIR / f"confusion_matrix_normalized_n{n}.png"
    # plot_confusion_matrix_save(y_test, preds, cm_norm_path, normalize=True)
    # print("Saved confusion matrices to:", cm_path, cm_norm_path)

    # ---- save feature importance (Commented out) ----
    # fi_path = PLOTS_DIR / f"feature_importance_n{n}.png"
    # plot_feature_importance(model, X_train, y_train, feature_names=features, outpath=fi_path)
    # print("Saved feature importance to:", fi_path)

    # ---- save learning curve (using CV on the training set) (Commented out) ----
    # lc_path = PLOTS_DIR / f"learning_curve_n{n}.png"
    # try:
    #     plot_learning_curve_save(clone(model), X_train, y_train, lc_path, cv=min(5, max(2, int(len(X_train)/20))))
    #     print("Saved learning curve to:", lc_path)
    # except Exception as e:
    #     print("Could not compute learning curve for n", n, ":", e)


# -----------------------
# Save results summary CSV
# -----------------------
results_df = pd.DataFrame(results)
results_df = results_df[['model','n','n_train','n_test','accuracy','precision','recall','f1','roc_auc','pr_auc']]
results_df.to_csv(RESULTS_CSV, index=False)
print("\nSaved results to:", RESULTS_CSV)
print(results_df)

# -----------------------
# Plot Accuracy vs n (summary) (Commented out)
# -----------------------
# plt.figure(figsize=(8,5))
# sns.lineplot(data=results_df, x='n', y='accuracy', marker='o')
# plt.title("XGBoost Accuracy vs Sample Size (Preeclampsia)")
# plt.xlabel("Sample size (rows used)")
# plt.ylabel("Accuracy")
# plt.xticks(SIZES)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# acc_plot = PLOTS_DIR / "accuracy_vs_size_xgb_preeclampsia.png"
# plt.savefig(acc_plot, dpi=150)
# plt.close()
# print("Saved accuracy plot to:", acc_plot)

# -----------------------
# ROC and PR combined plots (Commented out)
# -----------------------
# plt.figure(figsize=(10,6))
# for n, (fpr, tpr) in roc_curves.items():
#     if fpr.size:
#         auc_val = np.trapz(tpr, fpr)
#         plt.plot(fpr, tpr, label=f"n={n} (AUC~{auc_val:.3f})")
# plt.plot([0,1],[0,1],'--', color='gray', linewidth=0.8)
# plt.title("ROC Curves - XGBoost (Preeclampsia)")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.tight_layout()
# roc_plot = PLOTS_DIR / "roc_xgb_preeclampsia_combined.png"
# plt.savefig(roc_plot, dpi=150)
# plt.close()
# print("Saved ROC plot to:", roc_plot)

# plt.figure(figsize=(10,6))
# for n, (precision, recall) in pr_curves.items():
#     if precision.size:
#         ap = np.trapz(precision, recall)  # approximate area
#         plt.plot(recall, precision, label=f"n={n} (A~{ap:.3f})")
# plt.title("Precision-Recall Curves - XGBoost (Preeclampsia)")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend()
# plt.tight_layout()
# pr_plot = PLOTS_DIR / "pr_xgb_preeclampsia_combined.png"
# plt.savefig(pr_plot, dpi=150)
# plt.close()
# print("Saved PR plot to:", pr_plot)

print("\nAll done. Models:", OUT_DIR, "Plots:", PLOTS_DIR, "Results CSV:", RESULTS_CSV)

# ============================================================================
# SAVE MODEL ARTIFACTS
# ============================================================================
from sklearn.pipeline import Pipeline
from shared.model_artifacts import save_model_artifacts

final_pipeline = Pipeline([
    ('classifier', model)
])

train_preds = model.predict(X_train)
train_probs = safe_predict_proba(model, X_train)

train_metrics = {
    "accuracy": float(accuracy_score(y_train, train_preds)),
    "precision": float(precision_score(y_train, train_preds, zero_division=0)),
    "recall": float(recall_score(y_train, train_preds, zero_division=0)),
    "f1": float(f1_score(y_train, train_preds, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_train, train_probs))
}

metrics_payload = {
    "metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc)
    },
    "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, preds).tolist()
}

save_model_artifacts(
    pipeline=final_pipeline,
    outcome="preeclampsia",
    outcome_id="preeclampsia",
    algorithm="xgboost",
    label="XGBoost",
    description="XGBoost model for predicting Preeclampsia.",
    target_column="Preeclampsia",
    feature_order=features,
    categorical_features=[],
    metrics_payload=metrics_payload,
    positive_label="Preeclampsia (1)",
    negative_label="Normal (0)",
    threshold=0.5,
    y_test=np.array(y_test),
    y_prob=np.array(probs),
    train_metrics=train_metrics,
    project_root=PROJECT_ROOT
)

