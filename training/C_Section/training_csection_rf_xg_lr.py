import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv('data/csection/raw/output_file_Copy.csv', encoding='latin1')

print("Dataset Shape:", df.shape)

# =====================================================
# 2. SELECT FEATURES
# =====================================================

features = [
    "labor_induction",
    "spontaneous_labor_onset",
    "bmi_delivery",
    "bmi_pre",
    "bmi_difference",
    "maternal_age",
    "maternal_height",
    "maternal_weight_delivery",
    "maternal_weight_pre",
    "gestational_diabetes",
    "occiput_posterior"
]

target = "cesarean_section"

df = df[features + [target]]

# =====================================================
# 3. CLEAN DATA
# =====================================================

df = df.replace(["?", ""], np.nan)

X = df[features]
y = df[target]

# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# =====================================================
# 5. FEATURE TYPES
# =====================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric:", numeric_features)
print("Categorical:", categorical_features)

# =====================================================
# 6. PREPROCESSING
# =====================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =====================================================
# 7. CLASS IMBALANCE
# =====================================================

scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

print("\nscale_pos_weight =", scale_pos_weight)

# =====================================================
# 8. BASE MODELS
# =====================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

# =====================================================
# 9. VOTING ENSEMBLE
# =====================================================

voting_model = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lr", lr)
    ],
    voting="soft"
)

voting_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", voting_model)
    ]
)

# =====================================================
# 10. STACKING ENSEMBLE
# =====================================================

stacking_model = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lr", lr)
    ],
    final_estimator=LogisticRegression(
        max_iter=2000
    ),
    cv=5,
    n_jobs=-1
)

stacking_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", stacking_model)
    ]
)

# =====================================================
# 11. TRAIN VOTING
# =====================================================

print("\nTraining Voting Ensemble...")

voting_pipeline.fit(
    X_train,
    y_train
)

# =====================================================
# 12. EVALUATE VOTING
# =====================================================

vote_pred = voting_pipeline.predict(X_test)

vote_prob = voting_pipeline.predict_proba(
    X_test
)[:, 1]

vote_acc = accuracy_score(
    y_test,
    vote_pred
)

vote_auc = roc_auc_score(
    y_test,
    vote_prob
)

print("\n==============================")
print("VOTING ENSEMBLE")
print("==============================")

print("Accuracy:", vote_acc)
print("ROC-AUC :", vote_auc)

print(
    classification_report(
        y_test,
        vote_pred
    )
)

# =====================================================
# 13. TRAIN STACKING
# =====================================================

print("\nTraining Stacking Ensemble...")

stacking_pipeline.fit(
    X_train,
    y_train
)

# =====================================================
# 14. EVALUATE STACKING
# =====================================================

stack_pred = stacking_pipeline.predict(
    X_test
)

stack_prob = stacking_pipeline.predict_proba(
    X_test
)[:, 1]

stack_acc = accuracy_score(
    y_test,
    stack_pred
)

stack_auc = roc_auc_score(
    y_test,
    stack_prob
)

print("\n==============================")
print("STACKING ENSEMBLE")
print("==============================")

print("Accuracy:", stack_acc)
print("ROC-AUC :", stack_auc)

print(
    classification_report(
        y_test,
        stack_pred
    )
)

# =====================================================
# 15. CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

vote_cv = cross_val_score(
    voting_pipeline,
    X,
    y,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

stack_cv = cross_val_score(
    stacking_pipeline,
    X,
    y,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

print("\n==============================")
print("CROSS VALIDATION")
print("==============================")

print(
    "Voting ROC-AUC:",
    vote_cv.mean(),
    "+/-",
    vote_cv.std()
)

print(
    "Stacking ROC-AUC:",
    stack_cv.mean(),
    "+/-",
    stack_cv.std()
)

# =====================================================
# 16. SAVE MODELS
# =====================================================

# joblib.dump(
#     voting_pipeline,
#     "voting_ensemble.pkl"
# )

# joblib.dump(
#     stacking_pipeline,
#     "stacking_ensemble.pkl"
# )

# print("\nModels Saved Successfully")
# print("voting_ensemble.pkl")
# print("stacking_ensemble.pkl")

# =====================================================
# OVERFITTING CHECK - VOTING
# =====================================================

train_pred_vote = voting_pipeline.predict(X_train)
train_prob_vote = voting_pipeline.predict_proba(X_train)[:, 1]

train_acc_vote = accuracy_score(y_train, train_pred_vote)
train_auc_vote = roc_auc_score(y_train, train_prob_vote)

print("\n==============================")
print("VOTING OVERFITTING CHECK")
print("==============================")

print(f"Train Accuracy : {train_acc_vote:.4f}")
print(f"Test Accuracy  : {vote_acc:.4f}")

print(f"Train ROC-AUC  : {train_auc_vote:.4f}")
print(f"Test ROC-AUC   : {vote_auc:.4f}")

print(f"Accuracy Gap   : {train_acc_vote - vote_acc:.4f}")
print(f"ROC-AUC Gap    : {train_auc_vote - vote_auc:.4f}")

# =====================================================
# OVERFITTING CHECK - STACKING
# =====================================================

train_pred_stack = stacking_pipeline.predict(X_train)
train_prob_stack = stacking_pipeline.predict_proba(X_train)[:, 1]

train_acc_stack = accuracy_score(y_train, train_pred_stack)
train_auc_stack = roc_auc_score(y_train, train_prob_stack)

print("\n==============================")
print("STACKING OVERFITTING CHECK")
print("==============================")

print(f"Train Accuracy : {train_acc_stack:.4f}")
print(f"Test Accuracy  : {stack_acc:.4f}")

print(f"Train ROC-AUC  : {train_auc_stack:.4f}")
print(f"Test ROC-AUC   : {stack_auc:.4f}")

print(f"Accuracy Gap   : {train_acc_stack - stack_acc:.4f}")
print(f"ROC-AUC Gap    : {train_auc_stack - stack_auc:.4f}")

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from shared.model_artifacts import save_model_artifacts

# Save voting ensemble
save_model_artifacts(
    pipeline=voting_pipeline,
    outcome="c_section",
    outcome_id="c_section",
    algorithm="rf+xgb+lr voting_ensemble",
    label="Random Forest, XGBoost, Logistic Regression - Voting Ensemble",
    description="Voting Ensemble classifier combining Random Forest, XGBoost, and Logistic Regression.",
    target_column="cesarean_section",
    feature_order=X_train.columns.tolist(),
    categorical_features=categorical_features,
    metrics_payload={
        "metrics": {
            "accuracy": float(vote_acc),
            "precision": float(precision_score(y_test, vote_pred, zero_division=0)),
            "recall": float(recall_score(y_test, vote_pred, zero_division=0)),
            "f1": float(f1_score(y_test, vote_pred, zero_division=0)),
            "roc_auc": float(vote_auc)
        },
        "classification_report": classification_report(y_test, vote_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, vote_pred).tolist()
    },
    positive_label="cesarean_section (1)",
    negative_label="vaginal_delivery (0)",
    threshold=0.5,
    y_test=y_test,
    y_prob=vote_prob,
    train_metrics={
        "accuracy": float(train_acc_vote),
        "precision": float(precision_score(y_train, train_pred_vote, zero_division=0)),
        "recall": float(recall_score(y_train, train_pred_vote, zero_division=0)),
        "f1": float(f1_score(y_train, train_pred_vote, zero_division=0)),
        "roc_auc": float(train_auc_vote)
    },
    project_root=PROJECT_ROOT
)

# Save stacking ensemble
save_model_artifacts(
    pipeline=stacking_pipeline,
    outcome="c_section",
    outcome_id="c_section",
    algorithm="rf+xgb+lr stacking_ensemble",
    label="Random Forest, XGBoost, Logistic Regression - Stacking Ensemble",
    description="Stacking Ensemble classifier combining Random Forest, XGBoost, and Logistic Regression with a Logistic Regression meta-estimator.",
    target_column="cesarean_section",
    feature_order=X_train.columns.tolist(),
    categorical_features=categorical_features,
    metrics_payload={
        "metrics": {
            "accuracy": float(stack_acc),
            "precision": float(precision_score(y_test, stack_pred, zero_division=0)),
            "recall": float(recall_score(y_test, stack_pred, zero_division=0)),
            "f1": float(f1_score(y_test, stack_pred, zero_division=0)),
            "roc_auc": float(stack_auc)
        },
        "classification_report": classification_report(y_test, stack_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, stack_pred).tolist()
    },
    positive_label="cesarean_section (1)",
    negative_label="vaginal_delivery (0)",
    threshold=0.5,
    y_test=y_test,
    y_prob=stack_prob,
    train_metrics={
        "accuracy": float(train_acc_stack),
        "precision": float(precision_score(y_train, train_pred_stack, zero_division=0)),
        "recall": float(recall_score(y_train, train_pred_stack, zero_division=0)),
        "f1": float(f1_score(y_train, train_pred_stack, zero_division=0)),
        "roc_auc": float(train_auc_stack)
    },
    project_root=PROJECT_ROOT
)

