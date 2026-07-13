import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from lightgbm import LGBMClassifier

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv('data/csection/raw/output_file_Copy.csv', encoding='latin1')

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
df = df.replace(["?", ""], np.nan)

X = df[features]
y = df[target]

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# =====================================================
# PREPROCESSING
# =====================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# =====================================================
# CLASS IMBALANCE
# =====================================================

scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

# =====================================================
# BASE MODELS
# =====================================================

lr = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# =====================================================
# SOFT VOTING ENSEMBLE
# =====================================================

ensemble = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("lgbm", lgbm)
    ],
    voting="soft"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ensemble)
])

# =====================================================
# TRAIN
# =====================================================

pipeline.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy :", round(acc, 4))
print("ROC-AUC  :", round(auc, 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# =====================================================
# CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

print("\n5-Fold ROC-AUC Scores")
print(cv_scores)

print(
    f"\nMean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
)

# =====================================================
# OVERFITTING CHECK
# =====================================================

# Train predictions
y_train_pred = pipeline.predict(X_train)
y_train_prob = pipeline.predict_proba(X_train)[:, 1]

# Train metrics
train_acc = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_prob)

# Test metrics (already calculated above)
test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("OVERFITTING CHECK")
print("==============================")

print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")

print(f"\nTrain ROC-AUC  : {train_auc:.4f}")
print(f"Test ROC-AUC   : {test_auc:.4f}")

acc_gap = train_acc - test_acc
auc_gap = train_auc - test_auc

print(f"\nAccuracy Gap : {acc_gap:.4f}")
print(f"ROC-AUC Gap  : {auc_gap:.4f}")

if auc_gap > 0.10:
    print("\n⚠ Possible Overfitting Detected")
elif auc_gap < 0.02:
    print("\n✓ Very good generalization")
else:
    print("\n✓ Mild overfitting (acceptable)")

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    pipeline,
    "lr_lgbm_ensemble.pkl"
)

print("\nModel saved as lr_lgbm_ensemble.pkl")