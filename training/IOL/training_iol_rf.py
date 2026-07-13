import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# 1. Load data
try:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='cp1252')

features = ['maternal_age', 'maternal_height', 'maternal_weight_pre', 'bmi_pre', 'gestational_diabetes', 'occiput_posterior']
target = 'labor_induction'
X = df[features]
y = df[target]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

# 4. Predictions & Overfitting Check
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print("=== Overfitting Check ===")
print(f"Train ROC-AUC: {train_auc:.3f}\nTest ROC-AUC : {test_auc:.3f}")

print("\n=== Test Set Performance ===")
print(classification_report(y_test, y_test_pred))

# 5. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nCross-validated ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 6. Feature Importances
importances = pd.Series(pipeline.named_steps['classifier'].feature_importances_, index=features)
print("\nFeature Importances (Gini Impurity):")
print(importances.sort_values(ascending=False))

# 7. Learning Curves
train_sizes_abs, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1
)
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
# plt.plot(train_sizes_abs, np.mean(val_scores, axis=1), 'o-', color='orange', label='Cross-validation score')
# plt.title('Random Forest Learning Curves'); plt.legend(); plt.grid(True); plt.show()

# 8. Feature Importance Plot
# plt.figure(figsize=(8, 5))
# importances.sort_values().plot(kind='barh')
# plt.title('Random Forest Feature Importance'); plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score
from shared.model_artifacts import save_model_artifacts

# Compute metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_test_proba))
}

train_metrics = {
    "accuracy": float(accuracy_score(y_train, y_train_pred)),
    "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
    "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
    "f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_train, y_train_proba))
}

metrics_payload = {
    "metrics": metrics,
    "classification_report": classification_report(y_test, y_test_pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
}

save_model_artifacts(
    pipeline=pipeline,
    outcome="iol",
    outcome_id="iol",
    algorithm="random_forest",
    label="Random Forest",
    description="Random Forest model for predicting Induction of Labor.",
    target_column="labor_induction",
    feature_order=features,
    categorical_features=[],
    metrics_payload=metrics_payload,
    positive_label="labor_induction (1)",
    negative_label="spontaneous_labor_onset (0)",
    threshold=0.5,
    y_test=y_test,
    y_prob=y_test_proba,
    train_metrics=train_metrics,
    project_root=PROJECT_ROOT
)

