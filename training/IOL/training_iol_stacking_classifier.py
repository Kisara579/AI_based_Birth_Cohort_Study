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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
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
ratio = (len(y_train) - sum(y_train)) / sum(y_train)

# 3. Define Stacking Ensemble
base_learners = [
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss'))
]
meta_learner = LogisticRegression(class_weight='balanced', random_state=42)

stacking_ensemble = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# 4. Pipeline Execution
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', stacking_ensemble)
])

pipeline.fit(X_train, y_train)

# 5. Predictions & Overfitting Metrics
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print("=== Overfitting Check ===")
print(f"Train ROC-AUC: {train_auc:.3f}\nTest ROC-AUC : {test_auc:.3f}\nDifference   : {train_auc - test_auc:.3f}")

print("\n=== Test Set Performance ===")
print(classification_report(y_test, y_test_pred))

# 6. Leak-Proof Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nCross-validated ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 7. Plot Learning Curves
train_sizes_abs, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1
)
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
# plt.plot(train_sizes_abs, np.mean(val_scores, axis=1), 'o-', color='orange', label='Cross-validation score')
# plt.xlabel('Training examples'); plt.ylabel('ROC-AUC'); plt.title('Stacking Ensemble Learning Curves'); plt.legend(); plt.grid(True)
# plt.show()

# 8. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f'Stacking (AUC = {test_auc:.3f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend(); plt.show()

# 9. Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
# plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.show()

# 10. Plot Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_test_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            # xticklabels=['No Induction', 'Induction'], yticklabels=['No Induction', 'Induction'])
# plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Stacking Confusion Matrix'); plt.show()

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
    algorithm="rf+xgb+lr stacking_classifier",
    label="Random Forest, Xgboost, Logistic Regression Stacking Ensemble",
    description="Stacking Classifier model for predicting Induction of Labor.",
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

