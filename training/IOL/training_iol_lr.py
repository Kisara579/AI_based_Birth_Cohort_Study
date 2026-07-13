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
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

try:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='cp1252')

# 2. Define features (pre-induction variables) and target
features = [
    'maternal_age',
    'maternal_height',
    'maternal_weight_pre',
    'bmi_pre',
    'gestational_diabetes',
    'occiput_posterior'
]
target = 'labor_induction'

X = df[features]
y = df[target]

# 3. Train/test split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build a Leak-Proof Pipeline
# This encapsulates imputation, scaling, and modeling. 
# During CV and learning curves, transformers will be fit ONLY on the current cv fold.
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# 5. Fit pipeline on full training set
pipeline.fit(X_train, y_train)

# 6. Predictions on train and test sets using the pipeline
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

# 7. Performance metrics & Overfitting Check
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print("=== Overfitting Check ===")
print(f"Train ROC-AUC: {train_auc:.3f}")
print(f"Test ROC-AUC : {test_auc:.3f}")
print(f"Difference   : {train_auc - test_auc:.3f}")
if train_auc - test_auc > 0.05:
    print("⚠️ Possible overfitting (train AUC > test AUC by > 0.05).")
else:
    print("✅ No strong overfitting detected.")

print("\n=== Test Set Performance ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print(f"ROC-AUC: {test_auc:.3f}")

# 8. Cross-validation (5-fold) on the PIPELINE (Fixes CV Leakage)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nCross-validated ROC-AUC (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 9. Feature coefficients (Extracted cleanly from the pipeline)
coefs = pd.Series(pipeline.named_steps['classifier'].coef_[0], index=features)
print("\nFeature Coefficients (scaled):")
print(coefs.sort_values(key=abs, ascending=False))

# 10. Learning curves function using the PIPELINE (Fixes Learning Curve Leakage)
# def plot_learning_curve(estimator, X, y, cv, scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 10)):
#     train_sizes_abs, train_scores, val_scores = learning_curve(
#         estimator, X, y, cv=cv, scoring=scoring,
#         train_sizes=train_sizes, random_state=42, n_jobs=-1
#     )
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     val_mean = np.mean(val_scores, axis=1)
#     val_std = np.std(val_scores, axis=1)

#     plt.figure(figsize=(8, 6))
#     plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
#     plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
#     plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
#     plt.plot(train_sizes_abs, val_mean, 'o-', color='orange', label='Cross-validation score')
#     plt.xlabel('Training examples')
#     plt.ylabel('ROC-AUC')
#     plt.title('Learning Curves')
#     plt.legend(loc='best')
#     plt.grid(True)
#     return plt

# 11. Plot learning curves using unscaled X_train because pipeline handles it internally now
# plot_learning_curve(pipeline, X_train, y_train, cv=cv)
# plt.show()

# 12. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {test_auc:.3f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()

# 13. Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.show()

# 14. Confusion matrix heatmap
cm = confusion_matrix(y_test, y_test_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            # xticklabels=['No Induction', 'Induction'],
            # yticklabels=['No Induction', 'Induction'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# 15. Feature coefficients bar plot
coefs_sorted = coefs.sort_values()
# plt.figure(figsize=(8, 5))
# coefs_sorted.plot(kind='barh')
# plt.xlabel('Coefficient (log-odds)')
# plt.title('Feature Importance (Logistic Regression Coefficients)')
# plt.grid(axis='x')
# plt.show()

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
    algorithm="logistic_regression",
    label="Logistic Regression",
    description="Logistic Regression model for predicting Induction of Labor.",
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

