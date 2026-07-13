import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import joblib
import warnings
import os

# Suppress the specific parallel warning (works even from deep inside sklearn)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    auc
)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
df = pd.read_csv('data/csection/raw/output_file_Copy.csv', encoding='latin1')
print("Initial Shape:", df.shape)

# -------------------------------------------------------------------
# 2. Select features + target
# -------------------------------------------------------------------
selected_features = [
    'labor_induction',
    'spontaneous_labor_onset',
    'bmi_delivery',
    'bmi_pre',
    'bmi_difference',
    'maternal_age',
    'maternal_height',
    'maternal_weight_delivery',
    'maternal_weight_pre',
    'gestational_diabetes',
    'occiput_posterior'
]
target = 'cesarean_section'

df = df[selected_features + [target]]
df = df.replace(['?', ''], np.nan)

# -------------------------------------------------------------------
# 3. X and y
# -------------------------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------------------------------------------
# 4. Train-test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# -------------------------------------------------------------------
# 5. Column types
# -------------------------------------------------------------------
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
print("\nNumeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# -------------------------------------------------------------------
# 6. Preprocessing
# -------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -------------------------------------------------------------------
# 7. Random Forest (parallel enabled)
# -------------------------------------------------------------------
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])

# -------------------------------------------------------------------
# 8. Smaller hyperparameter distribution for RandomizedSearch
# -------------------------------------------------------------------
param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [5, 10, 15, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None],
    'model__bootstrap': [True, False]
}

# -------------------------------------------------------------------
# 9. RandomizedSearchCV (fast, parallel, warning suppressed)
# -------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,                # try only 30 random combinations (instead of exhaustive)
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,                # parallel processing for speed
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print("\nBest Parameters:", random_search.best_params_)
print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")

# -------------------------------------------------------------------
# 10. Cross-validation stability (parallel)
# -------------------------------------------------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print("\n5-Fold CV ROC-AUC Scores:", cv_scores)
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------------------------------------------
# 11. Test evaluation
# -------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=['Vaginal', 'C-Section']))

precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"\nROC-AUC : {test_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"Accuracy: {test_acc:.4f}")

# -------------------------------------------------------------------
# 12. Overfitting check
# -------------------------------------------------------------------
train_pred = best_model.predict(X_train)
train_proba = best_model.predict_proba(X_train)[:, 1]
train_acc = accuracy_score(y_train, train_pred)
train_auc = roc_auc_score(y_train, train_proba)

print("\nOverfitting Check")
print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f} | Diff: {train_acc - test_acc:.4f}")
print(f"Train ROC-AUC : {train_auc:.4f} | Test ROC-AUC : {test_auc:.4f} | Diff: {train_auc - test_auc:.4f}")

# -------------------------------------------------------------------
# 13. Confusion Matrix
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            # xticklabels=['Vaginal', 'C-Section'],
            # yticklabels=['Vaginal', 'C-Section'])
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

# -------------------------------------------------------------------
# 14. ROC Curve
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {test_auc:.3f}')
# plt.plot([0, 1], [0, 1], '--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid(True)
# plt.show()

# -------------------------------------------------------------------
# 15. Feature Importance
# -------------------------------------------------------------------
try:
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1][:15]

#     plt.figure(figsize=(10, 6))
#     plt.barh(range(len(indices)), importances[indices])
#     plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#     plt.gca().invert_yaxis()
#     plt.title("Top 15 Feature Importances (Random Forest)")
#     plt.tight_layout()
#     plt.show()
except Exception as e:
    print("Feature importance error:", e)

# -------------------------------------------------------------------
# 16. Save model
# -------------------------------------------------------------------
# joblib.dump(best_model, 'c_section_rf_pipeline.pkl')
# print("\nModel saved as 'c_section_rf_pipeline.pkl'")

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from shared.model_artifacts import save_model_artifacts

# Compute metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_proba))
}

train_metrics = {
    "accuracy": float(accuracy_score(y_train, train_pred)),
    "precision": float(precision_score(y_train, train_pred, zero_division=0)),
    "recall": float(recall_score(y_train, train_pred, zero_division=0)),
    "f1": float(f1_score(y_train, train_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_train, train_proba))
}

metrics_payload = {
    "metrics": metrics,
    "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

save_model_artifacts(
    pipeline=best_model,
    outcome="c_section",
    outcome_id="c_section",
    algorithm="random_forest",
    label="Random Forest",
    description="Random Forest model for predicting Caesarean Section.",
    target_column="cesarean_section",
    feature_order=X_train.columns.tolist(),
    categorical_features=categorical_features,
    metrics_payload=metrics_payload,
    positive_label="cesarean_section (1)",
    negative_label="vaginal_delivery (0)",
    threshold=0.5,
    y_test=y_test,
    y_prob=y_proba,
    train_metrics=train_metrics,
    project_root=PROJECT_ROOT
)

