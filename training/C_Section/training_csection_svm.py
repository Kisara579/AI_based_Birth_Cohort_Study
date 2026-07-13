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

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

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
# 2. Select only the required features + target
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

# Keep only selected features and target
df = df[selected_features + [target]]

# Replace missing markers
df = df.replace(['?', ''], np.nan)

# -------------------------------------------------------------------
# 3. Define X and y
# -------------------------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------------------------------------------
# 4. Train-Test Split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# -------------------------------------------------------------------
# 5. Identify column types (numeric + categorical)
# -------------------------------------------------------------------
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# -------------------------------------------------------------------
# 6. Preprocessing Pipeline (numeric imputation + scaling)
# -------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())                # SVM requires scaling
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -------------------------------------------------------------------
# 7. SVM model (with class weight balancing)
# -------------------------------------------------------------------
svm_model = SVC(
    probability=True,               # to get predict_proba for ROC/PR curves
    class_weight='balanced',        # handles imbalance automatically
    random_state=42
)

# -------------------------------------------------------------------
# 8. Full Pipeline
# -------------------------------------------------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', svm_model)
])

# -------------------------------------------------------------------
# 9. Hyperparameter Grid for SVM
# -------------------------------------------------------------------
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__gamma': ['scale', 'auto', 0.1, 1],
    'model__kernel': ['rbf', 'linear']
}

# -------------------------------------------------------------------
# 10. Grid Search
# -------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)
print(f"\nBest CV ROC-AUC: {grid.best_score_:.4f}")

# -------------------------------------------------------------------
# 11. Cross-validation stability
# -------------------------------------------------------------------
cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=5, scoring='roc_auc'
)
print("\n5-Fold CV ROC-AUC Scores")
print(cv_scores)
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------------------------------------------
# 12. Test Evaluation
# -------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

print("\nClassification Report")
print(classification_report(
    y_test, y_pred,
    target_names=['Vaginal', 'C-Section']
))

# -------------------------------------------------------------------
# 13. Precision-Recall AUC
# -------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"\nROC-AUC : {test_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"Accuracy: {test_acc:.4f}")

# -------------------------------------------------------------------
# 14. Overfitting Check
# -------------------------------------------------------------------
train_pred = best_model.predict(X_train)
train_proba = best_model.predict_proba(X_train)[:, 1]
train_acc = accuracy_score(y_train, train_pred)
train_auc = roc_auc_score(y_train, train_proba)

print("\nOverfitting Check")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Difference    : {train_acc - test_acc:.4f}")
print(f"\nTrain ROC-AUC: {train_auc:.4f}")
print(f"Test ROC-AUC : {test_auc:.4f}")
print(f"Difference   : {train_auc - test_auc:.4f}")

# -------------------------------------------------------------------
# 15. Confusion Matrix
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            # xticklabels=['Vaginal', 'C-Section'],
            # yticklabels=['Vaginal', 'C-Section'])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.show()

# -------------------------------------------------------------------
# 16. ROC Curve
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
# 17. Feature Importance (SVM does not provide native importances)
# -------------------------------------------------------------------
print("\nSVM does not provide built-in feature importances. Skipping importance plot.")

# -------------------------------------------------------------------
# 18. Save Model
# -------------------------------------------------------------------
# joblib.dump(best_model, 'c_section_svm_pipeline.pkl')
# print("\nModel saved as 'c_section_svm_pipeline.pkl'")

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
    algorithm="svm",
    label="Support Vector Machine",
    description="Support Vector Machine model for predicting Caesarean Section.",
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

