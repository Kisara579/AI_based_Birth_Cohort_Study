import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("data/postpartum_depression/Processed/ppd.csv")
df.columns = df.columns.str.strip()


# =====================================================
# 2. SELECT REQUIRED COLUMNS
# =====================================================

columns_to_keep = [
    "EPDS Result",
    "Age",
    "Education Level",
    "Marital status",
    "Husband's education level",
    "Occupation After Your Latest Childbirth",
    "Recieved Support"
]

df = df[columns_to_keep].copy()


# =====================================================
# 3. REMOVE EMPTY TARGET VALUES
# =====================================================

df = df.dropna(subset=["EPDS Result"])


# =====================================================
# 4. SPLIT FEATURES AND TARGET
# =====================================================

X = df.drop("EPDS Result", axis=1)
y = df["EPDS Result"].astype(int)


# =====================================================
# 5. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 6. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# =====================================================

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

print("Numeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)


# =====================================================
# 7. PREPROCESSING
# =====================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# =====================================================
# 8. LOGISTIC REGRESSION MODEL
# =====================================================

base_model = LogisticRegression(
    C=0.5,
    max_iter=2000,
    random_state=42
)

model = BaggingClassifier(
    estimator=base_model,
    n_estimators=50,
    random_state=42
)


# =====================================================
# 9. PIPELINE WITH SMOTE
# =====================================================

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", model)
])


# =====================================================
# 10. TRAIN MODEL
# =====================================================

pipeline.fit(X_train, y_train)


# =====================================================
# 11. PREDICTIONS USING THRESHOLD 0.4
# =====================================================

threshold = 0.4

y_train_prob = pipeline.predict_proba(X_train)[:, 1]
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

y_train_pred = (y_train_prob >= threshold).astype(int)
y_test_pred = (y_test_prob >= threshold).astype(int)


# =====================================================
# 12. ACCURACY AND OVERFITTING CHECK
# =====================================================

training_accuracy = accuracy_score(y_train, y_train_pred)
testing_accuracy = accuracy_score(y_test, y_test_pred)
accuracy_difference = training_accuracy - testing_accuracy

if accuracy_difference > 0.10:
    overfitting_status = "Overfitting Detected"
elif accuracy_difference > 0.05:
    overfitting_status = "Slight Overfitting"
else:
    overfitting_status = "No Serious Overfitting"


# =====================================================
# 13. METRICS
# =====================================================

accuracy = testing_accuracy
precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_prob)


# =====================================================
# 14. 5-FOLD CROSS VALIDATION
# =====================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_accuracy = cross_val_score(
    pipeline,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)


# =====================================================
# 15. PRINT RESULTS
# =====================================================

print("\n========== BAGGING LOGISTIC REGRESSION RESULTS WITH SMOTE ==========")
print(f"Threshold           : {threshold}")
print(f"Training Accuracy   : {training_accuracy:.4f}")
print(f"Testing Accuracy    : {testing_accuracy:.4f}")
print(f"Accuracy Difference : {accuracy_difference:.4f}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"ROC AUC             : {roc_auc:.4f}")
print(f"CV Accuracy Scores  : {cv_accuracy}")
print(f"Mean CV Accuracy    : {np.mean(cv_accuracy):.4f}")
print(f"Overfitting Status  : {overfitting_status}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred, zero_division=0))


# =====================================================
# 16. CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_test_pred)

# ConfusionMatrixDisplay(cm).plot()
# plt.title("Confusion Matrix - Bagging Logistic Regression")
# plt.show()


# =====================================================
# 17. ROC CURVE
# =====================================================

# RocCurveDisplay.from_predictions(y_test, y_test_prob)
# plt.title("ROC Curve - Bagging Logistic Regression")
# plt.show()


# =====================================================
# 18. FEATURE IMPORTANCE
# =====================================================

from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    pipeline,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="f1"
)

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_importance.importances_mean
}).sort_values(by="Importance", ascending=False)

# =====================================================
# 19. FEATURE IMPORTANCE CHART
# =====================================================

# plt.figure(figsize=(10, 6))

# top_features = feature_importance_df.head(10)

# plt.barh(top_features["Feature"], top_features["Importance"])
# plt.gca().invert_yaxis()
# plt.xlabel("Absolute Coefficient Value")
# plt.title("Top 10 Feature Importance - Bagging Logistic Regression")
# plt.tight_layout()
# plt.show()


# =====================================================
# 20. SAVE MODEL ARTIFACTS
# =====================================================

# 1. Use 5-fold cross_val_predict to get Out-of-Fold probabilities safely
from sklearn.model_selection import cross_val_predict
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_prob_cv = cross_val_predict(
    pipeline, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
)[:, 1]

# 2. Find best threshold dynamically on the OOF training predictions to prevent leakage
thresholds = np.arange(0.1, 0.91, 0.05)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred_t = (y_train_prob_cv >= t).astype(int)
    f1 = f1_score(y_train, y_pred_t, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

best_threshold = float(round(best_threshold, 2))
print("Best threshold found on CV:", best_threshold)

# 3. Compute train_metrics purely on the cross_val predictions at the BEST threshold
y_train_pred_cv = (y_train_prob_cv >= best_threshold).astype(int)

train_metrics = {
    "accuracy": float(accuracy_score(y_train, y_train_pred_cv)),
    "precision": float(precision_score(y_train, y_train_pred_cv, zero_division=0)),
    "recall": float(recall_score(y_train, y_train_pred_cv, zero_division=0)),
    "f1": float(f1_score(y_train, y_train_pred_cv, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_train, y_train_prob_cv)),
}

# 4. Predict on absolute hold-out test set
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred).tolist()

metrics_payload = {
    "metrics": metrics,
    "classification_report": report,
    "confusion_matrix": cm
}

from shared.model_artifacts import save_model_artifacts

save_model_artifacts(
    pipeline=pipeline,
    outcome="ppd",
    outcome_id="ppd",
    algorithm="bagginglr",
    label="Bagging Logistic Regression",
    description="Bagging Logistic Regression classifier trained on postpartum depression data.",
    target_column="EPDS Result",
    feature_order=X.columns.tolist(),
    categorical_features=categorical_cols,
    metrics_payload=metrics_payload,
    positive_label="Postpartum Depression",
    negative_label="No Postpartum Depression",
    threshold=best_threshold,
    y_test=y_test,
    y_prob=y_prob,
    train_metrics=train_metrics,
)

