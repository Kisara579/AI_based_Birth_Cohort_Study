import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from pathlib import Path
from shared.model_artifacts import save_model_artifacts

# Load dataset from 'data' folder relative to project root
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "postpartum_depression" / "Processed" / "ppd.csv"

# Sanity check
if not data_path.exists():
    raise FileNotFoundError(f"PPD data file not found: {data_path}")

# Load dataset
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Keep only required columns
columns_to_keep = [
    "EPDS Result",
    "Age",
    "Education Level",
    "Marital status",
    "Husband's education level",
    "Occupation After Your Latest Childbirth",
    "Recieved Support"
]
df = df[columns_to_keep]

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Split features and target
X = df.drop("EPDS Result", axis=1)
y = df["EPDS Result"]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ],
    remainder='passthrough'
)

# Model pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('xrt', ExtraTreesClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# =========================
# Training Performance
# =========================
y_train_pred = pipeline.predict(X_train)

print("=== Training Performance ===")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nTrain Classification Report:\n", classification_report(y_train, y_train_pred))

# =========================
# Testing Performance
# =========================
y_prob = pipeline.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.91, 0.05)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

best_threshold = float(round(best_threshold, 2))
print("Best threshold:", best_threshold)
print("Best F1:", best_f1)

y_test_pred = (y_prob >= best_threshold).astype(int)

print("\n=== Testing Performance ===")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

metrics = {
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
}

report = classification_report(y_test, y_test_pred, output_dict=True)
cm = confusion_matrix(y_test, y_test_pred).tolist()

metrics_payload = {
    "metrics": metrics,
    "classification_report": report,
    "confusion_matrix": cm,
}

print("\nModel Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# =========================
# Overfitting Check
# =========================
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n=== Overfitting Check ===")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")

if train_acc - test_acc > 0.05:
    print("⚠️ Model is likely OVERFITTING")
else:
    print("✅ Model is not overfitting significantly")

# =========================
# Cross Validation
# =========================
cv_scores = cross_val_score(pipeline, X, y, cv=5)

print("\n=== Cross Validation ===")
print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


save_model_artifacts(
    pipeline=pipeline,
    outcome="ppd",
    outcome_id="ppd",
    algorithm="extratrees",
    label="ExtraTrees",
    description="Extra Trees classifier trained on postpartum depression data.",
    target_column="EPDS Result",
    feature_order=X.columns.tolist(),
    categorical_features=categorical_cols,
    metrics_payload=metrics_payload,
    positive_label="Postpartum Depression",
    negative_label="No Postpartum Depression",
    threshold=best_threshold,
)