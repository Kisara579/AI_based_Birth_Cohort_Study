import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
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

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

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
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
pipeline.fit(X_train, y_train)

# Use 5-fold cross_val_predict to get out-of-fold probabilities on train split
y_train_prob_cv = cross_val_predict(
    pipeline, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1
)[:, 1]

# Find best threshold dynamically on OOF training predictions
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
print("Best F1 on CV:", best_f1)

# Compute train metrics on OOF predictions at best threshold
y_train_pred_cv = (y_train_prob_cv >= best_threshold).astype(int)

train_metrics = {
    "accuracy": float(accuracy_score(y_train, y_train_pred_cv)),
    "precision": float(precision_score(y_train, y_train_pred_cv, zero_division=0)),
    "recall": float(recall_score(y_train, y_train_pred_cv, zero_division=0)),
    "f1": float(f1_score(y_train, y_train_pred_cv, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_train, y_train_prob_cv)),
}

print("\n=== Training Performance (OOF CV) ===")
print("Train Accuracy:", train_metrics["accuracy"])
print("\nTrain Classification Report:\n", classification_report(y_train, y_train_pred_cv, zero_division=0))

# Predict on hold-out test set
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_prob >= best_threshold).astype(int)

print("\n=== Testing Performance ===")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

metrics = {
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
}

report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_test_pred).tolist()

metrics_payload = {
    "metrics": metrics,
    "classification_report": report,
    "confusion_matrix": cm,
}

print("\nModel Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


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
    y_test=y_test,
    y_prob=y_prob,
    train_metrics=train_metrics,
)