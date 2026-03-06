import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path("data/low_birth_weight/processed/lbw_train_processed.csv")

df = pd.read_csv(DATA_PATH)

# Separate features and target variable
X = df.drop(columns=["low_birth_weight"])
y = df["low_birth_weight"]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

# Logistic Regression model
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": float(accuracy_score(y_test, y_pred)),
    "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
    "ROC AUC": float(roc_auc_score(y_test, y_prob))
}

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred).tolist()

metrics_payload = {
    "metrics": metrics,
    "classification_report": report,
    "confusion_matrix": cm
}

print("\nModel Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save the model
ALGORITHM = "logisticregression"
MODEL_DIR = Path("backend/models/low_birth_weight") / ALGORITHM
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"
joblib.dump(pipeline, MODEL_PATH)

# Save meta for backend
META_PATH = MODEL_DIR / "meta.json"
meta = {
    "outcome_id": "lbw",
    "model_type": ALGORITHM,
    "target": "low_birth_weight",
    "positive_label": "Low Birth Weight (<2500g)",
    "negative_label": "Normal Birth Weight",
    "feature_order": X.columns.tolist(),
    "threshold": 0.5,
    "categorical_features": categorical_features
}
META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Save metrics
METRICS_PATH = MODEL_DIR / "metrics.json"
METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

print("\nSaved:")
print(" -", MODEL_PATH)
print(" -", META_PATH)
print(" -", METRICS_PATH)

# Update registry — mark this algorithm as trained and active
REGISTRY_PATH = Path("backend/models/low_birth_weight/registry.json")

if REGISTRY_PATH.exists():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
else:
    registry = {
        "outcome": "low_birth_weight",
        "outcome_id": "lbw",
        "active": ALGORITHM,
        "models": []
    }

registry["active"] = ALGORITHM

found = False
for entry in registry.get("models", []):
    if entry.get("algorithm") == ALGORITHM:
        entry["trained"] = True
        found = True
        break

if not found:
    registry.setdefault("models", []).append({
        "algorithm": ALGORITHM,
        "label": "Logistic Regression",
        "description": "Linear baseline classifier trained on CDC Natality data.",
        "trained": True
    })

REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
print(" -", REGISTRY_PATH)