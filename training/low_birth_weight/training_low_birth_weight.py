import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

DATA_PATH = Path("data/low_birth_weight/processed/lbw_train_processed.csv")

df = pd.read_csv(DATA_PATH)

# Separate features and target variable
X = df.drop(columns=["low_birth_weight"])
y = df["low_birth_weight"]

# spilt data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object","string"]).columns.tolist()

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)


# xgboost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42,
    eval_metric="logloss"
)


pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}


print("\nModel Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")



# Save the model

MODEL_DIR = Path("backend/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lbw_model.pkl"
joblib.dump(pipeline, MODEL_PATH)


# Save meta for backend
META_PATH = MODEL_DIR / "lbw_model.meta.json"
meta = {
    "outcome_id": "lbw",
    "model_type": "xgboost",
    "target": "low_birth_weight",
    "positive_label": "Low Birth Weight (<2500g)",
    "negative_label": "Normal Birth Weight",
    "feature_order": X.columns.tolist(),
    "threshold": 0.5,
    "categorical_features": categorical_features
}
META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# save metrics 
METRICS_PATH = MODEL_DIR / "lbw_metrics.json"
METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("\nSaved:")
print(" -", MODEL_PATH)
print(" -", META_PATH)
print(" -", METRICS_PATH)