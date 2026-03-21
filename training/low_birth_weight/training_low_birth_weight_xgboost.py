import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    

import pandas as pd
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import f1_score
import numpy as np

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

y_prob = pipeline.predict_proba(X_test)[:, 1]


thresholds = np.arange(0.1, 0.91, 0.05)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best threshold:", best_threshold)
print("Best F1:", best_f1)

y_pred = (y_prob >= best_threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}

report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

metrics_payload = {
    "metrics": metrics,
    "classification_report": report,
    "confusion_matrix": cm
}


print("\nModel Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")



from shared.model_artifacts import save_model_artifacts

save_model_artifacts(
    pipeline=pipeline,
    outcome="lbw",
    outcome_id="lbw",
    algorithm="xgboost",
    label="XGBoost",
    description="Gradient-boosted tree classifier trained on CDC Natality data.",
    target_column="low_birth_weight",
    feature_order=X.columns.tolist(),
    categorical_features=categorical_features,
    metrics_payload=metrics_payload,
    positive_label="Low Birth Weight (<2500g)",
    negative_label="Normal Birth Weight",
    threshold=best_threshold,
    y_test=y_test,
    y_prob=y_prob
)