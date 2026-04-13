import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
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
from sklearn.preprocessing import OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = Path("data/low_birth_weight/processed/lbw_train_processed.csv")

df = pd.read_csv(DATA_PATH)

# Separate features and target variable
X = df.drop(columns=["low_birth_weight"])
y = df["low_birth_weight"]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

numeric_transformer = SklearnPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = SklearnPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"
)

# Gradient Boosting model
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
        ("model", model)
])

pipeline.fit(X_train, y_train)

from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

# 1. Use 5-fold cross_val_predict to get Out-of-Fold probabilities safely
y_train_prob_cv = cross_val_predict(
    pipeline, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1
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
    outcome="lbw",
    outcome_id="lbw",
    algorithm="gradientboosting",
    label="Gradient Boosting",
    description="Gradient Boosting classifier trained on CDC Natality data.",
    target_column="low_birth_weight",
    feature_order=X.columns.tolist(),
    categorical_features=categorical_features,
    metrics_payload=metrics_payload,
    positive_label="Low Birth Weight (<2500g)",
    negative_label="Normal Birth Weight",
    threshold=best_threshold,
    y_test=y_test,
    y_prob=y_prob,
    train_metrics=train_metrics,
)
