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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


DATA_PATH = Path("data/premature_delivery/processed/premature_delivery_processed.csv")

df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=["premature_delivery"])
y = df["premature_delivery"]

# Define feature groups
categorical_cols = [
    "SEX",
    "MARITAL",
    "LOUTCOME",
    "RACEMOM",
    "RACEDAD",
    "HISPMOM",
    "HISPDAD",
]

numeric_cols = [
    "FAGE",
    "GAINED",
    "VISITS",
    "MAGE",
    "FEDUC",
    "MEDUC",
    "TOTALP",
    "BDEAD",
    "TERMS",
    "CIGNUM",
    "DRINKNUM",
    "ANEMIA",
    "CARDIAC",
    "ACLUNG",
    "DIABETES",
    "HERPES",
    "HYDRAM",
    "HEMOGLOB",
    "HYPERCH",
    "HYPERPR",
    "ECLAMP",
    "CERVIX",
    "PINFANT",
    "PRETERM",
    "RENAL",
    "RHSEN",
    "UTERINE",
]

# Train test split
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

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Pipeline
base_svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", CalibratedClassifierCV(base_svm, cv=3))
])


# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
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
    outcome="pd",
    outcome_id="pd",
    algorithm="linearsvm",
    label="Linear SVM",
    description="Linear SVM classifier trained on birth dataset",
    target_column="premature_delivery",
    feature_order=X.columns.tolist(),
    categorical_features=categorical_features,
    metrics_payload=metrics_payload,
    positive_label="premature delivery(week < 37)",
    negative_label="full-term delivery(week >= 37)",
)