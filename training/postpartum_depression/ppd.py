import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

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
y_test_pred = pipeline.predict(X_test)

print("\n=== Testing Performance ===")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

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