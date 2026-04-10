import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

from pathlib import Path

# Load dataset from 'data' folder relative to project root
project_root = Path(__file__).resolve().parents[2]  # ../.. from training/postpartum_depression
data_path = project_root / "data" / "postpartum_depression" / "Processed" / "ppd.csv"

# do a sanity check
if not data_path.exists():
    raise FileNotFoundError(f"PPD data file not found: {data_path}")

# Load dataset
df = pd.read_csv(data_path)

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

# Fill missing values safely
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

# Preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ],
    remainder='passthrough'  # numeric columns remain as-is
)

# Build pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('xrt', ExtraTreesClassifier(n_estimators=100, random_state=42))
])

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))