import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/premature_delivery/raw/birth-weight-prediction/baby-weights-dataset.csv")

df = pd.read_csv(RAW_PATH)


print("Rows, Columns:", df.shape)
print("\nColumn count:", len(df.columns))
print("\nFirst 30 columns:\n", df.columns[:30].tolist())
print("\nLast 10 columns:\n", df.columns[-10:].tolist())
print("\nPreview (5 rows):\n", df.head())

print(df.columns.tolist())

print("\nColumn types:\n")
print(df.dtypes)


print("\nMissing values per column:\n")
print(df.isnull().sum())

print("\nWEEKS summary:\n")
print(df["WEEKS"].describe())

print("\nUnique values in WEEKS (sample):\n")
print(sorted(df["WEEKS"].dropna().unique())[:20])
print("...")
print(sorted(df["WEEKS"].dropna().unique())[-20:])

selected_columns = [
    "MAGE",
    "CARDIAC",
    "ACLUNG",
    "DIABETES",
    "HYPERCH",
    "HYPERPR",
    "PINFANT",
    "RENAL",
    "PRETERM",
    "WEEKS",
]

df = df[selected_columns].copy()

df = df.dropna(subset=["WEEKS"])

df["premature_delivery"] = (df["WEEKS"] < 37).astype(int)

print("\nTarget distribution:")
print(df["premature_delivery"].value_counts())

print("Shape:", df.shape)

df = df.drop(columns=["WEEKS"])

print("\nColumns after keeping only requested features and removing WEEKS:\n")
print(df.columns.tolist())

print("\nNew shape:", df.shape)

numeric_cols = [
    "MAGE",
    "CARDIAC",
    "ACLUNG",
    "DIABETES",
    "HYPERCH",
    "HYPERPR",
    "PINFANT",
    "RENAL",
    "PRETERM",
]

target_col = "premature_delivery"


# Fill numeric columns with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in {col} with median {median_val}")

# Check again
print("\nRemaining missing values:")
print(df.isnull().sum().sum())

# Group by unique feature combinations and assign the majority label
features = [col for col in df.columns if col != "premature_delivery"]
df = df.groupby(features)["premature_delivery"].agg(lambda x: x.mode()[0]).reset_index()

print("Final unique dataset shape:", df.shape)
print("Target summary:\n", df["premature_delivery"].value_counts())
print("\nSample of final dataset:\n", df.head())

PROCESSED_PATH = Path("data/premature_delivery/processed/premature_delivery_processed.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(PROCESSED_PATH, index=False)

print("\nSaved processed dataset to:")
print(PROCESSED_PATH)






















