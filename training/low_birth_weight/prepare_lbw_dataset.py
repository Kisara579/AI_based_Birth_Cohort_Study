import pandas as pd
from pathlib import Path
import numpy as np

RAW_PATH = Path("data/low_birth_weight/raw/prediction-interval-competition-i-birth-weight/train.csv")

df = pd.read_csv(RAW_PATH)

print("Rows, Columns:", df.shape)
print("\nColumn count:", len(df.columns))
print("\nFirst 30 columns:\n", df.columns[:30].tolist())
print("\nLast 10 columns:\n", df.columns[-10:].tolist())
print("\nPreview (5 rows):\n", df.head())

print(df.columns.tolist())

print("\nColumn types:\n")
print(df.dtypes)

# Create target
df["low_birth_weight"] = (df["DBWT"] < 2500).astype(int)

# keep wanted columns
df = df[["BMI", "CIG_0", "MAGER", "MEDUC", "PREVIS", "SEX", "low_birth_weight"]]


print("\nNew shape:", df.shape)
print("\nLBW distribution:\n", df["low_birth_weight"].value_counts())
print("\nTop 10 missing values:\n", df.isna().sum().sort_values(ascending=False).head(10))


# check correlation with target
print("\nCorrelation with LBW (top 10 strongest):\n")
corr = df.corr(numeric_only=True)["low_birth_weight"].abs().sort_values(ascending=False)
print(corr.head(15))





# Group by unique feature combinations and assign the majority label
features = [col for col in df.columns if col != "low_birth_weight"]
df = df.groupby(features)["low_birth_weight"].agg(lambda x: x.mode()[0]).reset_index()

print("Final unique dataset shape:", df.shape)
print("Target summary:\n", df["low_birth_weight"].value_counts())
print("\nSample of final dataset:\n", df.head())


PROCESSED_PATH = Path("data/low_birth_weight/processed/lbw_train_processed.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(PROCESSED_PATH, index=False)

print("\nSaved processed dataset to:")
print(PROCESSED_PATH)