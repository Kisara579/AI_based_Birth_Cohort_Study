import pandas as pd
from pathlib import Path
import numpy as np

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

# create temporary target just for checking
temp_target = (df["WEEKS"] < 37).astype(int)

print("\nPremature delivery count (WEEKS < 37):\n")
print(temp_target.value_counts())

print("\nPremature delivery percentage:\n")
print(temp_target.value_counts(normalize=True) * 100)

df = df.dropna(subset=["WEEKS"])

# create target variable
df["premature_delivery"] = (df["WEEKS"] < 37).astype(int)

print("\nTarget distribution:")
print(df["premature_delivery"].value_counts())

print("Shape:", df.shape)


drop_columns = ["ID", "WEEKS", "BWEIGHT"]

df = df.drop(columns=drop_columns)

print("\nColumns after dropping :\n")
print(df.columns.tolist())

print("\nNew shape:", df.shape)

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

target_col = "premature_delivery"

all_used = categorical_cols + numeric_cols + [target_col]

print("Categorical count:", len(categorical_cols))
print("Numeric count:", len(numeric_cols))
print("Total used:", len(all_used))
print("DF column count:", len(df.columns))

missing_from_df = [c for c in all_used if c not in df.columns]
extra_in_df = [c for c in df.columns if c not in all_used]

print("\nMissing from df:", missing_from_df)
print("Extra in df:", extra_in_df)


# Fill numeric columns with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in {col} with median {median_val}")

# Check again
print("\nRemaining missing values:")
print(df.isnull().sum().sum())


PROCESSED_PATH = Path("data/premature_delivery/processed/premature_delivery_processed.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(PROCESSED_PATH, index=False)

print("\nSaved processed dataset to:")
print(PROCESSED_PATH)

print("Final shape:", df.shape)