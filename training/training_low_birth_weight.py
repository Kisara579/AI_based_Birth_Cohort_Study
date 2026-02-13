import pandas as pd
import numpy as np
import os


#load dataset
DATA_PATH = "data/low_birth_weight/raw/lbw1.xlsx"

df = pd.read_excel(DATA_PATH)

print("Original shape:", df.shape)


#create target variable

df["low_birth_weight"] = (df["bwght"] < 2500).astype(int)

print("Class distribution:")
print(df["low_birth_weight"].value_counts())

# processed dataset path
processed_path = "data/low_birth_weight/processed"
os.makedirs(processed_path, exist_ok=True)

df.to_csv(os.path.join(processed_path, "lbw1_processed.csv"), index=False)

print("Processed dataset saved.")

