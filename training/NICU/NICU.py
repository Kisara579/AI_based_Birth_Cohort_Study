import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------
# 1. Load your dataset  >>> YOU MUST FILL THIS ONE LINE <<<
# ----------------------------------------------------------
df = pd.read_csv(r"D:\Y2 1st\Group project-Mathru EHR\dataset_1000.csv")
  # <-- put dataset_1000.csv or your file

# ----------------------------------------------------------
# 2. Define target and features
# ----------------------------------------------------------
y = df["NICU_admission"].map({"yes":1, "no":0})   # convert target to 0/1

categorical_cols = [
    'Diabetes_Mellitus',
    'Family_History_High_BP',
    'Chronic_Hypertension',
    'PIH',
    'Preeclampsia',
    'Eclampsia',
    'Preexisting_Diabetes',
    'Current_Diabetes',
    'Advanced_maternal_age_gt40',
    'BMI_ge_25',
    'History_of_GDM',
    'Past_history_abruption',
    'Past_history_preterm_delivery'
]

numeric_cols = [
    'Period_of_Gestation_weeks',
    'BMI',
    'Parity',
    'HbA1c_percent',
    'OGTT_1h',
    'OGTT_2h',
    'FBS'
]

X = df[categorical_cols + numeric_cols]

# ----------------------------------------------------------
# 3. Preprocessor (One-Hot Encoding + passthrough numeric)
# ----------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# ----------------------------------------------------------
# 4. Build Pipeline
# ----------------------------------------------------------
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('logreg', LogisticRegression(max_iter=500))
])

# ----------------------------------------------------------
# 5. Split data
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------
# 6. Train model
# ----------------------------------------------------------
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 7. Predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# 8. Evaluation
# ----------------------------------------------------------
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))
