import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# 1. Load data
try:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='cp1252')

features = ['maternal_age', 'maternal_height', 'maternal_weight_pre', 'bmi_pre', 'gestational_diabetes', 'occiput_posterior']
target = 'labor_induction'
X = df[features]
y = df[target]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
ratio = (len(y_train) - sum(y_train)) / sum(y_train)

# 3. Define base models for the ensemble
lr_base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
xgb_base = XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss')

# Create the voting wrapper
voting_ensemble = VotingClassifier(
    estimators=[('lr', lr_base), ('rf', rf_base), ('xgb', xgb_base)],
    voting='soft'  # 'soft' averages predicted probabilities for stable AUC calculation
)

# 4. Leak-Proof Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', voting_ensemble)
])

pipeline.fit(X_train, y_train)

# 5. Predictions & Metrics
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print("=== Overfitting Check ===")
print(f"Train ROC-AUC: {train_auc:.3f}\nTest ROC-AUC : {test_auc:.3f}")
print("\n=== Test Set Performance ===")
print(classification_report(y_test, y_test_pred))

# 6. Leak-Proof Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nCross-validated ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 7. Learning Curves
train_sizes_abs, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
plt.plot(train_sizes_abs, np.mean(val_scores, axis=1), 'o-', color='orange', label='Cross-validation score')
plt.title('Voting Ensemble Learning Curves'); plt.legend(); plt.grid(True); plt.show()