import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    auc
)

import xgboost as xgb

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
df = pd.read_csv('data/csection/raw/output_file_Copy.csv', encoding='latin1')
print("Initial Shape:", df.shape)

# -------------------------------------------------------------------
# 2. Select only the required features + target
# -------------------------------------------------------------------
selected_features = [
    'labor_induction',
    'spontaneous_labor_onset',
    'bmi_delivery',
    'bmi_pre',
    'bmi_difference',
    'maternal_age',
    'maternal_height',
    'maternal_weight_delivery',
    'maternal_weight_pre',
    'gestational_diabetes',
    'occiput_posterior'
]

target = 'cesarean_section'

# Keep only selected features and target
df = df[selected_features + [target]]

# Replace missing markers
df = df.replace(['?', ''], np.nan)

# -------------------------------------------------------------------
# 3. Define X and y
# -------------------------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------------------------------------------
# 4. Train-Test Split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# -------------------------------------------------------------------
# 5. Identify column types (all features are numeric here)
# -------------------------------------------------------------------
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# -------------------------------------------------------------------
# 6. Preprocessing Pipeline (only numeric imputation needed)
# -------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# -------------------------------------------------------------------
# 7. Handle class imbalance
# -------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight = {scale_pos_weight:.3f}")

# -------------------------------------------------------------------
# 8. XGBoost model
# -------------------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_jobs=-1
)

# -------------------------------------------------------------------
# 9. Full Pipeline
# -------------------------------------------------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# -------------------------------------------------------------------
# 10. Hyperparameter Grid
# -------------------------------------------------------------------
param_grid = {
    'model__max_depth': [3, 4, 5],
    'model__learning_rate': [0.03, 0.05, 0.1],
    'model__n_estimators': [100, 200],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0],
    'model__min_child_weight': [1, 3],
    'model__gamma': [0, 0.1]
}

# -------------------------------------------------------------------
# 11. Grid Search
# -------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)
print(f"\nBest CV ROC-AUC: {grid.best_score_:.4f}")

# -------------------------------------------------------------------
# 12. Cross-validation stability
# -------------------------------------------------------------------
cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=5, scoring='roc_auc'
)
print("\n5-Fold CV ROC-AUC Scores")
print(cv_scores)
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -------------------------------------------------------------------
# 13. Test Evaluation
# -------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

print("\nClassification Report")
print(classification_report(
    y_test, y_pred,
    target_names=['Vaginal', 'C-Section']
))

# -------------------------------------------------------------------
# 14. Precision-Recall AUC
# -------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"\nROC-AUC : {test_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"Accuracy: {test_acc:.4f}")

# -------------------------------------------------------------------
# 15. Overfitting Check
# -------------------------------------------------------------------
train_pred = best_model.predict(X_train)
train_proba = best_model.predict_proba(X_train)[:, 1]
train_acc = accuracy_score(y_train, train_pred)
train_auc = roc_auc_score(y_train, train_proba)

print("\nOverfitting Check")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Difference    : {train_acc - test_acc:.4f}")
print(f"\nTrain ROC-AUC: {train_auc:.4f}")
print(f"Test ROC-AUC : {test_auc:.4f}")
print(f"Difference   : {train_auc - test_auc:.4f}")

# -------------------------------------------------------------------
# 16. Confusion Matrix
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Vaginal', 'C-Section'],
            yticklabels=['Vaginal', 'C-Section'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 17. ROC Curve
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {test_auc:.3f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# 18. Feature Importance
# -------------------------------------------------------------------
try:
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['model'].feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Feature importance error:", e)

# -------------------------------------------------------------------
# 19. Save Model
# -------------------------------------------------------------------
joblib.dump(best_model, 'c_section_xgb_pipeline.pkl')
print("\nModel saved as 'c_section_xgb_pipeline.pkl'") 