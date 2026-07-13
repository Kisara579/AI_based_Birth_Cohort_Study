import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# 1. Load data with encoding fallback
try:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv('data/IOL/raw/output_file_Copy.csv', encoding='cp1252')

# 2. Define features and target
features = ['maternal_age', 'maternal_height', 'maternal_weight_pre', 'bmi_pre', 'gestational_diabetes', 'occiput_posterior']
target = 'labor_induction'
X = df[features]
y = df[target]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Leak-Proof Pipeline (Scaling is highly critical for SVM)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight='balanced', probability=True, random_state=42))
])

# 5. Fit pipeline
pipeline.fit(X_train, y_train)

# 6. Predictions
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

# 7. Overfitting Check & Metrics
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print("=== Overfitting Check ===")
print(f"Train ROC-AUC: {train_auc:.3f}\nTest ROC-AUC : {test_auc:.3f}\nDifference   : {train_auc - test_auc:.3f}")

print("\n=== Test Set Performance ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# 8. Leak-Proof Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nCross-validated ROC-AUC (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 9. Learning curves function
def plot_learning_curve(estimator, X, y, cv, scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, random_state=42, n_jobs=-1
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes_abs, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
    plt.plot(train_sizes_abs, np.mean(val_scores, axis=1), 'o-', color='orange', label='Cross-validation score')
    plt.xlabel('Training examples'); plt.ylabel('ROC-AUC'); plt.title('SVM Learning Curves'); plt.legend(); plt.grid(True)
    return plt

plot_learning_curve(pipeline, X_train, y_train, cv=cv)
plt.show()

# 10. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVM (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend(); plt.show()

# 11. Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
avg_precision = average_precision_score(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.show()

# 12. Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Induction', 'Induction'], yticklabels=['No Induction', 'Induction'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('SVM Confusion Matrix'); plt.show()