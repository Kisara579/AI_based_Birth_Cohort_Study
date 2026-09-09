"""
STACKING ENSEMBLE FOR IUGR PREDICTION (2 MODELS)
Combines: Logistic Regression + SMOTE and Random Forest + SMOTE
Meta-Learner: Logistic Regression
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'iugr_stacking_2models'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("STACKING ENSEMBLE (2 MODELS) FOR IUGR PREDICTION")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING AND PREPARING DATA")
print("=" * 70)

# Load data
df = pd.read_csv('data/IUGR/raw/birth_weight_dataset.csv')
print(f"\n✓ Loaded {len(df)} samples")

# Define features
feature_columns = [
    'age', 'has_diabetes', 'household_income', 'education_level',
    'number_of_prenatal_visits', 'pre_pregnancy_bmi',
    'smoking_status', 'alcohol_consumption'
]

target_column = 'birth_weight_category'

# Prepare features
X = df[feature_columns].copy()
y = (df[target_column] == 'Low').astype(int)

# ============================================================================
# HANDLE MISSING VALUES
# ============================================================================
print("\n" + "=" * 70)
print("CHECKING FOR MISSING VALUES")
print("=" * 70)

print("\nMissing values per column:")
print(X.isnull().sum())

print(f"\nRows before dropping: {len(X)}")
X = X.dropna()
y = y[X.index]
print(f"Rows after dropping: {len(X)}")
print(f"\nRemaining missing values: {X.isnull().sum().sum()}")

# ============================================================================
# Encode categorical variables
# ============================================================================
print("\nEncoding categorical variables...")

X['smoking_status'] = (X['smoking_status'] == 'Yes').astype(int)
X['alcohol_consumption'] = (X['alcohol_consumption'] == 'Yes').astype(int)
education_map = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}
X['education_level'] = X['education_level'].map(education_map)

feature_names = X.columns.tolist()
print(f"\nFeatures used ({len(feature_names)}): {feature_names}")

# Class distribution
class_counts = y.value_counts()
print(f"\nClass distribution:")
print(f"  Class 0 (Normal): {class_counts[0]} ({class_counts[0] / len(y) * 100:.1f}%)")
print(f"  Class 1 (Low/IUGR): {class_counts[1]} ({class_counts[1] / len(y) * 100:.1f}%)")
print(f"  Ratio: {class_counts[0] / class_counts[1]:.1f}:1")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL CONFIGURATION")
print("=" * 70)

print("""
Model Configuration
• Algorithm: Stacking Ensemble (2 Models)
• Base Models (Level 0): 
    - Logistic Regression (with SMOTE) - Highest Recall, F1, Balanced Acc, MCC
    - Random Forest (with SMOTE) - Highest Accuracy, Precision, AUC
• Meta-Learner (Level 1): Logistic Regression
• Stacking Method: predict_proba (uses predicted probabilities)
• Cross-Validation in Stacking: 5-Fold (prevents overfitting)
• SMOTE k_neighbors: 3
• SMOTE sampling_strategy: auto
• Scaling: STANDARDIZATION (StandardScaler)
• Cross-Validation Evaluation: 5-Fold Stratified
""")

# ============================================================================
# STEP 2: CREATE BASE MODELS AND STACKING ENSEMBLE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: CREATING MODELS")
print("=" * 70)

# Base Model 1: Logistic Regression (Best for Recall, F1, Balanced Acc, MCC)
logreg = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    penalty='l2',
    C=1.0,
    random_state=RANDOM_STATE
)

# Base Model 2: Random Forest (Best for Accuracy, Precision, AUC)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Meta-Learner: Logistic Regression
meta_learner = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    C=1.0,
    random_state=RANDOM_STATE
)

# Stacking Ensemble
stacking_ensemble = StackingClassifier(
    estimators=[
        ('lr', logreg),
        ('rf', rf)
    ],
    final_estimator=meta_learner,
    cv=5,  # Prevents overfitting
    stack_method='predict_proba'  # Uses predicted probabilities
)

print("\n✓ Base Models created:")
print("  - Logistic Regression + SMOTE: Highest Recall, F1, Balanced Acc, MCC")
print("  - Random Forest + SMOTE: Highest Accuracy, Precision, AUC")
print("\n✓ Meta-Learner: Logistic Regression")
print("\n✓ Stacking Ensemble created with 2 base models + Logistic Regression meta-learner")

# ============================================================================
# STEP 3: CROSS-VALIDATION WITH SMOTE (NO LEAKAGE)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: STRATIFIED 5-FOLD CROSS-VALIDATION WITH SMOTE")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Store metrics for each fold
fold_metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'balanced_acc': [], 'mcc': [], 'auc': []
}

# Store all predictions for final aggregation
all_y_true = []
all_y_pred = []
all_y_pred_proba = []

# For overfitting detection
train_scores = []
val_scores = []

print("\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scale features (fit ONLY on training data to prevent leakage)
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)

    # Apply SMOTE on training data ONLY (NO LEAKAGE)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Create base models for this fold
    fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                     min_samples_leaf=3, max_features='sqrt', bootstrap=True,
                                     random_state=RANDOM_STATE, n_jobs=-1)

    # Create stacking ensemble for this fold
    fold_stacking = StackingClassifier(
        estimators=[('lr', fold_logreg), ('rf', fold_rf)],
        final_estimator=LogisticRegression(solver='liblinear', max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method='predict_proba'
    )

    # Train stacking ensemble
    fold_stacking.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = fold_stacking.predict(X_val_scaled)
    y_pred_proba = fold_stacking.predict_proba(X_val_scaled)[:, 1]

    # Training score (for overfitting detection)
    y_train_pred = fold_stacking.predict(X_train_resampled)
    train_acc = accuracy_score(y_train_resampled, y_train_pred)
    train_scores.append(train_acc)

    # Validation accuracy
    val_acc = accuracy_score(y_val, y_pred)
    val_scores.append(val_acc)

    # Store predictions
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_pred_proba.extend(y_pred_proba)

    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    fold_metrics['accuracy'].append(acc)
    fold_metrics['precision'].append(prec)
    fold_metrics['recall'].append(rec)
    fold_metrics['f1'].append(f1)
    fold_metrics['balanced_acc'].append(bal_acc)
    fold_metrics['mcc'].append(mcc)
    fold_metrics['auc'].append(auc)

    print(f"Fold {fold}:")
    print(f"  Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
    print(f"  SMOTE applied: {len(X_train)} → {len(X_train_resampled)} samples")
    print()

# ============================================================================
# STEP 4: AGGREGATED METRICS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: AVERAGE METRICS (Mean ± Std over 5 folds)")
print("=" * 70)

print(f"\nAccuracy      : {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
print(f"Precision     : {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
print(f"Recall        : {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
print(f"F1            : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
print(f"Balanced_acc  : {np.mean(fold_metrics['balanced_acc']):.4f} ± {np.std(fold_metrics['balanced_acc']):.4f}")
print(f"MCC           : {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
print(f"AUC           : {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")

# ============================================================================
# STEP 5: FINAL CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: FINAL CLASSIFICATION REPORT")
print("=" * 70)

print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Low/IUGR']))

# Overall confusion matrix
overall_cm = confusion_matrix(all_y_true, all_y_pred)
tn, fp, fn, tp = overall_cm.ravel()
print(f"\nOverall Confusion Matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")
print(f"  Overall Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")

# Calculate and display additional metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\nAdditional Clinical Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  Positive Predictive Value (Precision): {ppv:.4f}")
print(f"  Negative Predictive Value: {npv:.4f}")

# ============================================================================
# STEP 6: OVERFITTING STATUS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: OVERFITTING STATUS")
print("=" * 70)

mean_train_acc = np.mean(train_scores)
mean_val_acc = np.mean(val_scores)
overfitting_gap = mean_train_acc - mean_val_acc

print(f"\nMean Training Accuracy: {mean_train_acc:.4f}")
print(f"Mean Validation Accuracy: {mean_val_acc:.4f}")
print(f"Overfitting Gap (Train - Val): {overfitting_gap:.4f}")

if overfitting_gap < 0.05:
    print("\n✓ OVERFITTING STATUS: LOW - Model generalizes well")
    print(f"  (Gap of {overfitting_gap:.4f} is acceptable)")
elif overfitting_gap < 0.10:
    print("\n⚠ OVERFITTING STATUS: MODERATE - Some overfitting detected")
    print(f"  (Gap of {overfitting_gap:.4f} suggests mild overfitting)")
else:
    print("\n✗ OVERFITTING STATUS: HIGH - Model is overfitting")
    print(f"  (Gap of {overfitting_gap:.4f} is too large)")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE (from Random Forest component)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: FEATURE IMPORTANCE (Random Forest from Ensemble)")
print("=" * 70)

# Train final model on full data to extract feature importance
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)
smote_final = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_resampled_final, y_resampled_final = smote_final.fit_resample(X_scaled_final, y)

final_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                  min_samples_leaf=3, max_features='sqrt', bootstrap=True,
                                  random_state=RANDOM_STATE, n_jobs=-1)
final_rf.fit(X_resampled_final, y_resampled_final)

importance = final_rf.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("\nFeature importance (by gain from Random Forest component):")
for i, idx in enumerate(sorted_idx):
    print(
        f"  {i + 1}. {feature_names[idx]:30s} : {importance[idx]:.4f} ({importance[idx] / sum(importance) * 100:.1f}%)")

# ============================================================================
# STEP 8: GENERATING 6 CHARTS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 8: GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# Chart 1: Per-fold performance (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'balanced_acc', 'auc']
colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_plot)))

for i, metric in enumerate(metrics_to_plot):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    values = fold_metrics[metric]
    ax.plot(range(1, 6), values, 'o-', color=colors[i], linewidth=2, markersize=8)
    ax.axhline(y=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} per Fold')
    ax.set_xticks(range(1, 6))
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Stacking Ensemble (LR+RF) - Per-Fold Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/stacking_2models_1_per_fold_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Average metrics bar chart
avg_metrics = {
    'Accuracy': np.mean(fold_metrics['accuracy']),
    'Precision': np.mean(fold_metrics['precision']),
    'Recall': np.mean(fold_metrics['recall']),
    'F1-Score': np.mean(fold_metrics['f1']),
    'Balanced Acc': np.mean(fold_metrics['balanced_acc']),
    'AUC': np.mean(fold_metrics['auc'])
}
std_metrics = {
    'Accuracy': np.std(fold_metrics['accuracy']),
    'Precision': np.std(fold_metrics['precision']),
    'Recall': np.std(fold_metrics['recall']),
    'F1-Score': np.std(fold_metrics['f1']),
    'Balanced Acc': np.std(fold_metrics['balanced_acc']),
    'AUC': np.std(fold_metrics['auc'])
}

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(avg_metrics.keys(), avg_metrics.values(), yerr=std_metrics.values(),
              capsize=5, color='steelblue', edgecolor='black')
ax.set_ylabel('Score')
ax.set_title('Stacking Ensemble (LR+RF) - Average Performance Metrics', fontweight='bold')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, avg_metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/stacking_2models_2_average_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Low/IUGR'],
            yticklabels=['Normal', 'Low/IUGR'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Stacking Ensemble (LR+RF) - Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/stacking_2models_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: ROC Curve with all folds
fig, ax = plt.subplots(figsize=(8, 6))

# Plot individual fold ROC curves
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                     min_samples_leaf=3, random_state=RANDOM_STATE)

    fold_stacking = StackingClassifier(
        estimators=[('lr', fold_logreg), ('rf', fold_rf)],
        final_estimator=LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
        cv=5,
        stack_method='predict_proba'
    )
    fold_stacking.fit(X_train_resampled, y_train_resampled)

    y_pred_proba = fold_stacking.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    auc_val = roc_auc_score(y_val, y_pred_proba)
    ax.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Fold {fold} (AUC={auc_val:.3f})')

# Calculate and plot mean ROC curve
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                     min_samples_leaf=3, random_state=RANDOM_STATE)

    fold_stacking = StackingClassifier(
        estimators=[('lr', fold_logreg), ('rf', fold_rf)],
        final_estimator=LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
        cv=5,
        stack_method='predict_proba'
    )
    fold_stacking.fit(X_train_resampled, y_train_resampled)

    y_pred_proba = fold_stacking.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)

mean_tpr /= 5
mean_auc = np.mean(fold_metrics['auc'])
ax.plot(mean_fpr, mean_tpr, 'b-', lw=2, label=f'Mean ROC (AUC={mean_auc:.3f})', color='darkred')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('Stacking Ensemble (LR+RF) - ROC Curves', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/stacking_2models_4_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Precision-Recall Curve
fig, ax = plt.subplots(figsize=(8, 6))
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                     min_samples_leaf=3, random_state=RANDOM_STATE)

    fold_stacking = StackingClassifier(
        estimators=[('lr', fold_logreg), ('rf', fold_rf)],
        final_estimator=LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
        cv=5,
        stack_method='predict_proba'
    )
    fold_stacking.fit(X_train_resampled, y_train_resampled)

    y_pred_proba = fold_stacking.predict_proba(X_val_scaled)[:, 1]
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
    ax.plot(recall_vals, precision_vals, lw=1, alpha=0.5, label=f'Fold {fold}')

ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision (Positive Predictive Value)')
ax.set_title('Stacking Ensemble (LR+RF) - Precision-Recall Curve', fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/stacking_2models_5_precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sorted_importance = importance[sorted_idx]
sorted_features = [feature_names[i] for i in sorted_idx]
colors_bar = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(sorted_features)))
bars = ax.barh(range(len(sorted_features)), sorted_importance, color=colors_bar, edgecolor='black')
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features)
ax.set_xlabel('Importance Score')
ax.set_title('Stacking Ensemble (LR+RF) - Feature Importance (Random Forest)', fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, sorted_importance):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center')
plt.tight_layout()
plt.savefig(f'{output_dir}/stacking_2models_6_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ All 6 charts saved to '{output_dir}/'")
print("  1. stacking_2models_1_per_fold_performance.png")
print("  2. stacking_2models_2_average_metrics.png")
print("  3. stacking_2models_3_confusion_matrix.png")
print("  4. stacking_2models_4_roc_curve.png")
print("  5. stacking_2models_5_precision_recall_curve.png")
print("  6. stacking_2models_6_feature_importance.png")

# ============================================================================
# STEP 9: THRESHOLD TUNING FOR CLINICAL USE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 9: THRESHOLD TUNING FOR CLINICAL DEPLOYMENT")
print("=" * 70)

# Train final model on full data for threshold tuning
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)
smote_full = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_resampled_full, y_resampled_full = smote_full.fit_resample(X_scaled_full, y)

final_stacking = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                      min_samples_leaf=3, random_state=RANDOM_STATE))
    ],
    final_estimator=LogisticRegression(solver='liblinear', max_iter=1000, random_state=RANDOM_STATE),
    cv=5,
    stack_method='predict_proba'
)
final_stacking.fit(X_resampled_full, y_resampled_full)

# Use validation set for threshold tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_full, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
smote_temp = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_train_resampled, y_train_resampled = smote_temp.fit_resample(X_train, y_train)
final_stacking.fit(X_train_resampled, y_train_resampled)

y_proba = final_stacking.predict_proba(X_val)[:, 1]

# Test different thresholds
thresholds = np.arange(0.1, 0.61, 0.05)
threshold_results = []

print("\nPerformance at different thresholds:")
print("-" * 70)
print(f"{'Threshold':<12} {'Recall':<12} {'Precision':<12} {'F1-Score':<12} {'Specificity':<12}")
print("-" * 70)

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    recall = recall_score(y_val, y_pred_thresh)
    precision = precision_score(y_val, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
    specificity = recall_score(1 - y_val, 1 - y_pred_thresh)

    threshold_results.append({
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'specificity': specificity
    })
    print(f"{threshold:<12.2f} {recall:<12.4f} {precision:<12.4f} {f1:<12.4f} {specificity:<12.4f}")

# Find optimal threshold for recall (clinical priority)
best_recall_idx = np.argmax([r['recall'] for r in threshold_results])
best_recall_thresh = threshold_results[best_recall_idx]['threshold']
best_recall_value = threshold_results[best_recall_idx]['recall']

# Get performance at 0.5 threshold
threshold_05 = [r for r in threshold_results if abs(r['threshold'] - 0.5) < 0.01][0]
default_recall = threshold_05['recall']

print("\n" + "-" * 70)
print(f"\n✓ Optimal threshold for maximum recall: {best_recall_thresh:.2f}")
print(f"  Recall at threshold 0.5: {default_recall:.4f}")
print(f"  Recall at threshold {best_recall_thresh:.2f}: {best_recall_value:.4f}")
print(f"  Improvement: {(best_recall_value - default_recall) * 100:.1f}% more IUGR cases detected")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("STACKING ENSEMBLE (2 MODELS) - FINAL SUMMARY")
print("=" * 70)

print(f"\n📊 Performance Summary (Mean ± Std over 5 folds):")
print(f"   • Accuracy      : {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
print(f"   • Precision     : {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
print(f"   • Recall        : {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
print(f"   • F1-Score      : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
print(f"   • Balanced Acc  : {np.mean(fold_metrics['balanced_acc']):.4f} ± {np.std(fold_metrics['balanced_acc']):.4f}")
print(f"   • MCC           : {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
print(f"   • AUC           : {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")

print(f"\n📈 Overfitting Status:")
print(f"   • Train Accuracy: {mean_train_acc:.4f}")
print(f"   • Val Accuracy  : {mean_val_acc:.4f}")
print(f"   • Gap           : {overfitting_gap:.4f}")
if overfitting_gap < 0.05:
    print(f"   • Status        : ✓ LOW OVERFITTING")
elif overfitting_gap < 0.10:
    print(f"   • Status        : ⚠ MODERATE OVERFITTING")
else:
    print(f"   • Status        : ✗ HIGH OVERFITTING")

print(f"\n📋 Ensemble Architecture:")
print(f"   Level 0 (Base Models):")
print(f"      • Logistic Regression + SMOTE - Highest Recall, F1, Balanced Acc, MCC")
print(f"      • Random Forest + SMOTE - Highest Accuracy, Precision, AUC")
print(f"   Level 1 (Meta-Learner):")
print(f"      • Logistic Regression - Combines base model predictions")

print(f"\n🎯 Clinical Deployment Recommendation:")
print(f"   • Use threshold: {best_recall_thresh:.2f} (not 0.5)")
print(f"   • Expected recall: {best_recall_value:.1%}")
print(f"\n   • Prediction ≥ {best_recall_thresh:.2f} → 'HIGH IUGR RISK - Further testing required'")
print(f"   • Prediction < {best_recall_thresh:.2f} → 'Low risk - Continue routine care'")

print(f"\n📁 All outputs saved to: '{output_dir}/'")
print("\n" + "=" * 70)
print("STACKING ENSEMBLE (2 MODELS) COMPLETED SUCCESSFULLY!")
print("=" * 70)

