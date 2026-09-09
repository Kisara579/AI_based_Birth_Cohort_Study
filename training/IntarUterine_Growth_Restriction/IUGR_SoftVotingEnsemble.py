import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
"""
SOFT VOTING ENSEMBLE FOR IUGR PREDICTION
Combines: Logistic Regression + Random Forest + XGBoost
All models use SMOTE for class balancing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
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
output_dir = 'iugr_soft_voting_ensemble'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 70)
print("SOFT VOTING ENSEMBLE FOR IUGR PREDICTION")
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
• Algorithm: Soft Voting Ensemble
• Components:
    - Logistic Regression (with SMOTE)
    - Random Forest (with SMOTE)
    - XGBoost (with SMOTE)
• Voting Type: SOFT (uses predicted probabilities)
• Ensemble Weights: [1.5, 1.0, 1.0] (LR weighted higher for recall)
• SMOTE k_neighbors: 3
• SMOTE sampling_strategy: auto
• Scaling: STANDARDIZATION (StandardScaler)
• Cross-Validation: 5-Fold Stratified
""")

# ============================================================================
# STEP 2: CREATE INDIVIDUAL MODELS AND ENSEMBLE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: CREATING MODELS")
print("=" * 70)

# Logistic Regression
logreg = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    penalty='l2',
    C=1.0,
    random_state=RANDOM_STATE
)

# Random Forest
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

# XGBoost
xgb = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False
)

# Soft Voting Ensemble (LR weighted higher for better recall)
voting_ensemble = VotingClassifier(
    estimators=[
        ('lr', logreg),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft',  # Uses predicted probabilities
    weights=[1.5, 1.0, 1.0]  # Give more weight to Logistic Regression
)

print("\n✓ Individual models created:")
print("  - Logistic Regression (Linear patterns)")
print("  - Random Forest (Nonlinear patterns + interactions)")
print("  - XGBoost (Complex boosting patterns)")
print("\n✓ Soft Voting Ensemble created with weights [1.5, 1.0, 1.0]")

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
train_metrics_folds = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
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

    # Create a new ensemble for this fold (to avoid leakage)
    fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, penalty='l2', C=1.0, random_state=RANDOM_STATE)
    fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                      min_samples_leaf=3, max_features='sqrt', bootstrap=True,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    fold_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, min_child_weight=3,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
                             random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)

    fold_ensemble = VotingClassifier(
        estimators=[('lr', fold_logreg), ('rf', fold_rf), ('xgb', fold_xgb)],
        voting='soft',
        weights=[1.5, 1.0, 1.0]
    )

    # Train ensemble
    fold_ensemble.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = fold_ensemble.predict(X_val_scaled)
    y_pred_proba = fold_ensemble.predict_proba(X_val_scaled)[:, 1]

    # Training score (for overfitting detection)
    y_train_pred = fold_ensemble.predict(X_train_resampled)
    y_train_proba = fold_ensemble.predict_proba(X_train_resampled)[:, 1]
    train_acc = accuracy_score(y_train_resampled, y_train_pred)
    train_scores.append(train_acc)
    train_metrics_folds['accuracy'].append(train_acc)
    train_metrics_folds['precision'].append(precision_score(y_train_resampled, y_train_pred, zero_division=0))
    train_metrics_folds['recall'].append(recall_score(y_train_resampled, y_train_pred, zero_division=0))
    train_metrics_folds['f1'].append(f1_score(y_train_resampled, y_train_pred, zero_division=0))
    train_metrics_folds['roc_auc'].append(roc_auc_score(y_train_resampled, y_train_proba))

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

# Train final ensemble on full data to extract feature importance
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
sorted_importance = importance[sorted_idx]
sorted_features = [feature_names[i] for i in sorted_idx]

print("\nFeature importance (by gain from Random Forest component):")
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1}. {feature_names[idx]:30s} : {importance[idx]:.4f} ({importance[idx]/sum(importance)*100:.1f}%)")

# ============================================================================
# STEP 8: GENERATING 6 CHARTS (COMMENTED OUT)
# ============================================================================

# print("\n" + "=" * 70)
# print("STEP 8: GENERATING VISUALIZATIONS")
# print("=" * 70)
# 
# # Set style for better-looking plots
# plt.style.use('seaborn-v0_8-darkgrid')
# 
# # Chart 1: Per-fold performance (2 rows, 3 columns)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'balanced_acc', 'auc']
# colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_plot)))
# 
# for i, metric in enumerate(metrics_to_plot):
#     row = i // 3
#     col = i % 3
#     ax = axes[row, col]
#     values = fold_metrics[metric]
#     ax.plot(range(1, 6), values, 'o-', color=colors[i], linewidth=2, markersize=8)
#     ax.axhline(y=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
#     ax.set_xlabel('Fold')
#     ax.set_ylabel(metric.capitalize())
#     ax.set_title(f'{metric.capitalize()} per Fold')
#     ax.set_xticks(range(1, 6))
#     ax.legend()
#     ax.grid(True, alpha=0.3)
# plt.suptitle('Soft Voting Ensemble - Per-Fold Performance', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/soft_voting_1_per_fold_performance.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 2: Average metrics bar chart
# avg_metrics = {
#     'Accuracy': np.mean(fold_metrics['accuracy']),
#     'Precision': np.mean(fold_metrics['precision']),
#     'Recall': np.mean(fold_metrics['recall']),
#     'F1-Score': np.mean(fold_metrics['f1']),
#     'Balanced Acc': np.mean(fold_metrics['balanced_acc']),
#     'AUC': np.mean(fold_metrics['auc'])
# }
# std_metrics = {
#     'Accuracy': np.std(fold_metrics['accuracy']),
#     'Precision': np.std(fold_metrics['precision']),
#     'Recall': np.std(fold_metrics['recall']),
#     'F1-Score': np.std(fold_metrics['f1']),
#     'Balanced Acc': np.std(fold_metrics['balanced_acc']),
#     'AUC': np.std(fold_metrics['auc'])
# }
# 
# fig, ax = plt.subplots(figsize=(12, 6))
# bars = ax.bar(avg_metrics.keys(), avg_metrics.values(), yerr=std_metrics.values(),
#               capsize=5, color='steelblue', edgecolor='black')
# ax.set_ylabel('Score')
# ax.set_title('Soft Voting Ensemble - Average Performance Metrics', fontweight='bold')
# ax.set_ylim(0, 1.1)
# ax.grid(True, alpha=0.3, axis='y')
# for bar, val in zip(bars, avg_metrics.values()):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
#             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/soft_voting_2_average_metrics.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 3: Confusion matrix heatmap
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Normal', 'Low/IUGR'],
#             yticklabels=['Normal', 'Low/IUGR'], ax=ax)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# ax.set_title('Soft Voting Ensemble - Confusion Matrix', fontweight='bold')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/soft_voting_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 4: ROC Curve with all folds
# fig, ax = plt.subplots(figsize=(8, 6))
# 
# # Plot individual fold ROC curves
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
# 
#     scaler_fold = StandardScaler()
#     X_train_scaled = scaler_fold.fit_transform(X_train)
#     X_val_scaled = scaler_fold.transform(X_val)
# 
#     smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
# 
#     fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
#     fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
#                                       min_samples_leaf=3, random_state=RANDOM_STATE)
#     fold_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
#                              random_state=RANDOM_STATE, eval_metric='logloss')
# 
#     fold_ensemble = VotingClassifier(
#         estimators=[('lr', fold_logreg), ('rf', fold_rf), ('xgb', fold_xgb)],
#         voting='soft', weights=[1.5, 1.0, 1.0]
#     )
#     fold_ensemble.fit(X_train_resampled, y_train_resampled)
# 
#     y_pred_proba = fold_ensemble.predict_proba(X_val_scaled)[:, 1]
#     fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
#     auc_val = roc_auc_score(y_val, y_pred_proba)
#     ax.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Fold {fold} (AUC={auc_val:.3f})')
# 
# # Calculate and plot mean ROC curve
# mean_fpr = np.linspace(0, 1, 100)
# mean_tpr = 0
# 
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
# 
#     scaler_fold = StandardScaler()
#     X_train_scaled = scaler_fold.fit_transform(X_train)
#     X_val_scaled = scaler_fold.transform(X_val)
# 
#     smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
# 
#     fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
#     fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
#                                       min_samples_leaf=3, random_state=RANDOM_STATE)
#     fold_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
#                              random_state=RANDOM_STATE, eval_metric='logloss')
# 
#     fold_ensemble = VotingClassifier(
#         estimators=[('lr', fold_logreg), ('rf', fold_rf), ('xgb', fold_xgb)],
#         voting='soft', weights=[1.5, 1.0, 1.0]
#     )
#     fold_ensemble.fit(X_train_resampled, y_train_resampled)
# 
#     y_pred_proba = fold_ensemble.predict_proba(X_val_scaled)[:, 1]
#     fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
#     mean_tpr += np.interp(mean_fpr, fpr, tpr)
# 
# mean_tpr /= 5
# mean_auc = np.mean(fold_metrics['auc'])
# ax.plot(mean_fpr, mean_tpr, 'b-', lw=2, label=f'Mean ROC (AUC={mean_auc:.3f})', color='darkred')
# 
# ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
# ax.set_xlabel('False Positive Rate (1 - Specificity)')
# ax.set_ylabel('True Positive Rate (Sensitivity)')
# ax.set_title('Soft Voting Ensemble - ROC Curves', fontweight='bold')
# ax.legend(loc='lower right')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/soft_voting_4_roc_curve.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 5: Precision-Recall Curve
# fig, ax = plt.subplots(figsize=(8, 6))
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
# 
#     scaler_fold = StandardScaler()
#     X_train_scaled = scaler_fold.fit_transform(X_train)
#     X_val_scaled = scaler_fold.transform(X_val)
# 
#     smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
# 
#     fold_logreg = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)
#     fold_rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
#                                       min_samples_leaf=3, random_state=RANDOM_STATE)
#     fold_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
#                              random_state=RANDOM_STATE, eval_metric='logloss')
# 
#     fold_ensemble = VotingClassifier(
#         estimators=[('lr', fold_logreg), ('rf', fold_rf), ('xgb', fold_xgb)],
#         voting='soft', weights=[1.5, 1.0, 1.0]
#     )
#     fold_ensemble.fit(X_train_resampled, y_train_resampled)
# 
#     y_pred_proba = fold_ensemble.predict_proba(X_val_scaled)[:, 1]
#     precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
#     ax.plot(recall_vals, precision_vals, lw=1, alpha=0.5, label=f'Fold {fold}')
# 
# ax.set_xlabel('Recall (Sensitivity)')
# ax.set_ylabel('Precision (Positive Predictive Value)')
# ax.set_title('Soft Voting Ensemble - Precision-Recall Curve', fontweight='bold')
# ax.legend(loc='best')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/soft_voting_5_precision_recall_curve.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 6: Feature importance bar chart
# # fig, ax = plt.subplots(figsize=(10, 6))
# # colors_bar = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(sorted_features)))
# # bars = ax.barh(range(len(sorted_features)), sorted_importance, color=colors_bar, edgecolor='black')
# # ax.set_yticks(range(len(sorted_features)))
# # ax.set_yticklabels(sorted_features)
# # ax.set_xlabel('Importance Score')
# # ax.set_title('Soft Voting Ensemble - Feature Importance (Random Forest Component)', fontweight='bold')
# # ax.invert_yaxis()
# # for bar, val in zip(bars, sorted_importance):
# #     ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
# # plt.tight_layout()
# # plt.savefig(f'{output_dir}/soft_voting_6_feature_importance.png', dpi=150, bbox_inches='tight')
# # plt.close()
# 
# # print(f"\n✓ All 6 charts saved to '{output_dir}/'")
# # print("  1. soft_voting_1_per_fold_performance.png")
# # print("  2. soft_voting_2_average_metrics.png")
# # print("  3. soft_voting_3_confusion_matrix.png")
# # print("  4. soft_voting_4_roc_curve.png")
# # print("  5. soft_voting_5_precision_recall_curve.png")
# # print("  6. soft_voting_6_feature_importance.png")

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

final_ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                       min_samples_leaf=3, random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                              random_state=RANDOM_STATE, eval_metric='logloss'))
    ],
    voting='soft',
    weights=[1.5, 1.0, 1.0]
)
final_ensemble.fit(X_resampled_full, y_resampled_full)

# Use validation set for threshold tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_full, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
smote_temp = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_train_resampled, y_train_resampled = smote_temp.fit_resample(X_train, y_train)
final_ensemble.fit(X_train_resampled, y_train_resampled)

y_proba = final_ensemble.predict_proba(X_val)[:, 1]

# Test different thresholds (including 0.5)
thresholds = np.arange(0.1, 0.61, 0.05)  # Now includes 0.5
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
    specificity = recall_score(1 - y_val, 1 - y_pred_thresh)  # Specificity = recall for negative class

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
default_precision = threshold_05['precision']
default_f1 = threshold_05['f1_score']

print("\n" + "-" * 70)
print(f"\n✓ Optimal threshold for maximum recall: {best_recall_thresh:.2f}")
print(f"  Recall at threshold 0.5: {default_recall:.4f}")
print(f"  Recall at threshold {best_recall_thresh:.2f}: {best_recall_value:.4f}")
print(f"  Improvement: {(best_recall_value - default_recall)*100:.1f}% more IUGR cases detected")

# ==================== SAVE MODEL ARTIFACTS ====================
from sklearn.pipeline import Pipeline
from shared.model_artifacts import save_model_artifacts

# Final ensemble fitted on full dataset with SMOTE
final_deploy_model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(solver='liblinear', max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                       min_samples_leaf=3, random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                              random_state=RANDOM_STATE, eval_metric='logloss'))
    ],
    voting='soft',
    weights=[1.5, 1.0, 1.0]
)
final_deploy_model.fit(X_resampled_full, y_resampled_full)

pipeline = Pipeline([('scaler', scaler_full), ('model', final_deploy_model)])

y_test_arr = np.array(all_y_true)
y_pred_arr = np.array(all_y_pred)
y_prob_arr = np.array(all_y_pred_proba)

fi_items = []
max_imp = float(np.max(importance)) if np.max(importance) > 0 else 1.0
for i in sorted_idx:
    fi_items.append({
        "feature": feature_names[i],
        "importance": round(float(importance[i] / max_imp), 4)
    })

metrics_payload = {
    "metrics": {
        "accuracy": float(accuracy_score(y_test_arr, y_pred_arr)),
        "precision": float(precision_score(y_test_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_test_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_test_arr, y_pred_arr, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_arr, y_prob_arr))
    },
    "classification_report": classification_report(y_test_arr, y_pred_arr, target_names=['Normal', 'Low/IUGR'], output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test_arr, y_pred_arr).tolist(),
    "feature_importance": fi_items,
    "feature_importance_v2": {
        "items": fi_items,
        "total_features": len(fi_items),
        "features_shown": len(fi_items)
    }
}

train_metrics = {
    "accuracy": float(np.mean(train_metrics_folds['accuracy'])),
    "precision": float(np.mean(train_metrics_folds['precision'])),
    "recall": float(np.mean(train_metrics_folds['recall'])),
    "f1": float(np.mean(train_metrics_folds['f1'])),
    "roc_auc": float(np.mean(train_metrics_folds['roc_auc']))
}

save_model_artifacts(
    pipeline=pipeline,
    outcome="iugr",
    outcome_id="iugr",
    algorithm="soft_voting_ensemble (Logistic Regression + Random Forest + XGBoost)",
    label="Soft Voting Ensemble(LR+RF+XGB)",
    description="Soft Voting Ensemble combining Logistic Regression, Random Forest, and XGBoost with SMOTE and StandardScaler for predicting Intrauterine Growth Restriction (IUGR).",
    target_column="birth_weight_category",
    feature_order=feature_names,
    categorical_features=[],
    metrics_payload=metrics_payload,
    positive_label="Low/IUGR (1)",
    negative_label="Normal (0)",
    threshold=0.5,
    y_test=y_test_arr,
    y_prob=y_prob_arr,
    train_metrics=train_metrics,
    project_root=PROJECT_ROOT
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SOFT VOTING ENSEMBLE - FINAL SUMMARY")
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

print(f"\n📋 Ensemble Components:")
print(f"   • Logistic Regression (weight: 1.5) - Linear patterns")
print(f"   • Random Forest (weight: 1.0) - Nonlinear patterns + interactions")
print(f"   • XGBoost (weight: 1.0) - Complex boosting patterns")

print(f"\n🎯 Clinical Deployment Recommendation:")
print(f"   • Use threshold: {best_recall_thresh:.2f} (not 0.5)")
print(f"   • Expected recall: {best_recall_value:.1%}")
print(f"   • Expected precision: {threshold_results[best_recall_idx]['precision']:.1%}")
print(f"   • Expected F1-Score: {threshold_results[best_recall_idx]['f1_score']:.1%}")
print(f"\n   • Prediction ≥ {best_recall_thresh:.2f} → 'HIGH IUGR RISK - Further testing required'")
print(f"   • Prediction < {best_recall_thresh:.2f} → 'Low risk - Continue routine care'")

print(f"\n📁 All outputs saved to: '{output_dir}/'")
print("\n" + "=" * 70)
print("SOFT VOTING ENSEMBLE COMPLETED SUCCESSFULLY!")
print("=" * 70)