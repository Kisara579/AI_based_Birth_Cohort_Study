
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""
Random Forest for IUGR Prediction - WITH SMOTE (External Class Balancing)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'iugr_randomforest_smote'
os.makedirs(output_dir, exist_ok=True)

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("RANDOM FOREST WITH SMOTE (EXTERNAL CLASS BALANCING)")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING AND PREPARING DATA")
print("="*70)

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
print("\n" + "="*70)
print("CHECKING FOR MISSING VALUES")
print("="*70)

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
print(f"  Class 0 (Normal): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"  Class 1 (Low/IUGR): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)

print("""
Model Configuration
• Algorithm: Random Forest (with SMOTE)
• SMOTE k_neighbors: 3
• SMOTE sampling_strategy: auto
• n_estimators: 100
• max_depth: 5
• min_samples_split: 5
• min_samples_leaf: 3
• max_features: 'sqrt'
• class_weight: None (SMOTE handles balancing)
• Bootstrap: True
• Scaling: STANDARDIZATION (StandardScaler)
• Cross-Validation: 5-Fold Stratified
""")

# ============================================================================
# STEP 2: CROSS-VALIDATION WITH SMOTE
# ============================================================================

print("\n" + "="*70)
print("STEP 2: SMOTE WITH STRATIFIED 5-FOLD CROSS-VALIDATION")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'balanced_acc': [], 'mcc': [], 'auc': []
}
train_metrics_folds = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
}

all_y_true = []
all_y_pred = []
all_y_pred_proba = []
train_scores = []  # For overfitting detection
val_scores = []    # For overfitting detection

print("\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scale features
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)

    # Apply SMOTE on training data ONLY
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Train Random Forest (NO class_weight - SMOTE handles balancing)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = rf_model.predict(X_val_scaled)
    y_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]

    # Training score (on original training data)
    y_train_pred = rf_model.predict(X_train_scaled)
    y_train_proba = rf_model.predict_proba(X_train_scaled)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_acc)
    train_metrics_folds['accuracy'].append(train_acc)
    train_metrics_folds['precision'].append(precision_score(y_train, y_train_pred, zero_division=0))
    train_metrics_folds['recall'].append(recall_score(y_train, y_train_pred, zero_division=0))
    train_metrics_folds['f1'].append(f1_score(y_train, y_train_pred, zero_division=0))
    train_metrics_folds['roc_auc'].append(roc_auc_score(y_train, y_train_proba))

    # Validation metrics
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
    print(f"  Balanced Acc: {bal_acc:.4f}, AUC: {auc:.4f}")
    print()

# ============================================================================
# STEP 3: AGGREGATED METRICS
# ============================================================================

print("\n" + "="*70)
print("STEP 3: AVERAGE METRICS (Mean ± Std over 5 folds)")
print("="*70)

print(f"\nAccuracy      : {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
print(f"Precision     : {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
print(f"Recall        : {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
print(f"F1            : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
print(f"Balanced_acc  : {np.mean(fold_metrics['balanced_acc']):.4f} ± {np.std(fold_metrics['balanced_acc']):.4f}")
print(f"MCC           : {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
print(f"AUC           : {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")

# ============================================================================
# STEP 4: FINAL CLASSIFICATION REPORT WITH OVERFITTING STATUS
# ============================================================================

print("\n" + "="*70)
print("STEP 4: FINAL CLASSIFICATION REPORT")
print("="*70)

print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Low/IUGR']))

# Overall confusion matrix
overall_cm = confusion_matrix(all_y_true, all_y_pred)
tn, fp, fn, tp = overall_cm.ravel()
print(f"\nOverall Confusion Matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")
print(f"  Overall Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")

# ============================================================================
# OVERFITTING STATUS
# ============================================================================

print("\n" + "="*70)
print("STEP 5: OVERFITTING STATUS")
print("="*70)

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
    print("  Consider: reducing max_depth, increasing min_samples_split, or adding more regularization")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*70)
print("STEP 6: FEATURE IMPORTANCE (Random Forest)")
print("="*70)

# Train final model on full data with SMOTE
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)
smote_final = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_resampled_final, y_resampled_final = smote_final.fit_resample(X_scaled_final, y)

final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_model.fit(X_resampled_final, y_resampled_final)

importance = final_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
sorted_importance = importance[sorted_idx]
sorted_features = [feature_names[i] for i in sorted_idx]

print("\nFeature importance:")
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1}. {feature_names[idx]:30s} : {importance[idx]:.4f} ({importance[idx]/sum(importance)*100:.1f}%)")

# ============================================================================
# STEP 7: GENERATING 6 CHARTS (COMMENTED OUT)
# ============================================================================

# print("\n" + "="*70)
# print("STEP 7: GENERATING VISUALIZATIONS")
# print("="*70)
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
# plt.suptitle('Random Forest with SMOTE - Per-Fold Performance', fontsize=14)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/rf_smote_1_per_fold_performance.png', dpi=150, bbox_inches='tight')
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
# ax.set_title('Random Forest with SMOTE - Average Performance Metrics')
# ax.set_ylim(0, 1.1)
# ax.grid(True, alpha=0.3, axis='y')
# for bar, val in zip(bars, avg_metrics.values()):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
#             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/rf_smote_2_average_metrics.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 3: Confusion matrix heatmap
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Normal', 'Low/IUGR'],
#             yticklabels=['Normal', 'Low/IUGR'], ax=ax)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# ax.set_title('Random Forest with SMOTE - Confusion Matrix')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/rf_smote_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 4: ROC Curve
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
#     rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
#                                        min_samples_leaf=3, max_features='sqrt',
#                                        random_state=RANDOM_STATE)
#     rf_model.fit(X_train_resampled, y_train_resampled)
# 
#     y_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
#     fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
#     auc_val = roc_auc_score(y_val, y_pred_proba)
#     ax.plot(fpr, tpr, lw=1, alpha=0.7, label=f'Fold {fold} (AUC={auc_val:.3f})')
# 
# ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.set_title('Random Forest with SMOTE - ROC Curves')
# ax.legend(loc='lower right')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/rf_smote_4_roc_curve.png', dpi=150, bbox_inches='tight')
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
#     rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
#                                        min_samples_leaf=3, max_features='sqrt',
#                                        random_state=RANDOM_STATE)
#     rf_model.fit(X_train_resampled, y_train_resampled)
# 
#     y_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
#     precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
#     ax.plot(recall_vals, precision_vals, lw=1, alpha=0.7, label=f'Fold {fold}')
# 
# ax.set_xlabel('Recall')
# ax.set_ylabel('Precision')
# ax.set_title('Random Forest with SMOTE - Precision-Recall Curve')
# ax.legend(loc='best')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/rf_smote_5_precision_recall_curve.png', dpi=150, bbox_inches='tight')
# plt.close()
# 
# # Chart 6: Feature importance bar chart
# # fig, ax = plt.subplots(figsize=(10, 6))
# # bars = ax.barh(range(len(sorted_features)), sorted_importance, color='coral', edgecolor='black')
# # ax.set_yticks(range(len(sorted_features)))
# # ax.set_yticklabels(sorted_features)
# # ax.set_xlabel('Importance Score')
# # ax.set_title('Random Forest with SMOTE - Feature Importance')
# # ax.invert_yaxis()
# # for bar, val in zip(bars, sorted_importance):
# #     ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
# # plt.tight_layout()
# # plt.savefig(f'{output_dir}/rf_smote_6_feature_importance.png', dpi=150, bbox_inches='tight')
# # plt.close()
# 
# # print(f"\n✓ All 6 charts saved to '{output_dir}/'")

# ==================== SAVE MODEL ARTIFACTS ====================
from sklearn.pipeline import Pipeline
from shared.model_artifacts import save_model_artifacts

pipeline = Pipeline([('scaler', scaler_final), ('model', final_model)])

y_test_arr = np.array(all_y_true)
y_pred_arr = np.array(all_y_pred)
y_prob_arr = np.array(all_y_pred_proba)

metrics_payload = {
    "metrics": {
        "accuracy": float(accuracy_score(y_test_arr, y_pred_arr)),
        "precision": float(precision_score(y_test_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_test_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_test_arr, y_pred_arr, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_arr, y_prob_arr))
    },
    "classification_report": classification_report(y_test_arr, y_pred_arr, target_names=['Normal', 'Low/IUGR'], output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test_arr, y_pred_arr).tolist()
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
    algorithm="random_forest",
    label="Random Forest",
    description="Random Forest classifier with SMOTE oversampling and StandardScaler for predicting Intrauterine Growth Restriction (IUGR).",
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

print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

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
else:
    print(f"   • Status        : ⚠ MODERATE/HIGH OVERFITTING")

print(f"\n📋 Top 5 Most Important Features:")
for i in range(min(5, len(sorted_features))):
    print(f"   {i+1}. {sorted_features[i]:30s} : {sorted_importance[i]:.4f} ({sorted_importance[i]/sum(importance)*100:.1f}%)")

print("\n" + "="*70)
print("RANDOM FOREST WITH SMOTE COMPLETED SUCCESSFULLY!")
print("="*70)