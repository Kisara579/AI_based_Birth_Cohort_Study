# """
# K-Nearest Neighbors (KNN) for GDM Prediction (Without Class Balancing)
# Using 5-Fold Stratified Cross-Validation with Standard Scaling
# Dataset: Gestational Diabetic Dat Set (3).xlsx
# """
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
#                              balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
#                              confusion_matrix, classification_report, roc_curve, auc,
#                              precision_recall_curve, average_precision_score)
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Load the dataset
# print("Loading data...")
# df = pd.read_excel("Gestational Diabetic Dat Set (3).xlsx", sheet_name="GDM-Final2022")
#
# print("\n" + "=" * 70)
# print("DATA SUMMARY")
# print("=" * 70)
# print(f"Total samples: {len(df)}")
#
# # Define features based on your dataset
# feature_names = ['Age', 'No of Pregnancy', 'BMI', 'Family History',
#                  'Large Child or Birth Default', 'PCOS', 'Sedentary Lifestyle']
#
# # Extract features (handle missing values)
# X = df[feature_names].copy()
#
# # Handle missing values in BMI (replace with median)
# if X['BMI'].isnull().any():
#     print(f"  Warning: {X['BMI'].isnull().sum()} missing values in BMI. Filling with median.")
#     X['BMI'].fillna(X['BMI'].median(), inplace=True)
#
# # Target variable
# y = df['Class Label(GDM /Non GDM)']
#
# print(f"Features used: {len(feature_names)}")
# print(f"Feature names: {feature_names}")
#
# # Class distribution
# class_counts = y.value_counts()
# print(f"\nClass distribution:")
# print(f"  Class 0 (Non-GDM): {class_counts[0]} ({class_counts[0] / len(y) * 100:.1f}%)")
# print(f"  Class 1 (GDM): {class_counts[1]} ({class_counts[1] / len(y) * 100:.1f}%)")
#
# # Create KNN pipeline with StandardScaler (KNN REQUIRES feature scaling)
# knn_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('knn', KNeighborsClassifier(
#         n_neighbors=5,  # Number of neighbors
#         weights='uniform',  # Uniform weights (all neighbors equally important)
#         algorithm='auto',  # Algorithm to compute nearest neighbors
#         leaf_size=30,  # Leaf size for BallTree or KDTree
#         p=2,  # Power parameter for Minkowski metric (2 = Euclidean distance)
#         metric='minkowski',  # Distance metric
#         n_jobs=-1  # Use all processors
#     ))
# ])
#
# # 5-fold stratified cross-validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# # Store metrics for each fold
# metrics = {
#     'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
#     'balanced_acc': [], 'mcc': [], 'auc': []
# }
#
# # Store predictions and probabilities for all folds
# all_y_true = []
# all_y_pred = []
# all_y_proba = []
#
# # Store training and validation scores for overfitting detection
# train_scores = []
# val_scores = []
#
# print("\n" + "=" * 70)
# print("KNN - 5-FOLD STRATIFIED CROSS-VALIDATION")
# print("=" * 70)
#
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#
#     # Clone the pipeline for each fold
#     from sklearn.base import clone
#
#     model = clone(knn_pipeline)
#
#     # Train model
#     model.fit(X_train, y_train)
#
#     # Predict on training set (for overfitting detection)
#     y_train_pred = model.predict(X_train)
#     train_acc = accuracy_score(y_train, y_train_pred)
#
#     # Predict on validation set
#     y_pred = model.predict(X_val)
#     y_proba = model.predict_proba(X_val)[:, 1]
#     val_acc = accuracy_score(y_val, y_pred)
#
#     train_scores.append(train_acc)
#     val_scores.append(val_acc)
#
#     # Calculate metrics
#     acc = accuracy_score(y_val, y_pred)
#     prec = precision_score(y_val, y_pred)
#     rec = recall_score(y_val, y_pred)
#     f1 = f1_score(y_val, y_pred)
#     bal_acc = balanced_accuracy_score(y_val, y_pred)
#     mcc = matthews_corrcoef(y_val, y_pred)
#     auc_score = roc_auc_score(y_val, y_proba)
#
#     # Store
#     metrics['accuracy'].append(acc)
#     metrics['precision'].append(prec)
#     metrics['recall'].append(rec)
#     metrics['f1'].append(f1)
#     metrics['balanced_acc'].append(bal_acc)
#     metrics['mcc'].append(mcc)
#     metrics['auc'].append(auc_score)
#
#     # Confusion matrix
#     tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
#
#     print(f"\nFold {fold}:")
#     print(f"  Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
#     print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {auc_score:.4f}")
#     print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
#
#     # Store all predictions for overall report
#     all_y_true.extend(y_val)
#     all_y_pred.extend(y_pred)
#     all_y_proba.extend(y_proba)
#
# # Calculate average metrics
# print("\n" + "=" * 70)
# print("AVERAGE METRICS (Mean ± Std)")
# print("=" * 70)
# for metric_name, values in metrics.items():
#     mean_val = np.mean(values)
#     std_val = np.std(values)
#     print(f"{metric_name.capitalize():15s}: {mean_val:.4f} ± {std_val:.4f}")
#
# # Calculate overfitting status
# avg_train_acc = np.mean(train_scores)
# avg_val_acc = np.mean(val_scores)
# overfitting_gap = avg_train_acc - avg_val_acc
#
# if overfitting_gap > 0.1:
#     overfitting_status = "HIGH - Model is significantly overfitting"
#     status_symbol = "⚠️"
# elif overfitting_gap > 0.05:
#     overfitting_status = "MODERATE - Some overfitting detected"
#     status_symbol = "⚡"
# elif overfitting_gap > 0.02:
#     overfitting_status = "LOW - Mild overfitting"
#     status_symbol = "📊"
# else:
#     overfitting_status = "NONE - Model generalizes well"
#     status_symbol = "✅"
#
# # Overall classification report
# print("\n" + "=" * 70)
# print("FINAL CLASSIFICATION REPORT (Aggregated over all folds)")
# print("=" * 70)
# print(classification_report(all_y_true, all_y_pred, target_names=['Non-GDM', 'GDM']))
#
# # Overall confusion matrix
# cm_total = confusion_matrix(all_y_true, all_y_pred)
# tn, fp, fn, tp = cm_total.ravel()
# total_acc = accuracy_score(all_y_true, all_y_pred)
# print(f"\nOverall Confusion Matrix:")
# print(f"  TN={tn}, FP={fp}")
# print(f"  FN={fn}, TP={tp}")
# print(f"  Accuracy: {total_acc:.4f}")
#
# # Overfitting Status
# print(f"\n" + "=" * 70)
# print("OVERFITTING STATUS")
# print("=" * 70)
# print(f"  Average Training Accuracy   : {avg_train_acc:.4f}")
# print(f"  Average Validation Accuracy : {avg_val_acc:.4f}")
# print(f"  Overfitting Gap             : {overfitting_gap:.4f}")
# print(f"  Status: {status_symbol} {overfitting_status}")
#
# # Note: KNN doesn't have built-in feature importance
# print("\n" + "=" * 70)
# print("FEATURE IMPORTANCE (KNN)")
# print("=" * 70)
# print("  Note: KNN is a distance-based algorithm and does not provide")
# print("  built-in feature importance scores. Feature importance can be")
# print("  estimated using permutation importance or coefficient of variation.")
# print("  For KNN, all features contribute equally when scaled properly.")
#
# # ============== PLOTTING (SEPARATE CHARTS) ==============
# print("\n" + "=" * 70)
# print("ANALYSIS COMPLETE - SUMMARY")
# print("=" * 70)
#
# print(f"\n📊 Performance Summary (Mean ± Std over 5 folds):")
# print(f"   • Accuracy      : {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
# print(f"   • Precision     : {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
# print(f"   • Recall        : {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
# print(f"   • F1-Score      : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
# print(f"   • Balanced Acc  : {np.mean(metrics['balanced_acc']):.4f} ± {np.std(metrics['balanced_acc']):.4f}")
# print(f"   • MCC           : {np.mean(metrics['mcc']):.4f} ± {np.std(metrics['mcc']):.4f}")
# print(f"   • AUC           : {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
#
# print(f"\n🎯 Overfitting Analysis:")
# print(f"   Training Accuracy   : {avg_train_acc:.4f}")
# print(f"   Validation Accuracy : {avg_val_acc:.4f}")
# print(f"   Gap                 : {overfitting_gap:.4f}")
# print(f"   Status              : {overfitting_status}")
#
# # ============== SAVE SEPARATE CHARTS ==============
#
# # Chart 1: Per-fold performance line chart
# plt.figure(figsize=(10, 6))
# folds = range(1, 6)
# for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
#     plt.plot(folds, metrics[metric], marker='o', linewidth=2, markersize=8, label=metric.capitalize())
# plt.xlabel('Fold', fontsize=12)
# plt.ylabel('Score', fontsize=12)
# plt.title('KNN - Per-Fold Performance Metrics', fontsize=14, fontweight='bold')
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.xticks(folds)
# plt.ylim([0.5, 1.0])
# plt.tight_layout()
# plt.savefig('knn_1_per_fold_performance.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("\n✅ Saved: knn_1_per_fold_performance.png")
#
# # Chart 2: Average metrics bar chart with error bars
# plt.figure(figsize=(12, 6))
# avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
# std_metrics = {k: np.std(v) for k, v in metrics.items()}
# metric_names = list(avg_metrics.keys())
# avg_values = list(avg_metrics.values())
# std_values = list(std_metrics.values())
#
# bars = plt.bar(metric_names, avg_values, yerr=std_values, capsize=8,
#                color='steelblue', edgecolor='black', alpha=0.8)
# plt.ylabel('Score', fontsize=12)
# plt.title('KNN - Average Metrics with Standard Deviation (5-fold CV)', fontsize=14, fontweight='bold')
# plt.ylim([0, 1])
# plt.xticks(rotation=45, ha='right')
# for bar, val in zip(bars, avg_values):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
# plt.grid(True, alpha=0.3, axis='y')
# plt.tight_layout()
# plt.savefig('knn_2_average_metrics.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("✅ Saved: knn_2_average_metrics.png")
#
# # Chart 3: Confusion matrix heatmap
# plt.figure(figsize=(8, 7))
# cm_display = confusion_matrix(all_y_true, all_y_pred)
# im = plt.imshow(cm_display, interpolation='nearest', cmap='Blues')
# plt.colorbar(im)
# plt.title(f'KNN - Confusion Matrix\nAccuracy: {total_acc:.3f}', fontsize=14, fontweight='bold')
# plt.xlabel('Predicted Label', fontsize=12)
# plt.ylabel('True Label', fontsize=12)
# plt.xticks([0, 1], ['Non-GDM', 'GDM'])
# plt.yticks([0, 1], ['Non-GDM', 'GDM'])
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, str(cm_display[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.savefig('knn_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("✅ Saved: knn_3_confusion_matrix.png")
#
# # Chart 4: ROC curve with AUC
# plt.figure(figsize=(8, 7))
# fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'KNN (AUC = {roc_auc:.3f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.500)')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
# plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
# plt.title('KNN - ROC Curve', fontsize=14, fontweight='bold')
# plt.legend(loc="lower right", fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('knn_4_roc_curve.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("✅ Saved: knn_4_roc_curve.png")
#
# # Chart 5: Precision-Recall curve
# plt.figure(figsize=(8, 7))
# precision_curve, recall_curve, _ = precision_recall_curve(all_y_true, all_y_proba)
# avg_precision = average_precision_score(all_y_true, all_y_proba)
# plt.plot(recall_curve, precision_curve, color='green', lw=2,
#          label=f'KNN (AP = {avg_precision:.3f})')
# plt.xlabel('Recall', fontsize=12)
# plt.ylabel('Precision', fontsize=12)
# plt.title('KNN - Precision-Recall Curve', fontsize=14, fontweight='bold')
# plt.legend(loc="lower left", fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.tight_layout()
# plt.savefig('knn_5_precision_recall_curve.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("✅ Saved: knn_5_precision_recall_curve.png")
#
# # Chart 6: Training vs Validation Accuracy (Overfitting plot)
# plt.figure(figsize=(8, 6))
# plt.plot(folds, train_scores, marker='o', linewidth=2, markersize=8, label='Training Accuracy', color='blue')
# plt.plot(folds, val_scores, marker='s', linewidth=2, markersize=8, label='Validation Accuracy', color='red')
# plt.xlabel('Fold', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# plt.title(f'KNN - Training vs Validation\nOverfitting Gap: {overfitting_gap:.4f}', fontsize=14, fontweight='bold')
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.xticks(folds)
# plt.ylim([0.5, 1.0])
# plt.tight_layout()
# plt.savefig('knn_6_overfitting_analysis.png', dpi=300, bbox_inches='tight')
# plt.close()
# print("✅ Saved: knn_6_overfitting_analysis.png")
#
# print("\n" + "=" * 70)
# print("✅ ALL CHARTS SAVED SUCCESSFULLY!")
# print("=" * 70)
# print("\n📁 Generated files:")
# print("   1. knn_1_per_fold_performance.png - Per-fold metrics line chart")
# print("   2. knn_2_average_metrics.png - Average metrics bar chart with error bars")
# print("   3. knn_3_confusion_matrix.png - Confusion matrix heatmap")
# print("   4. knn_4_roc_curve.png - ROC curve with AUC score")
# print("   5. knn_5_precision_recall_curve.png - Precision-Recall curve")
# print("   6. knn_6_overfitting_analysis.png - Training vs Validation accuracy plot")
#
# print("\n" + "=" * 70)
# print("MODEL CONFIGURATION")
# print("=" * 70)
# print("• Algorithm: K-Nearest Neighbors (KNN)")
# print("• Class Balancing: None")
# print("• n_neighbors: 5")
# print("• weights: uniform")
# print("• metric: Euclidean (p=2)")
# print("• algorithm: auto")
# print("• Scaling: StandardScaler (REQUIRED for KNN)")
# print("• Cross-Validation: 5-Fold Stratified")

"""
K-Nearest Neighbors (KNN) for GDM Prediction (WITH SMOTE Class Balancing)
Using 5-Fold Stratified Cross-Validation with Standard Scaling
Dataset: Gestational Diabetic Dat Set (3).xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading data...")
df = pd.read_excel("Gestational Diabetic Dat Set (3).xlsx", sheet_name="GDM-Final2022")

print("\n" + "="*70)
print("DATA SUMMARY")
print("="*70)
print(f"Total samples: {len(df)}")

# Define features based on your dataset
feature_names = ['Age', 'No of Pregnancy', 'BMI', 'Family History',
                 'Large Child or Birth Default', 'PCOS', 'Sedentary Lifestyle']

# Extract features (handle missing values)
X = df[feature_names].copy()

# Handle missing values in BMI (replace with median)
if X['BMI'].isnull().any():
    print(f"  Warning: {X['BMI'].isnull().sum()} missing values in BMI. Filling with median.")
    X['BMI'].fillna(X['BMI'].median(), inplace=True)

# Target variable
y = df['Class Label(GDM /Non GDM)']

print(f"Features used: {len(feature_names)}")
print(f"Feature names: {feature_names}")

# Class distribution BEFORE SMOTE
class_counts_before = y.value_counts()
print(f"\nClass distribution BEFORE SMOTE:")
print(f"  Class 0 (Non-GDM): {class_counts_before[0]} ({class_counts_before[0]/len(y)*100:.1f}%)")
print(f"  Class 1 (GDM): {class_counts_before[1]} ({class_counts_before[1]/len(y)*100:.1f}%)")

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for each fold
metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'balanced_acc': [], 'mcc': [], 'auc': []
}

# Store predictions and probabilities for all folds
all_y_true = []
all_y_pred = []
all_y_proba = []

# Store training and validation scores for overfitting detection
train_scores = []
val_scores = []

print("\n" + "="*70)
print("KNN WITH SMOTE - 5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE for first fold
    if fold == 1:
        print(f"\n  Fold {fold} - Class distribution AFTER SMOTE:")
        unique, counts = np.unique(y_train_resampled, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"    Class {cls}: {count} samples")

    # Scale the resampled training data (KNN REQUIRES scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)

    # Initialize KNN model
    knn_model = KNeighborsClassifier(
        n_neighbors=5,          # Number of neighbors
        weights='uniform',      # Uniform weights (all neighbors equally important)
        algorithm='auto',       # Algorithm to compute nearest neighbors
        leaf_size=30,           # Leaf size for BallTree or KDTree
        p=2,                    # Power parameter for Minkowski metric (2 = Euclidean distance)
        metric='minkowski',     # Distance metric
        n_jobs=-1               # Use all processors
    )

    # Train model on resampled and scaled data
    knn_model.fit(X_train_scaled, y_train_resampled)

    # Predict on training set (for overfitting detection)
    y_train_pred = knn_model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train_resampled, y_train_pred)

    # Predict on validation set
    y_pred = knn_model.predict(X_val_scaled)
    y_proba = knn_model.predict_proba(X_val_scaled)[:, 1]
    val_acc = accuracy_score(y_val, y_pred)

    train_scores.append(train_acc)
    val_scores.append(val_acc)

    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_proba)

    # Store
    metrics['accuracy'].append(acc)
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f1'].append(f1)
    metrics['balanced_acc'].append(bal_acc)
    metrics['mcc'].append(mcc)
    metrics['auc'].append(auc_score)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    print(f"\nFold {fold}:")
    print(f"  Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {auc_score:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Store all predictions for overall report
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)

# Calculate average metrics
print("\n" + "="*70)
print("AVERAGE METRICS (Mean ± Std)")
print("="*70)
for metric_name, values in metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric_name.capitalize():15s}: {mean_val:.4f} ± {std_val:.4f}")

# Calculate overfitting status
avg_train_acc = np.mean(train_scores)
avg_val_acc = np.mean(val_scores)
overfitting_gap = avg_train_acc - avg_val_acc

if overfitting_gap > 0.1:
    overfitting_status = "HIGH - Model is significantly overfitting"
    status_symbol = "⚠️"
elif overfitting_gap > 0.05:
    overfitting_status = "MODERATE - Some overfitting detected"
    status_symbol = "⚡"
elif overfitting_gap > 0.02:
    overfitting_status = "LOW - Mild overfitting"
    status_symbol = "📊"
else:
    overfitting_status = "NONE - Model generalizes well"
    status_symbol = "✅"

# Overall classification report
print("\n" + "="*70)
print("FINAL CLASSIFICATION REPORT (Aggregated over all folds)")
print("="*70)
print(classification_report(all_y_true, all_y_pred, target_names=['Non-GDM', 'GDM']))

# Overall confusion matrix
cm_total = confusion_matrix(all_y_true, all_y_pred)
tn, fp, fn, tp = cm_total.ravel()
total_acc = accuracy_score(all_y_true, all_y_pred)
print(f"\nOverall Confusion Matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")
print(f"  Accuracy: {total_acc:.4f}")

# Overfitting Status
print(f"\n" + "="*70)
print("OVERFITTING STATUS")
print("="*70)
print(f"  Average Training Accuracy   : {avg_train_acc:.4f}")
print(f"  Average Validation Accuracy : {avg_val_acc:.4f}")
print(f"  Overfitting Gap             : {overfitting_gap:.4f}")
print(f"  Status: {status_symbol} {overfitting_status}")

# Note: KNN doesn't have built-in feature importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE (KNN with SMOTE)")
print("="*70)
print("  Note: KNN is a distance-based algorithm and does not provide")
print("  built-in feature importance scores. Feature importance can be")
print("  estimated using permutation importance or coefficient of variation.")
print("  For KNN, all features contribute equally when scaled properly.")

# ============== PLOTTING (SEPARATE CHARTS) ==============
print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

print(f"\n📊 Performance Summary (Mean ± Std over 5 folds):")
print(f"   • Accuracy      : {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"   • Precision     : {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
print(f"   • Recall        : {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
print(f"   • F1-Score      : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
print(f"   • Balanced Acc  : {np.mean(metrics['balanced_acc']):.4f} ± {np.std(metrics['balanced_acc']):.4f}")
print(f"   • MCC           : {np.mean(metrics['mcc']):.4f} ± {np.std(metrics['mcc']):.4f}")
print(f"   • AUC           : {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")

print(f"\n🎯 Overfitting Analysis:")
print(f"   Training Accuracy   : {avg_train_acc:.4f}")
print(f"   Validation Accuracy : {avg_val_acc:.4f}")
print(f"   Gap                 : {overfitting_gap:.4f}")
print(f"   Status              : {overfitting_status}")

# ============== SAVE SEPARATE CHARTS ==============

# Chart 1: Per-fold performance line chart
plt.figure(figsize=(10, 6))
folds = range(1, 6)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    plt.plot(folds, metrics[metric], marker='o', linewidth=2, markersize=8, label=metric.capitalize())
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('KNN with SMOTE - Per-Fold Performance Metrics', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(folds)
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig('knn_smote_1_per_fold_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: knn_smote_1_per_fold_performance.png")

# Chart 2: Average metrics bar chart with error bars
plt.figure(figsize=(12, 6))
avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
std_metrics = {k: np.std(v) for k, v in metrics.items()}
metric_names = list(avg_metrics.keys())
avg_values = list(avg_metrics.values())
std_values = list(std_metrics.values())

bars = plt.bar(metric_names, avg_values, yerr=std_values, capsize=8,
               color='steelblue', edgecolor='black', alpha=0.8)
plt.ylabel('Score', fontsize=12)
plt.title('KNN with SMOTE - Average Metrics with Standard Deviation (5-fold CV)', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, avg_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('knn_smote_2_average_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: knn_smote_2_average_metrics.png")

# Chart 3: Confusion matrix heatmap
plt.figure(figsize=(8, 7))
cm_display = confusion_matrix(all_y_true, all_y_pred)
im = plt.imshow(cm_display, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
plt.title(f'KNN with SMOTE - Confusion Matrix\nAccuracy: {total_acc:.3f}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks([0, 1], ['Non-GDM', 'GDM'])
plt.yticks([0, 1], ['Non-GDM', 'GDM'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm_display[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_smote_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: knn_smote_3_confusion_matrix.png")

# Chart 4: ROC curve with AUC
plt.figure(figsize=(8, 7))
fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'KNN with SMOTE (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('KNN with SMOTE - ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_smote_4_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: knn_smote_4_roc_curve.png")

# Chart 5: Precision-Recall curve
plt.figure(figsize=(8, 7))
precision_curve, recall_curve, _ = precision_recall_curve(all_y_true, all_y_proba)
avg_precision = average_precision_score(all_y_true, all_y_proba)
plt.plot(recall_curve, precision_curve, color='green', lw=2,
         label=f'KNN with SMOTE (AP = {avg_precision:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('KNN with SMOTE - Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('knn_smote_5_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: knn_smote_5_precision_recall_curve.png")

# Chart 6: Training vs Validation Accuracy (Overfitting plot)
plt.figure(figsize=(8, 6))
plt.plot(folds, train_scores, marker='o', linewidth=2, markersize=8, label='Training Accuracy', color='blue')
plt.plot(folds, val_scores, marker='s', linewidth=2, markersize=8, label='Validation Accuracy', color='red')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'KNN with SMOTE - Training vs Validation\nOverfitting Gap: {overfitting_gap:.4f}', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(folds)
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.savefig('knn_smote_6_overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: knn_smote_6_overfitting_analysis.png")

print("\n" + "="*70)
print("✅ ALL CHARTS SAVED SUCCESSFULLY!")
print("="*70)
print("\n📁 Generated files:")
print("   1. knn_smote_1_per_fold_performance.png - Per-fold metrics line chart")
print("   2. knn_smote_2_average_metrics.png - Average metrics bar chart with error bars")
print("   3. knn_smote_3_confusion_matrix.png - Confusion matrix heatmap")
print("   4. knn_smote_4_roc_curve.png - ROC curve with AUC score")
print("   5. knn_smote_5_precision_recall_curve.png - Precision-Recall curve")
print("   6. knn_smote_6_overfitting_analysis.png - Training vs Validation accuracy plot")

print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)
print("• Algorithm: K-Nearest Neighbors (KNN)")
print("• Class Balancing: SMOTE (Synthetic Minority Over-sampling Technique)")
print("• n_neighbors: 5")
print("• weights: uniform")
print("• metric: Euclidean (p=2)")
print("• algorithm: auto")
print("• Scaling: StandardScaler (REQUIRED for KNN)")
print("• Cross-Validation: 5-Fold Stratified")