# # ============================================
# # LOGISTIC REGRESSION - Complete Analysis
# # GDM Prediction with 6 Charts
# # No Class Balancing, 5-Fold Stratified CV
# # ============================================
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, roc_curve, auc, confusion_matrix,
#                              classification_report, balanced_accuracy_score,
#                              matthews_corrcoef, precision_recall_curve,
#                              average_precision_score)
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # ============================================
# # 1. LOAD AND PREPARE DATA
# # ============================================
# print("Loading data...")
# df = pd.read_excel('Gestational Diabetic Dat Set (3).xlsx', sheet_name='GDM-Final2022')
#
# # Columns to drop as specified
# columns_to_drop = [
#     "Case Number",
#     "HDL",
#     "Hemoglobin",
#     "OGTT",
#     "Prediabetes",
#     "Sys BP",
#     "Dia BP",
#     "Gestation in previous Pregnancy",
#     "unexplained prenetal loss"  # Note: your file has "unexplained prenetal loss" - check spelling
# ]
#
# # Target column
# target_column = "Class Label(GDM /Non GDM)"
#
# # Get feature columns (all columns not in drop list and not the target)
# feature_columns = [col for col in df.columns if col not in columns_to_drop and col != target_column]
#
# X = df[feature_columns]
# y = df[target_column]
#
# # Convert to numeric and handle missing values
# X = X.apply(pd.to_numeric, errors='coerce')
# X = X.fillna(X.mean())
#
# print(f"\n{'=' * 70}")
# print(f"DATA SUMMARY")
# print(f"{'=' * 70}")
# print(f"Total samples: {len(X)}")
# print(f"Features used: {len(feature_columns)}")
# print(f"Feature names: {list(feature_columns)}")
# print(f"\nClass distribution:")
# print(f"  Class 0 (Non-GDM): {sum(y == 0)} ({sum(y == 0) / len(y) * 100:.1f}%)")
# print(f"  Class 1 (GDM): {sum(y == 1)} ({sum(y == 1) / len(y) * 100:.1f}%)")
#
# # Scale features (required for Logistic Regression)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # ============================================
# # 2. 5-FOLD STRATIFIED CROSS-VALIDATION
# # ============================================
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# # Store metrics for each fold
# fold_metrics = {
#     'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
#     'balanced_acc': [], 'mcc': [], 'auc': [], 'avg_precision': []
# }
# fold_probas = []
# all_y_test = []
# all_y_pred = []
# all_y_proba = []
#
# print(f"\n{'=' * 70}")
# print("LOGISTIC REGRESSION - 5-FOLD STRATIFIED CROSS-VALIDATION")
# print(f"{'=' * 70}")
#
# for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), 1):
#     X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#     # Train model
#     model = LogisticRegression(max_iter=1000, class_weight=None, random_state=42)
#     model.fit(X_train, y_train)
#
#     # Predictions
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#
#     # Store for aggregate
#     all_y_test.extend(y_test)
#     all_y_pred.extend(y_pred)
#     all_y_proba.extend(y_proba)
#
#     # Calculate metrics
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, zero_division=0)
#     rec = recall_score(y_test, y_pred, zero_division=0)
#     f1 = f1_score(y_test, y_pred, zero_division=0)
#     bal_acc = balanced_accuracy_score(y_test, y_pred)
#     mcc = matthews_corrcoef(y_test, y_pred)
#     roc_auc = auc(*roc_curve(y_test, y_proba)[:2])
#     avg_prec = average_precision_score(y_test, y_proba)
#
#     fold_metrics['accuracy'].append(acc)
#     fold_metrics['precision'].append(prec)
#     fold_metrics['recall'].append(rec)
#     fold_metrics['f1'].append(f1)
#     fold_metrics['balanced_acc'].append(bal_acc)
#     fold_metrics['mcc'].append(mcc)
#     fold_metrics['auc'].append(roc_auc)
#     fold_metrics['avg_precision'].append(avg_prec)
#
#     # Store curve data
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
#     fold_probas.append((fpr, tpr, roc_auc, prec_curve, rec_curve, avg_prec))
#
#     print(f"\nFold {fold}:")
#     print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {roc_auc:.4f}")
#
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
#
# # Train final model on all data for feature importance
# final_model = LogisticRegression(max_iter=1000, class_weight=None, random_state=42)
# final_model.fit(X_scaled, y)
#
# # ============================================
# # 3. AVERAGE METRICS
# # ============================================
# print(f"\n{'=' * 70}")
# print("AVERAGE METRICS (Mean ± Std)")
# print(f"{'=' * 70}")
# avg_metrics = {}
# for metric in fold_metrics:
#     mean_val = np.mean(fold_metrics[metric])
#     std_val = np.std(fold_metrics[metric])
#     avg_metrics[metric] = (mean_val, std_val)
#     print(f"{metric.capitalize():14s}: {mean_val:.4f} ± {std_val:.4f}")
#
# # ============================================
# # 4. FINAL CLASSIFICATION REPORT
# # ============================================
# print(f"\n{'=' * 70}")
# print("FINAL CLASSIFICATION REPORT (Aggregated over all folds)")
# print(f"{'=' * 70}")
# print(classification_report(all_y_test, all_y_pred, target_names=['Non-GDM', 'GDM']))
#
# # Aggregate confusion matrix
# cm = confusion_matrix(all_y_test, all_y_pred)
# tn, fp, fn, tp = cm.ravel()
# print(f"\nOverall Confusion Matrix:")
# print(f"  TN={tn}, FP={fp}")
# print(f"  FN={fn}, TP={tp}")
# print(f"  Accuracy: {(tn + tp) / (tn + tp + fn + fp):.4f}")
#
# # ============================================
# # CHART 1: PER-FOLD PERFORMANCE (Line Chart)
# # ============================================
# fig, ax = plt.subplots(figsize=(12, 7))
# folds = range(1, 6)
#
# ax.plot(folds, fold_metrics['accuracy'], 'b-o', label='Accuracy', linewidth=2, markersize=8, markeredgecolor='black')
# ax.plot(folds, fold_metrics['precision'], 'orange', marker='s', label='Precision', linewidth=2, markersize=8,
#         markeredgecolor='black')
# ax.plot(folds, fold_metrics['recall'], 'r-^', label='Recall', linewidth=2, markersize=8, markeredgecolor='black')
# ax.plot(folds, fold_metrics['f1'], 'g-d', label='F1-Score', linewidth=2, markersize=8, markeredgecolor='black')
#
# ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
# ax.set_ylabel('Score', fontsize=12, fontweight='bold')
# ax.set_title('Logistic Regression - Per-Fold Performance\n(5-Fold Stratified CV, No Class Balancing)',
#              fontsize=14, fontweight='bold')
# ax.legend(loc='best', fontsize=10)
# ax.grid(True, alpha=0.3, linestyle='--')
# ax.set_xticks(folds)
# ax.set_ylim([0, 1])
# ax.set_xlim([0.8, 5.2])
#
# plt.tight_layout()
# plt.savefig('logreg_1_per_fold_performance.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================
# # CHART 2: AVERAGE METRICS (Bar Chart)
# # ============================================
# fig, ax = plt.subplots(figsize=(10, 6))
#
# metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Balanced Acc', 'MCC', 'AUC', 'Avg Precision']
# metrics_values = [avg_metrics['accuracy'][0], avg_metrics['precision'][0], avg_metrics['recall'][0],
#                   avg_metrics['f1'][0], avg_metrics['balanced_acc'][0], avg_metrics['mcc'][0],
#                   avg_metrics['auc'][0], avg_metrics['avg_precision'][0]]
# metrics_stds = [avg_metrics['accuracy'][1], avg_metrics['precision'][1], avg_metrics['recall'][1],
#                 avg_metrics['f1'][1], avg_metrics['balanced_acc'][1], avg_metrics['mcc'][1],
#                 avg_metrics['auc'][1], avg_metrics['avg_precision'][1]]
#
# colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics_names)))
# bars = ax.bar(metrics_names, metrics_values, yerr=metrics_stds, capsize=5, color=colors,
#               edgecolor='black', linewidth=1.5)
#
# ax.set_ylabel('Score', fontsize=12, fontweight='bold')
# ax.set_title('Logistic Regression - Average Metrics Across 5 Folds\n(Mean ± Std Deviation)',
#              fontsize=14, fontweight='bold')
# ax.set_ylim([0, 1])
# ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
# ax.legend()
# ax.grid(True, alpha=0.3, axis='y')
#
# # Add value labels
# for bar, val, std in zip(bars, metrics_values, metrics_stds):
#     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
#             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
#
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('logreg_2_average_metrics.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================
# # CHART 3: CONFUSION MATRIX (Heatmap)
# # ============================================
# fig, ax = plt.subplots(figsize=(8, 6))
#
# cm_matrix = confusion_matrix(all_y_test, all_y_pred)
# cm_percent = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis] * 100
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Non-GDM', 'GDM'],
#             yticklabels=['Non-GDM', 'GDM'],
#             cbar_kws={'label': 'Count'},
#             ax=ax, annot_kws={'size': 14})
#
# # Add percentage labels
# for i in range(2):
#     for j in range(2):
#         ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
#                 ha='center', va='center', fontsize=10, color='black')
#
# ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
# ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
# ax.set_title(f'Logistic Regression - Confusion Matrix (Aggregated)\nTotal Samples: {len(all_y_test)}',
#              fontsize=14, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('logreg_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================
# # CHART 4: ROC CURVE (All Folds + Mean)
# # ============================================
# fig, ax = plt.subplots(figsize=(8, 8))
#
# mean_fpr = np.linspace(0, 1, 100)
# all_tprs = []
#
# for i, (fpr, tpr, roc_auc, _, _, _) in enumerate(fold_probas):
#     ax.plot(fpr, tpr, lw=2, alpha=0.6, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')
#     interp_tpr = np.interp(mean_fpr, fpr, tpr)
#     interp_tpr[0] = 0.0
#     all_tprs.append(interp_tpr)
#
# mean_tpr = np.mean(all_tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# ax.plot(mean_fpr, mean_tpr, 'k-', lw=3, label=f'Mean ROC (AUC = {mean_auc:.3f}', color='darkblue')
# ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)', color='gray')
#
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
# ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
# ax.set_title(
#     f'Logistic Regression - ROC Curves (5-Fold CV)\nMean AUC = {mean_auc:.3f} ± {np.std(fold_metrics["auc"]):.3f}',
#     fontsize=14, fontweight='bold')
# ax.legend(loc='lower right', fontsize=9)
# ax.grid(True, alpha=0.3, linestyle='--')
#
# plt.tight_layout()
# plt.savefig('logreg_4_roc_curve.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================
# # CHART 5: PRECISION-RECALL CURVE
# # ============================================
# fig, ax = plt.subplots(figsize=(8, 8))
#
# all_avg_precisions = []
# for i, (_, _, _, prec_curve, rec_curve, avg_prec) in enumerate(fold_probas):
#     ax.plot(rec_curve, prec_curve, lw=2, alpha=0.6, label=f'Fold {i + 1} (AP = {avg_prec:.3f})')
#     all_avg_precisions.append(avg_prec)
#
# baseline = sum(y) / len(y)
# ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, label=f'Baseline (AP = {baseline:.3f})')
#
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
# ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
# ax.set_title(
#     f'Logistic Regression - Precision-Recall Curves (5-Fold CV)\nMean AP = {np.mean(all_avg_precisions):.3f} ± {np.std(all_avg_precisions):.3f}',
#     fontsize=14, fontweight='bold')
# ax.legend(loc='best', fontsize=9)
# ax.grid(True, alpha=0.3, linestyle='--')
#
# plt.tight_layout()
# plt.savefig('logreg_5_precision_recall_curve.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================
# # CHART 6: FEATURE IMPORTANCE (Top 5)
# # ============================================
# # Get absolute coefficients
# feature_names = X.columns
# coefficients = np.abs(final_model.coef_[0])
#
# # Create dataframe with feature importance
# feature_importance = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': coefficients
# })
#
# # Sort by importance (descending) and get top 5
# feature_importance_sorted = feature_importance.sort_values('Importance', ascending=False)
# top_5_features = feature_importance_sorted.head(5)
#
# # Calculate percentages for better visualization
# total_importance = feature_importance['Importance'].sum()
# top_5_features['Percentage'] = (top_5_features['Importance'] / total_importance) * 100
#
# print(f"\n{'=' * 70}")
# print("TOP 5 FEATURE IMPORTANCE (Logistic Regression Coefficients)")
# print(f"{'=' * 70}")
# for idx, row in top_5_features.iterrows():
#     print(f"  {row['Feature']:25s}: {row['Importance']:.6f} ({row['Percentage']:.1f}%)")
#
# # Create horizontal bar chart for top 5 features
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Sort ascending for horizontal bar chart (so highest is on top)
# top_5_ascending = top_5_features.sort_values('Importance', ascending=True)
#
# # Create color gradient based on importance
# colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_5_ascending)))
# bars = ax.barh(top_5_ascending['Feature'], top_5_ascending['Importance'],
#                color=colors, edgecolor='black', linewidth=1.5)
#
# # Add value labels on bars
# for bar, val, pct in zip(bars, top_5_ascending['Importance'], top_5_ascending['Percentage']):
#     ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
#             f'{val:.4f} ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')
#
# ax.set_xlabel('Coefficient Magnitude (Absolute Value)', fontsize=12, fontweight='bold')
# ax.set_ylabel('Features', fontsize=12, fontweight='bold')
# ax.set_title('Logistic Regression - Top 5 Feature Importance\n(Based on Absolute Coefficients)',
#              fontsize=14, fontweight='bold')
# ax.grid(True, alpha=0.3, axis='x')
#
# # Add interpretation note
# ax.text(0.95, -0.1, f'Note: Higher coefficient = stronger influence on GDM prediction',
#         transform=ax.transAxes, ha='right', va='bottom', fontsize=9, style='italic',
#         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#
# plt.tight_layout()
# plt.savefig('logreg_6_feature_importance.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # Optional: Create a pie chart for feature importance distribution
# fig, ax = plt.subplots(figsize=(8, 8))
#
# # Get top 5 for pie chart, group others
# other_sum = feature_importance_sorted.iloc[5:]['Importance'].sum() if len(feature_importance_sorted) > 5 else 0
#
# if other_sum > 0:
#     pie_data = list(top_5_features['Importance']) + [other_sum]
#     pie_labels = list(top_5_features['Feature']) + ['Other Features']
#     pie_percentages = list(top_5_features['Percentage']) + [other_sum / total_importance * 100]
# else:
#     pie_data = list(top_5_features['Importance'])
#     pie_labels = list(top_5_features['Feature'])
#     pie_percentages = list(top_5_features['Percentage'])
#
# colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
# wedges, texts, autotexts = ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
#                                   colors=colors_pie, startangle=90,
#                                   textprops={'fontsize': 10})
# ax.set_title('Logistic Regression - Feature Importance Distribution\n(Percentage Contribution of Top 5 Features)',
#              fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.savefig('logreg_6b_feature_importance_pie.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================
# # SUMMARY OUTPUT
# # ============================================
# print(f"\n{'=' * 70}")
# print("ANALYSIS COMPLETE - SUMMARY")
# print(f"{'=' * 70}")
# print(f"\n📊 Performance Summary (Mean ± Std over 5 folds):")
# print(f"   • Accuracy      : {avg_metrics['accuracy'][0]:.4f} ± {avg_metrics['accuracy'][1]:.4f}")
# print(f"   • Precision     : {avg_metrics['precision'][0]:.4f} ± {avg_metrics['precision'][1]:.4f}")
# print(f"   • Recall        : {avg_metrics['recall'][0]:.4f} ± {avg_metrics['recall'][1]:.4f}")
# print(f"   • F1-Score      : {avg_metrics['f1'][0]:.4f} ± {avg_metrics['f1'][1]:.4f}")
# print(f"   • Balanced Acc  : {avg_metrics['balanced_acc'][0]:.4f} ± {avg_metrics['balanced_acc'][1]:.4f}")
# print(f"   • MCC           : {avg_metrics['mcc'][0]:.4f} ± {avg_metrics['mcc'][1]:.4f}")
# print(f"   • AUC           : {avg_metrics['auc'][0]:.4f} ± {avg_metrics['auc'][1]:.4f}")
#
# print(f"\n📈 Top 5 Most Important Features (by coefficient magnitude):")
# for idx, row in top_5_features.iterrows():
#     print(f"   {idx + 1}. {row['Feature']:25s}: {row['Importance']:.4f} ({row['Percentage']:.1f}% contribution)")
#
# print(f"\n✅ All 6 charts have been saved:")
# print(f"   1. logreg_1_per_fold_performance.png - Per-fold metrics line chart")
# print(f"   2. logreg_2_average_metrics.png - Average metrics bar chart")
# print(f"   3. logreg_3_confusion_matrix.png - Confusion matrix heatmap")
# print(f"   4. logreg_4_roc_curve.png - ROC curves with AUC")
# print(f"   5. logreg_5_precision_recall_curve.png - Precision-Recall curves")
# print(f"   6. logreg_6_feature_importance.png - Top 5 feature importance (horizontal bar chart)")
# print(f"   Bonus: logreg_6b_feature_importance_pie.png - Feature importance pie chart")

"""
GDM PREDICTION - LOGISTIC REGRESSION WITH SMOTE
Complete analysis with 5-Fold Stratified Cross-Validation
All outputs in the requested format
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Create output directory
output_dir = "gdm_logreg_results"
os.makedirs(output_dir, exist_ok=True)
plt.style.use('default')

print("Loading data...")
print("")

# ==================== LOAD DATA ====================
df = pd.read_excel("Gestational Diabetic Dat Set (3).xlsx", sheet_name="GDM-Final2022")

# ==================== EXACT MATCHING FEATURES ====================
feature_cols = ['Age', 'No of Pregnancy', 'BMI', 'Family History',
                'Large Child or Birth Default', 'PCOS', 'Sedentary Lifestyle']
available_features = [col for col in feature_cols if col in df.columns]

# Create target
df['GDM_Present'] = df['Class Label(GDM /Non GDM)'].apply(lambda x: 1 if x == 1 else 0)

X = df[available_features].copy()
y = df['GDM_Present'].copy()

# Handle missing values
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        X[col].fillna(X[col].median(), inplace=True)

# ==================== DATA SUMMARY ====================
print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"Total samples: {len(df)}")
print(f"Features used: {len(available_features)}")
print(f"Feature names: {available_features}")
print()
print("Class distribution:")
gdm_count = y.sum()
non_gdm_count = len(y) - gdm_count
print(f"  Class 0 (Non-GDM): {non_gdm_count} ({non_gdm_count / len(y) * 100:.1f}%)")
print(f"  Class 1 (GDM): {gdm_count} ({gdm_count / len(y) * 100:.1f}%)")

# ==================== 5-FOLD STRATIFIED CROSS-VALIDATION WITH SMOTE ====================
print()
print("=" * 70)
print("LOGISTIC REGRESSION WITH SMOTE - 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)
print()

# Store metrics for each fold
metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'balanced_acc': [], 'mcc': [], 'auc': [], 'avg_precision': []
}

# Store confusion matrices
all_y_true = []
all_y_pred = []
all_y_proba = []

# 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_num = 1
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_res)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    # Store
    metrics['accuracy'].append(acc)
    metrics['precision'].append(prec)
    metrics['recall'].append(rec)
    metrics['f1'].append(f1)
    metrics['balanced_acc'].append(bal_acc)
    metrics['mcc'].append(mcc)
    metrics['auc'].append(auc)
    metrics['avg_precision'].append(avg_prec)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Store for aggregated report
    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_y_proba.extend(y_proba.tolist())

    print(f"Fold {fold_num}:")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()

    fold_num += 1

# ==================== AVERAGE METRICS ====================
print("=" * 70)
print("AVERAGE METRICS (Mean ± Std)")
print("=" * 70)
print(f"Accuracy      : {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"Precision     : {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
print(f"Recall        : {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
print(f"F1            : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
print(f"Balanced_acc  : {np.mean(metrics['balanced_acc']):.4f} ± {np.std(metrics['balanced_acc']):.4f}")
print(f"Mcc           : {np.mean(metrics['mcc']):.4f} ± {np.std(metrics['mcc']):.4f}")
print(f"Auc           : {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
print(f"Avg_precision : {np.mean(metrics['avg_precision']):.4f} ± {np.std(metrics['avg_precision']):.4f}")

# ==================== FINAL CLASSIFICATION REPORT ====================
print()
print("=" * 70)
print("FINAL CLASSIFICATION REPORT (Aggregated over all folds)")
print("=" * 70)
print(classification_report(all_y_true, all_y_pred, target_names=['Non-GDM', 'GDM']))

# Overall confusion matrix
tn_total, fp_total, fn_total, tp_total = confusion_matrix(all_y_true, all_y_pred).ravel()
overall_acc = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)

print()
print("Overall Confusion Matrix:")
print(f"  TN={tn_total}, FP={fp_total}")
print(f"  FN={fn_total}, TP={tp_total}")
print(f"  Accuracy: {overall_acc:.4f}")

# ==================== FEATURE IMPORTANCE (Trained on full data with SMOTE) ====================
print()
print("=" * 70)
print("TOP FEATURE IMPORTANCE (Logistic Regression Coefficients)")
print("=" * 70)

# Train on full data with SMOTE to get feature importance
smote_full = SMOTE(random_state=42)
X_res_full, y_res_full = smote_full.fit_resample(X, y)

scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_res_full)

model_full = LogisticRegression(random_state=42, max_iter=1000)
model_full.fit(X_scaled_full, y_res_full)

# Calculate feature importance percentages
coef_abs = np.abs(model_full.coef_[0])
coef_sum = np.sum(coef_abs)
coef_df = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': model_full.coef_[0],
    'Abs_Coefficient': coef_abs,
    'Percentage': (coef_abs / coef_sum) * 100
}).sort_values('Abs_Coefficient', ascending=False)

for i, row in coef_df.iterrows():
    feature_idx = available_features.index(row['Feature']) + 1
    print(f"  {feature_idx}. {row['Feature']:<30} : {row['Coefficient']:.6f} ({row['Percentage']:.1f}%)")

# ==================== PLOT 1: PER-FOLD PERFORMANCE TABLE ====================
# ==================== PLOT 1: PER-FOLD PERFORMANCE LINE CHART ====================
fig, ax = plt.subplots(figsize=(10, 6))

folds = [1, 2, 3, 4, 5]

# Metrics to plot
metrics_to_plot = {
    'Accuracy': metrics['accuracy'],
    'Precision': metrics['precision'],
    'Recall': metrics['recall'],
    'F1-Score': metrics['f1']
}

colors = {'Accuracy': '#1E88E5', 'Precision': '#43A047',
          'Recall': '#FB8C00', 'F1-Score': '#E53935'}
markers = {'Accuracy': 'o', 'Precision': 's', 'Recall': '^', 'F1-Score': 'D'}

# Plot each metric
for metric_name, values in metrics_to_plot.items():
    ax.plot(folds, values, marker=markers[metric_name], linewidth=2,
            markersize=8, label=metric_name, color=colors[metric_name])

# Add value labels on each point
for metric_name, values in metrics_to_plot.items():
    for i, (fold, val) in enumerate(zip(folds, values)):
        ax.annotate(f'{val:.2f}', (fold, val),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9, fontweight='bold')

# Customize chart
ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Fold Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(folds)
ax.set_xticklabels([f'Fold {i}' for i in folds])
ax.set_ylim([0.65, 1.0])
ax.set_yticks(np.arange(0.65, 1.01, 0.05))
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=10)

# Add horizontal line at 0.80 for reference
ax.axhline(y=0.80, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_1_per_fold_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_1_per_fold_performance.png")
# ==================== PLOT 2: AVERAGE METRICS BAR CHART ====================
fig, ax = plt.subplots(figsize=(12, 6))
avg_metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_acc', 'mcc', 'auc']
avg_values = [np.mean(metrics[m]) for m in avg_metrics]
std_values = [np.std(metrics[m]) for m in avg_metrics]

bars = ax.bar(avg_metrics, avg_values, yerr=std_values, capsize=5,
              color='steelblue', edgecolor='black', alpha=0.8)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Average Performance Metrics (Mean ± Std over 5 folds)', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, avg_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_2_average_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_2_average_metrics.png")

# ==================== PLOT 3: CONFUSION MATRIX ====================
fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-GDM', 'GDM'], yticklabels=['Non-GDM', 'GDM'],
            annot_kws={'size': 16})
ax.set_title(f'Confusion Matrix - Overall (Accuracy: {overall_acc:.4f})', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_3_confusion_matrix.png")

# ==================== PLOT 4: ROC CURVE ====================
fig, ax = plt.subplots(figsize=(8, 7))

# Compute ROC for each fold and plot
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

# Reset fold counter for ROC
skf_roc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf_roc.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_res)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    aucs.append(auc_val)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    ax.plot(fpr, tpr, alpha=0.3, color='blue', linewidth=1)

# Plot mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, 'b-', linewidth=2, label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
ax.fill_between(mean_fpr, mean_tpr - np.std(tprs, axis=0), mean_tpr + np.std(tprs, axis=0),
                alpha=0.2, color='blue', label='± 1 std. dev.')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - 5-Fold Cross-Validation', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_4_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_4_roc_curve.png")

# ==================== PLOT 5: PRECISION-RECALL CURVE ====================
fig, ax = plt.subplots(figsize=(8, 7))

# Compute PR curve for each fold
precisions = []
recalls = []
avg_precisions = []

for train_idx, test_idx in skf_roc.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_res)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    precisions.append(precision)
    recalls.append(recall)
    avg_precisions.append(avg_prec)

    ax.plot(recall, precision, alpha=0.3, color='green', linewidth=1)

# Interpolate for mean curve
mean_ap = np.mean(avg_precisions)
std_ap = np.std(avg_precisions)

# Plot baseline
ax.axhline(y=mean_ap, color='red', linestyle='--', linewidth=1,
           label=f'Baseline (AP = {mean_ap:.4f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - 5-Fold CV', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_5_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_5_precision_recall_curve.png")

# ==================== PLOT 6: FEATURE IMPORTANCE ====================
fig, ax = plt.subplots(figsize=(10, 7))

# Sort for horizontal bar chart
coef_sorted = coef_df.sort_values('Coefficient', ascending=True)
colors = ['red' if c < 0 else 'green' for c in coef_sorted['Coefficient']]

bars = ax.barh(coef_sorted['Feature'], coef_sorted['Coefficient'], color=colors)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature Importance - Logistic Regression Coefficients\n(Red=Risk Factor, Green=Protective)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, coef_sorted['Coefficient']):
    x_pos = val + (0.02 if val > 0 else -0.08)
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_6_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_6_feature_importance.png")

# ==================== BONUS: FEATURE IMPORTANCE PIE CHART ====================
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(coef_df['Percentage'], labels=coef_df['Feature'],
                                  autopct='%1.1f%%', startangle=90,
                                  colors=plt.cm.Set3(np.linspace(0, 1, len(coef_df))))
ax.set_title('Feature Importance Distribution (by Coefficient Magnitude)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/logreg_6b_feature_importance_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_dir}/logreg_6b_feature_importance_pie.png")

# ==================== FINAL SUMMARY ====================
print()
print("=" * 70)
print("ANALYSIS COMPLETE - SUMMARY")
print("=" * 70)
print()
print("📊 Performance Summary (Mean ± Std over 5 folds):")
print(f"   • Accuracy      : {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"   • Precision     : {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
print(f"   • Recall        : {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
print(f"   • F1-Score      : {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
print(f"   • Balanced Acc  : {np.mean(metrics['balanced_acc']):.4f} ± {np.std(metrics['balanced_acc']):.4f}")
print(f"   • MCC           : {np.mean(metrics['mcc']):.4f} ± {np.std(metrics['mcc']):.4f}")
print(f"   • AUC           : {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
print()
print("📈 Top 5 Most Important Features (by coefficient magnitude):")
for i, row in coef_df.head(5).iterrows():
    feature_idx = available_features.index(row['Feature']) + 1
    print(f"   {feature_idx}. {row['Feature']:<30} : {row['Coefficient']:.4f} ({row['Percentage']:.1f}% contribution)")
print()
print("✅ All 6 charts have been saved:")
print("   1. logreg_1_per_fold_performance.png - Per-fold metrics line chart")
print("   2. logreg_2_average_metrics.png - Average metrics bar chart")
print("   3. logreg_3_confusion_matrix.png - Confusion matrix heatmap")
print("   4. logreg_4_roc_curve.png - ROC curves with AUC")
print("   5. logreg_5_precision_recall_curve.png - Precision-Recall curves")
print("   6. logreg_6_feature_importance.png - Feature importance (horizontal bar chart)")
print("   Bonus: logreg_6b_feature_importance_pie.png - Feature importance pie chart")
print()
print("=" * 70)
print("LOGISTIC REGRESSION WITH SMOTE COMPLETED SUCCESSFULLY!")
print("=" * 70)















