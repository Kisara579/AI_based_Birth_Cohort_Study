# """
# XGBoost for IUGR Prediction - WITH BUILT-IN CLASS BALANCING (scale_pos_weight)
# No SMOTE - uses scale_pos_weight parameter for handling imbalance
# """
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
#                              balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
#                              roc_curve, precision_recall_curve, confusion_matrix,
#                              classification_report)
# from sklearn.utils import resample
# import xgboost as xgb
# import warnings
# import os
#
# warnings.filterwarnings('ignore')
#
# # Create output directory
# output_dir = 'iugr_xgboost_balanced'
# os.makedirs(output_dir, exist_ok=True)
#
# # Set random seed
# RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)
#
# print("=" * 70)
# print("XGBOOST WITH BUILT-IN CLASS BALANCING (scale_pos_weight)")
# print("=" * 70)
#
# # ============================================================================
# # STEP 1: LOAD AND PREPARE DATA
# # ============================================================================
# print("\n" + "=" * 70)
# print("STEP 1: LOADING AND PREPARING DATA")
# print("=" * 70)
#
# # Load data
# df = pd.read_csv('birth_weight_dataset.csv')
# print(f"\n✓ Loaded {len(df)} samples")
#
# # Define features
# feature_columns = [
#     'age',  # Maternal_Age
#     'has_diabetes',  # Pre_existing_diabetes
#     'household_income',  # Income_Level
#     'education_level',  # Education_Level
#     'number_of_prenatal_visits',  # Num_Prenatal_Visits
#     'pre_pregnancy_bmi',  # BMI_at_booking
#     'smoking_status',  # Smoking_Status
#     'alcohol_consumption'  # Alcohol_Consumption
# ]
#
# target_column = 'birth_weight_category'
#
# # Prepare features
# X = df[feature_columns].copy()
# y = (df[target_column] == 'Low').astype(int)  # 1 = IUGR/Low, 0 = Normal
#
# # Encode categorical variables
# X['smoking_status'] = (X['smoking_status'] == 'Yes').astype(int)
# X['alcohol_consumption'] = (X['alcohol_consumption'] == 'Yes').astype(int)
# education_map = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}
# X['education_level'] = X['education_level'].map(education_map)
#
# feature_names = X.columns.tolist()
# print(f"\nFeatures used ({len(feature_names)}): {feature_names}")
#
# # Class distribution
# class_counts = y.value_counts()
# print(f"\nClass distribution:")
# print(f"  Class 0 (Normal birth weight): {class_counts[0]} ({class_counts[0] / len(y) * 100:.1f}%)")
# print(f"  Class 1 (Low birth weight/IUGR): {class_counts[1]} ({class_counts[1] / len(y) * 100:.1f}%)")
#
# # Calculate scale_pos_weight
# scale_pos_weight = class_counts[0] / class_counts[1]
# print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")
#
#
# # ============================================================================
# # STEP 2: BOOTSTRAP FUNCTION FOR CONFIDENCE INTERVALS
# # ============================================================================
#
# def bootstrap_evaluation_balanced(model, X, y, n_bootstrap=500):
#     """Bootstrap with built-in balancing and 95% confidence intervals"""
#     aucs, accs, precs, recs, f1s = [], [], [], [], []
#
#     for i in range(n_bootstrap):
#         # Bootstrap sampling
#         indices = np.arange(len(X))
#         boot_idx = resample(indices, n_samples=len(X), replace=True)
#         oob_idx = np.setdiff1d(indices, boot_idx)
#
#         if len(oob_idx) < 20 or len(np.unique(y.iloc[oob_idx])) < 2:
#             continue
#
#         X_train = X.iloc[boot_idx]
#         y_train = y.iloc[boot_idx]
#         X_test = X.iloc[oob_idx]
#         y_test = y.iloc[oob_idx]
#
#         # Scale features (fit ONLY on training data to prevent leakage)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#         # Calculate class weights for this bootstrap sample
#         pos_count = sum(y_train)
#         neg_count = len(y_train) - pos_count
#         spw = neg_count / pos_count if pos_count > 0 else 1
#
#         # Train model with built-in balancing
#         model_clone = xgb.XGBClassifier(
#             n_estimators=50,
#             max_depth=3,
#             learning_rate=0.05,
#             scale_pos_weight=spw,
#             random_state=RANDOM_STATE,
#             use_label_encoder=False,
#             eval_metric='logloss'
#         )
#         model_clone.fit(X_train_scaled, y_train)
#
#         # Predict
#         y_pred = model_clone.predict(X_test_scaled)
#         y_pred_proba = model_clone.predict_proba(X_test_scaled)[:, 1]
#
#         # Metrics
#         accs.append(accuracy_score(y_test, y_pred))
#         precs.append(precision_score(y_test, y_pred, zero_division=0))
#         recs.append(recall_score(y_test, y_pred, zero_division=0))
#         f1s.append(f1_score(y_test, y_pred, zero_division=0))
#         aucs.append(roc_auc_score(y_test, y_pred_proba))
#
#     return {
#         'accuracy': (np.mean(accs), np.percentile(accs, 2.5), np.percentile(accs, 97.5)),
#         'precision': (np.mean(precs), np.percentile(precs, 2.5), np.percentile(precs, 97.5)),
#         'recall': (np.mean(recs), np.percentile(recs, 2.5), np.percentile(recs, 97.5)),
#         'f1': (np.mean(f1s), np.percentile(f1s, 2.5), np.percentile(f1s, 97.5)),
#         'auc': (np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
#     }
#
#
# # ============================================================================
# # STEP 3: CROSS-VALIDATION WITH BUILT-IN BALANCING (NO LEAKAGE)
# # ============================================================================
#
# print("\n" + "=" * 70)
# print("STEP 2: STRATIFIED 5-FOLD CROSS-VALIDATION")
# print("=" * 70)
#
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#
# fold_metrics = {
#     'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
#     'balanced_acc': [], 'mcc': [], 'auc': []
# }
#
# fold_confusion_matrices = []
# all_y_true = []
# all_y_pred = []
# all_y_pred_proba = []
#
# print("\n")
#
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     # Split data
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#
#     # Scale (fit ONLY on training data to prevent leakage)
#     scaler_fold = StandardScaler()
#     X_train_scaled = scaler_fold.fit_transform(X_train)
#     X_val_scaled = scaler_fold.transform(X_val)
#
#     # Calculate scale_pos_weight for this fold
#     pos_count = sum(y_train)
#     neg_count = len(y_train) - pos_count
#     spw = neg_count / pos_count if pos_count > 0 else 1
#
#     # Train XGBoost with built-in class balancing
#     xgb_model = xgb.XGBClassifier(
#         n_estimators=50,
#         max_depth=3,
#         learning_rate=0.05,
#         scale_pos_weight=spw,
#         random_state=RANDOM_STATE,
#         use_label_encoder=False,
#         eval_metric='logloss'
#     )
#     xgb_model.fit(X_train_scaled, y_train)
#
#     # Predict
#     y_pred = xgb_model.predict(X_val_scaled)
#     y_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
#
#     # Store for aggregation
#     all_y_true.extend(y_val)
#     all_y_pred.extend(y_pred)
#     all_y_pred_proba.extend(y_pred_proba)
#
#     # Calculate metrics
#     acc = accuracy_score(y_val, y_pred)
#     prec = precision_score(y_val, y_pred, zero_division=0)
#     rec = recall_score(y_val, y_pred, zero_division=0)
#     f1 = f1_score(y_val, y_pred, zero_division=0)
#     bal_acc = balanced_accuracy_score(y_val, y_pred)
#     mcc = matthews_corrcoef(y_val, y_pred)
#     auc = roc_auc_score(y_val, y_pred_proba)
#     cm = confusion_matrix(y_val, y_pred)
#
#     fold_metrics['accuracy'].append(acc)
#     fold_metrics['precision'].append(prec)
#     fold_metrics['recall'].append(rec)
#     fold_metrics['f1'].append(f1)
#     fold_metrics['balanced_acc'].append(bal_acc)
#     fold_metrics['mcc'].append(mcc)
#     fold_metrics['auc'].append(auc)
#     fold_confusion_matrices.append(cm)
#
#     print(f"Fold {fold}:")
#     print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
#     print(f"  scale_pos_weight: {spw:.2f}")
#     print(f"  Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")
#     print()
#
# # ============================================================================
# # STEP 4: BOOTSTRAP CONFIDENCE INTERVALS
# # ============================================================================
#
# print("=" * 70)
# print("STEP 3: BOOTSTRAP CONFIDENCE INTERVALS (500 iterations)")
# print("=" * 70)
#
# bootstrap_results = bootstrap_evaluation_balanced(None, X, y, n_bootstrap=500)
#
# print("\nBootstrap Results (95% Confidence Intervals):")
# print(
#     f"  Accuracy  : {bootstrap_results['accuracy'][0]:.4f} (95% CI: {bootstrap_results['accuracy'][1]:.4f}-{bootstrap_results['accuracy'][2]:.4f})")
# print(
#     f"  Precision : {bootstrap_results['precision'][0]:.4f} (95% CI: {bootstrap_results['precision'][1]:.4f}-{bootstrap_results['precision'][2]:.4f})")
# print(
#     f"  Recall    : {bootstrap_results['recall'][0]:.4f} (95% CI: {bootstrap_results['recall'][1]:.4f}-{bootstrap_results['recall'][2]:.4f})")
# print(
#     f"  F1-Score  : {bootstrap_results['f1'][0]:.4f} (95% CI: {bootstrap_results['f1'][1]:.4f}-{bootstrap_results['f1'][2]:.4f})")
# print(
#     f"  AUC       : {bootstrap_results['auc'][0]:.4f} (95% CI: {bootstrap_results['auc'][1]:.4f}-{bootstrap_results['auc'][2]:.4f})")
#
# # ============================================================================
# # STEP 5: AGGREGATED METRICS
# # ============================================================================
#
# print("\n" + "=" * 70)
# print("STEP 4: AVERAGE METRICS (Mean ± Std over 5 folds)")
# print("=" * 70)
#
# print(f"\nAccuracy      : {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
# print(f"Precision     : {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
# print(f"Recall        : {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
# print(f"F1            : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
# print(f"Balanced_acc  : {np.mean(fold_metrics['balanced_acc']):.4f} ± {np.std(fold_metrics['balanced_acc']):.4f}")
# print(f"MCC           : {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
# print(f"AUC           : {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
#
# # ============================================================================
# # STEP 6: FINAL CLASSIFICATION REPORT
# # ============================================================================
#
# print("\n" + "=" * 70)
# print("STEP 5: FINAL CLASSIFICATION REPORT (Aggregated over all folds)")
# print("=" * 70)
#
# print("\nClassification Report:")
# print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Low/IUGR']))
#
# # Overall confusion matrix
# overall_cm = confusion_matrix(all_y_true, all_y_pred)
# tn, fp, fn, tp = overall_cm.ravel()
# print(f"\nOverall Confusion Matrix:")
# print(f"  TN={tn}, FP={fp}")
# print(f"  FN={fn}, TP={tp}")
# print(f"  Overall Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
#
# # ============================================================================
# # STEP 7: FEATURE IMPORTANCE
# # ============================================================================
#
# print("\n" + "=" * 70)
# print("STEP 6: FEATURE IMPORTANCE (XGBoost)")
# print("=" * 70)
#
# # Train final model on full data with built-in balancing
# scaler_final = StandardScaler()
# X_scaled_final = scaler_final.fit_transform(X)
#
# final_model = xgb.XGBClassifier(
#     n_estimators=50,
#     max_depth=3,
#     learning_rate=0.05,
#     scale_pos_weight=scale_pos_weight,
#     random_state=RANDOM_STATE,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )
# final_model.fit(X_scaled_final, y)
#
# importance = final_model.feature_importances_
# sorted_idx = np.argsort(importance)[::-1]
#
# print("\nFeature importance (by gain):")
# for i, idx in enumerate(sorted_idx):
#     print(
#         f"  {i + 1}. {feature_names[idx]:30s} : {importance[idx]:.4f} ({importance[idx] / sum(importance) * 100:.1f}%)")
#
# # ============================================================================
# # ============================================================================
# # STEP 7: GENERATING VISUALIZATIONS
# # ============================================================================
#
# print("\n" + "=" * 70)
# print("STEP 7: GENERATING VISUALIZATIONS")
# print("=" * 70)
#
# # 1. Per-fold performance (using 3x2 grid for 6 metrics)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Changed to 2 rows, 3 columns
# metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'balanced_acc', 'auc']
# colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_to_plot)))
#
# for i, metric in enumerate(metrics_to_plot):
#     row = i // 3  # Integer division for row (0 or 1)
#     col = i % 3  # Modulo for column (0, 1, or 2)
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
# plt.suptitle('XGBoost with Built-in Balancing - Per-Fold Performance', fontsize=14)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_1_per_fold_performance.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# # 2. Average metrics bar chart
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
# fig, ax = plt.subplots(figsize=(12, 6))  # Slightly wider for 6 metrics
# bars = ax.bar(avg_metrics.keys(), avg_metrics.values(), yerr=std_metrics.values(),
#               capsize=5, color='steelblue', edgecolor='black')
# ax.set_ylabel('Score')
# ax.set_title('XGBoost with Built-in Balancing - Average Performance Metrics')
# ax.set_ylim(0, 1.1)
# ax.grid(True, alpha=0.3, axis='y')
# for bar, val in zip(bars, avg_metrics.values()):
#     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
#             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_2_average_metrics.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# # 3. Confusion matrix heatmap
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Normal', 'Low/IUGR'],
#             yticklabels=['Normal', 'Low/IUGR'], ax=ax)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('Actual')
# ax.set_title('XGBoost with Built-in Balancing - Overall Confusion Matrix')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# # 4. ROC Curve
# fig, ax = plt.subplots(figsize=(8, 6))
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#
#     scaler_fold = StandardScaler()
#     X_train_scaled = scaler_fold.fit_transform(X_train)
#     X_val_scaled = scaler_fold.transform(X_val)
#
#     pos_count = sum(y_train)
#     neg_count = len(y_train) - pos_count
#     spw = neg_count / pos_count if pos_count > 0 else 1
#
#     xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
#                                   scale_pos_weight=spw, random_state=RANDOM_STATE,
#                                   use_label_encoder=False, eval_metric='logloss')
#     xgb_model.fit(X_train_scaled, y_train)
#
#     y_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
#     fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
#     auc_val = roc_auc_score(y_val, y_pred_proba)
#     ax.plot(fpr, tpr, lw=1, alpha=0.7, label=f'Fold {fold} (AUC={auc_val:.3f})')
#
# ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.set_title('XGBoost with Built-in Balancing - ROC Curves')
# ax.legend(loc='lower right')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_4_roc_curve.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# # 5. Precision-Recall Curve
# fig, ax = plt.subplots(figsize=(8, 6))
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#
#     scaler_fold = StandardScaler()
#     X_train_scaled = scaler_fold.fit_transform(X_train)
#     X_val_scaled = scaler_fold.transform(X_val)
#
#     pos_count = sum(y_train)
#     neg_count = len(y_train) - pos_count
#     spw = neg_count / pos_count if pos_count > 0 else 1
#
#     xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
#                                   scale_pos_weight=spw, random_state=RANDOM_STATE,
#                                   use_label_encoder=False, eval_metric='logloss')
#     xgb_model.fit(X_train_scaled, y_train)
#
#     y_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
#     precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_pred_proba)
#     ax.plot(recall_vals, precision_vals, lw=1, alpha=0.7, label=f'Fold {fold}')
#
# ax.set_xlabel('Recall')
# ax.set_ylabel('Precision')
# ax.set_title('XGBoost with Built-in Balancing - Precision-Recall Curves')
# ax.legend(loc='best')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_5_precision_recall_curve.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# # 6. Feature importance bar chart
# fig, ax = plt.subplots(figsize=(10, 6))
# sorted_importance = importance[sorted_idx]
# sorted_features = [feature_names[i] for i in sorted_idx]
# bars = ax.barh(range(len(sorted_features)), sorted_importance, color='coral', edgecolor='black')
# ax.set_yticks(range(len(sorted_features)))
# ax.set_yticklabels(sorted_features)
# ax.set_xlabel('Importance Score')
# ax.set_title('XGBoost with Built-in Balancing - Feature Importance')
# ax.invert_yaxis()
# for bar, val in zip(bars, sorted_importance):
#     ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_6_feature_importance.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# # 7. Feature importance pie chart
# fig, ax = plt.subplots(figsize=(10, 8))
# colors_pie = plt.cm.Set3(np.linspace(0, 1, len(sorted_features)))
# wedges, texts, autotexts = ax.pie(sorted_importance, labels=sorted_features, autopct='%1.1f%%',
#                                   colors=colors_pie, startangle=90)
# ax.set_title('XGBoost with Built-in Balancing - Feature Importance Distribution')
# plt.tight_layout()
# plt.savefig(f'{output_dir}/xgboost_balanced_6b_feature_importance_pie.png', dpi=150, bbox_inches='tight')
# plt.close()
#
# print(f"\n✓ All 7 charts saved to '{output_dir}/'")
#
# # ============================================================================
# # SUMMARY
# # ============================================================================
#
# print("\n" + "=" * 70)
# print("ANALYSIS COMPLETE - SUMMARY")
# print("=" * 70)
#
# print(f"\n📊 Performance Summary (Mean ± Std over 5 folds):")
# print(f"   • Accuracy      : {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
# print(f"   • Precision     : {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
# print(f"   • Recall        : {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
# print(f"   • F1-Score      : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
# print(f"   • Balanced Acc  : {np.mean(fold_metrics['balanced_acc']):.4f} ± {np.std(fold_metrics['balanced_acc']):.4f}")
# print(f"   • MCC           : {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
# print(f"   • AUC           : {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
#
# print(f"\n📈 Bootstrap Validation (95% CI):")
# print(
#     f"   • AUC: {bootstrap_results['auc'][0]:.4f} ({bootstrap_results['auc'][1]:.4f}-{bootstrap_results['auc'][2]:.4f})")
# print(
#     f"   • Recall: {bootstrap_results['recall'][0]:.4f} ({bootstrap_results['recall'][1]:.4f}-{bootstrap_results['recall'][2]:.4f})")
#
# print(f"\n📋 Top 5 Most Important Features:")
# for i in range(min(5, len(sorted_features))):
#     print(
#         f"   {i + 1}. {sorted_features[i]:30s} : {sorted_importance[i]:.4f} ({sorted_importance[i] / sum(importance) * 100:.1f}%)")
#
# print(f"\n✅ All charts saved to '{output_dir}/':")
# print(f"   1. xgboost_balanced_1_per_fold_performance.png")
# print(f"   2. xgboost_balanced_2_average_metrics.png")
# print(f"   3. xgboost_balanced_3_confusion_matrix.png")
# print(f"   4. xgboost_balanced_4_roc_curve.png")
# print(f"   5. xgboost_balanced_5_precision_recall_curve.png")
# print(f"   6. xgboost_balanced_6_feature_importance.png")
# print(f"   7. xgboost_balanced_6b_feature_importance_pie.png")
#
# print("\n" + "=" * 70)
# print("XGBOOST WITH BUILT-IN CLASS BALANCING COMPLETED SUCCESSFULLY!")
# print("=" * 70)

"""
XGBoost for IUGR Prediction - WITH SMOTE (External Class Balancing)
Includes missing value handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             classification_report)
from sklearn.utils import resample
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'iugr_xgboost_smote'
os.makedirs(output_dir, exist_ok=True)

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("XGBOOST WITH SMOTE (EXTERNAL CLASS BALANCING)")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING AND PREPARING DATA")
print("="*70)

# Load data
df = pd.read_csv('birth_weight_dataset.csv')
print(f"\n✓ Loaded {len(df)} samples")

# Define features
feature_columns = [
    'age',                       # Maternal_Age
    'has_diabetes',              # Pre_existing_diabetes
    'household_income',          # Income_Level
    'education_level',           # Education_Level
    'number_of_prenatal_visits', # Num_Prenatal_Visits
    'pre_pregnancy_bmi',         # BMI_at_booking
    'smoking_status',            # Smoking_Status
    'alcohol_consumption'        # Alcohol_Consumption
]

target_column = 'birth_weight_category'

# Prepare features
X = df[feature_columns].copy()
y = (df[target_column] == 'Low').astype(int)  # 1 = IUGR/Low, 0 = Normal

# ============================================================================
# FIX: HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*70)
print("CHECKING FOR MISSING VALUES")
print("="*70)

print("\nMissing values per column:")
print(X.isnull().sum())

# Remove rows with missing values
print(f"\nRows before dropping: {len(X)}")
X = X.dropna()
y = y[X.index]  # Align y with X after dropping
print(f"Rows after dropping: {len(X)}")

# Verify no missing values remain
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
print(f"  Class 0 (Normal birth weight): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"  Class 1 (Low birth weight/IUGR): {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
print("\n" + "="*70)
print("MODEL CONFIGURATION")
print("="*70)

print("""
Model Configuration
• Algorithm: XGBoost (with SMOTE)
• SMOTE k_neighbors: 3
• SMOTE sampling_strategy: auto (balance to majority class)
• n_estimators: 50
• max_depth: 3
• learning_rate: 0.05
• min_child_weight: 3
• subsample: 0.8
• colsample_bytree: 0.8
• reg_alpha: 0.5
• reg_lambda: 1.0
• Class Balancing: SMOTE (oversampling minority class)
• Scaling: STANDARDIZATION (StandardScaler)
• Cross-Validation: 5-Fold Stratified
""")

# ============================================================================
# STEP 2: CROSS-VALIDATION WITH SMOTE (NO LEAKAGE)
# ============================================================================

print("\n" + "="*70)
print("STEP 2: SMOTE WITH STRATIFIED 5-FOLD CROSS-VALIDATION")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'balanced_acc': [], 'mcc': [], 'auc': []
}

fold_confusion_matrices = []
all_y_true = []
all_y_pred = []
all_y_pred_proba = []

print("\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Scale (fit ONLY on training data to prevent leakage)
    scaler_fold = StandardScaler()
    X_train_scaled = scaler_fold.fit_transform(X_train)
    X_val_scaled = scaler_fold.transform(X_val)

    # Apply SMOTE on training data ONLY (NO LEAKAGE)
    # Note: SMOTE now works because we removed NaN values
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = xgb_model.predict(X_val_scaled)
    y_pred_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]

    # Store for aggregation
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
    cm = confusion_matrix(y_val, y_pred)

    fold_metrics['accuracy'].append(acc)
    fold_metrics['precision'].append(prec)
    fold_metrics['recall'].append(rec)
    fold_metrics['f1'].append(f1)
    fold_metrics['balanced_acc'].append(bal_acc)
    fold_metrics['mcc'].append(mcc)
    fold_metrics['auc'].append(auc)
    fold_confusion_matrices.append(cm)

    print(f"Fold {fold}:")
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"  Balanced Acc: {bal_acc:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
    print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
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
# STEP 4: FINAL CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*70)
print("STEP 4: FINAL CLASSIFICATION REPORT (Aggregated over all folds)")
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
# STEP 5: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*70)
print("STEP 5: FEATURE IMPORTANCE (XGBoost)")
print("="*70)

# Train final model on full data with SMOTE
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)
smote_final = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
X_resampled_final, y_resampled_final = smote_final.fit_resample(X_scaled_final, y)

final_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_resampled_final, y_resampled_final)

importance = final_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]

print("\nFeature importance (by gain):")
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1}. {feature_names[idx]:30s} : {importance[idx]:.4f} ({importance[idx]/sum(importance)*100:.1f}%)")

# ============================================================================
# STEP 6: VISUALIZATIONS (Fixed 2x3 grid)
# ============================================================================

print("\n" + "="*70)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*70)

# 1. Per-fold performance (2 rows, 3 columns for 6 metrics)
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
plt.suptitle('XGBoost with SMOTE - Per-Fold Performance', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/xgboost_smote_1_per_fold_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Average metrics bar chart
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
ax.set_title('XGBoost with SMOTE - Average Performance Metrics')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, avg_metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/xgboost_smote_2_average_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Low/IUGR'],
            yticklabels=['Normal', 'Low/IUGR'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('XGBoost with SMOTE - Overall Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{output_dir}/xgboost_smote_3_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sorted_importance = importance[sorted_idx]
sorted_features = [feature_names[i] for i in sorted_idx]
bars = ax.barh(range(len(sorted_features)), sorted_importance, color='coral', edgecolor='black')
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels(sorted_features)
ax.set_xlabel('Importance Score')
ax.set_title('XGBoost with SMOTE - Feature Importance')
ax.invert_yaxis()
for bar, val in zip(bars, sorted_importance):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
plt.tight_layout()
plt.savefig(f'{output_dir}/xgboost_smote_4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Feature importance pie chart
fig, ax = plt.subplots(figsize=(10, 8))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(sorted_features)))
wedges, texts, autotexts = ax.pie(sorted_importance, labels=sorted_features, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90)
ax.set_title('XGBoost with SMOTE - Feature Importance Distribution')
plt.tight_layout()
plt.savefig(f'{output_dir}/xgboost_smote_5_feature_importance_pie.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ All charts saved to '{output_dir}/'")

# ============================================================================
# SUMMARY
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

print(f"\n📋 Top 5 Most Important Features:")
for i in range(min(5, len(sorted_features))):
    print(f"   {i+1}. {sorted_features[i]:30s} : {sorted_importance[i]:.4f} ({sorted_importance[i]/sum(importance)*100:.1f}%)")

print("\n" + "="*70)
print("XGBOOST WITH SMOTE COMPLETED SUCCESSFULLY!")
print("="*70)