import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, auc

def extract_feature_importance(pipeline, feature_order, categorical_features, top_n=None):
    """
    Extract feature importance from a sklearn Pipeline for any supported estimator.
    """
    try:
        estimator = None
        for name, step in reversed(pipeline.steps):
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                estimator = step
                break
            if hasattr(step, "calibrated_classifiers_"):
                base = step.calibrated_classifiers_[0].estimator
                if hasattr(base, "coef_") or hasattr(base, "feature_importances_"):
                    estimator = base
                    break

        if estimator is None:
            return None

        if hasattr(estimator, "feature_importances_"):
            raw_importances = np.array(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = np.array(estimator.coef_)
            raw_importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        preprocessor = None
        for name, step in pipeline.steps:
            if hasattr(step, "transformers_"):
                preprocessor = step
                break

        if preprocessor is not None:
            transformed_names = []
            for trans_name, transformer, columns in preprocessor.transformers_:
                if trans_name == "remainder":
                    if columns == "passthrough":
                        remainder_cols = [c for c in feature_order if c not in categorical_features]
                        transformed_names.extend(remainder_cols)
                    elif isinstance(columns, list):
                        transformed_names.extend(columns)
                else:
                    fitted_transformer = preprocessor.named_transformers_.get(trans_name, transformer)

                    if transformer in ("drop", "passthrough"):
                        if transformer == "passthrough":
                            if isinstance(columns, list):
                                transformed_names.extend(columns)
                            elif columns is not None:
                                transformed_names.append(str(columns))
                        continue

                    # Some transformers (e.g., OHE on empty categorical set) are intentionally
                    # left unfitted. Fall back to raw column names in that case.
                    if hasattr(fitted_transformer, "get_feature_names_out"):
                        try:
                            if isinstance(columns, list) and len(columns) == 0:
                                continue
                            out_names = fitted_transformer.get_feature_names_out(columns)
                            transformed_names.extend(list(out_names))
                            continue
                        except Exception:
                            pass

                    if isinstance(columns, list):
                        transformed_names.extend(columns)
                    elif columns is not None:
                        transformed_names.append(str(columns))
        else:
            transformed_names = list(feature_order)

        n = len(raw_importances)
        if len(transformed_names) < n:
            transformed_names.extend([f"feature_{i}" for i in range(len(transformed_names), n)])
        transformed_names = transformed_names[:n]

        aggregated = defaultdict(float)
        cat_set = set(categorical_features) if categorical_features else set()
        for fname, imp in zip(transformed_names, raw_importances):
            original = fname
            for cat in cat_set:
                if fname == cat or fname.startswith(cat + "_") or fname.startswith("cat__" + cat + "_") or fname.startswith("cat__" + cat + " "):
                    original = cat
                    break
            if original == fname and "__" in fname:
                parts = fname.split("__", 1)
                if len(parts) == 2:
                    col_part = parts[1]
                    underscore_idx = col_part.rfind("_")
                    if underscore_idx > 0:
                        candidate = col_part[:underscore_idx]
                        if candidate in cat_set:
                            original = candidate
            aggregated[original] += float(imp)

        all_sorted = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        total_features = len(all_sorted)
        selected = all_sorted[:top_n] if top_n is not None else all_sorted

        if selected and selected[0][1] > 0:
            max_imp = selected[0][1]
            items = [{"feature": f, "importance": round(v / max_imp, 4)} for f, v in selected]
        else:
            items = [{"feature": f, "importance": round(v, 4)} for f, v in selected]

        return {
            "items": items,
            "total_features": total_features,
            "features_shown": len(selected),
        }

    except Exception as e:
        print(f"[WARN] Could not extract feature importance: {e}")
        return None

def compute_roc_curve(y_test, y_prob, max_points=200):
    if y_test is None or y_prob is None:
        return None
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Downsample if too many points for frontend performance
        if len(fpr) > max_points:
            indices = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
            fpr = fpr[indices]
            tpr = tpr[indices]
            
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc)
        }
    except Exception as e:
        print(f"[WARN] Could not compute ROC curve: {e}")
        return None

def compute_pr_curve(y_test, y_prob, max_points=200):
    if y_test is None or y_prob is None:
        return None
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        # Sort by recall to make it easy to plot as line chart
        sort_indices = np.argsort(recall)
        recall = recall[sort_indices]
        precision = precision[sort_indices]
        
        if len(recall) > max_points:
            indices = np.linspace(0, len(recall) - 1, max_points, dtype=int)
            recall = recall[indices]
            precision = precision[indices]
            
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist()
        }
    except Exception as e:
        print(f"[WARN] Could not compute PR curve: {e}")
        return None

def compute_threshold_metrics(y_test, y_prob, num_steps=50):
    if y_test is None or y_prob is None:
        return None
    try:
        thresholds = np.linspace(0.05, 0.95, num_steps)
        precisions = []
        recalls = []
        f1s = []
        
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            precisions.append(float(precision_score(y_test, y_pred, zero_division=0)))
            recalls.append(float(recall_score(y_test, y_pred, zero_division=0)))
            f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
            
        return {
            "thresholds": thresholds.tolist(),
            "precision": precisions,
            "recall": recalls,
            "f1": f1s
        }
    except Exception as e:
        print(f"[WARN] Could not compute threshold metrics: {e}")
        return None

def compute_class_distribution(y_test):
    if y_test is None:
        return None
    try:
        # Assuming binary 0/1 for now
        n_total = len(y_test)
        n_pos = int(np.sum(y_test))
        n_neg = n_total - n_pos
        return {
            "labels": ["0", "1"],
            "counts": [n_neg, n_pos],
            "percentages": [round(n_neg/n_total*100, 1), round(n_pos/n_total*100, 1)]
        }
    except Exception as e:
        print(f"[WARN] Could not compute class distribution: {e}")
        return None

def generate_all_chart_data(pipeline, feature_order, categorical_features, y_test=None, y_prob=None):
    """
    Generates data for all optional frontend charts. Returns a dict to merge into metrics.json.
    """
    chart_data = {}
    
    # Feature Importance (always available if pipeline is provided)
    fi_result = extract_feature_importance(pipeline, feature_order, categorical_features)
    if fi_result is not None:
        chart_data["feature_importance_v2"] = fi_result
        
        # Also return flat format for backward compatibility in backend
        chart_data["feature_importance"] = fi_result["items"]
        chart_data["total_features"] = fi_result["total_features"]
        chart_data["features_shown"] = fi_result["features_shown"]
        
    # Validation metrics curves (only available if y_test/y_prob are provided)
    if y_test is not None and y_prob is not None:
        chart_data["roc_curve"] = compute_roc_curve(y_test, y_prob)
        chart_data["pr_curve"] = compute_pr_curve(y_test, y_prob)
        chart_data["threshold_metrics"] = compute_threshold_metrics(y_test, y_prob)
        chart_data["class_distribution"] = compute_class_distribution(y_test)
        
    # Remove nulls
    return {k: v for k, v in chart_data.items() if v is not None}
