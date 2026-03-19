import json
import joblib
import numpy as np
from pathlib import Path
from collections import defaultdict


def _extract_feature_importance(pipeline, feature_order, categorical_features, top_n=15):
    """
    Extract feature importance from a sklearn Pipeline for any supported estimator.

    Supports:
      - RandomForestClassifier / XGBClassifier  → .feature_importances_
      - LogisticRegression                       → abs(.coef_[0])
      - CalibratedClassifierCV(LinearSVC)        → abs(.coef_[0]) from base estimator
      - GradientBoostingClassifier               → .feature_importances_

    One-hot-encoded columns are aggregated back to the original categorical feature name.
    Returns a list of {feature, importance} dicts sorted descending, capped at top_n.
    """
    try:
        #Find the estimator inside the pipeline 
        estimator = None
        for name, step in reversed(pipeline.steps):
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                estimator = step
                break
            # CalibratedClassifierCV wraps a base estimator
            if hasattr(step, "calibrated_classifiers_"):
                base = step.calibrated_classifiers_[0].estimator
                if hasattr(base, "coef_") or hasattr(base, "feature_importances_"):
                    estimator = base
                    break

        if estimator is None:
            return None

        # Get raw importances 
        if hasattr(estimator, "feature_importances_"):
            raw_importances = np.array(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            coef = np.array(estimator.coef_)
            raw_importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return None

        # Map transformed feature indices back to original names 
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
                elif hasattr(transformer, "get_feature_names_out"):
                    ohe_names = transformer.get_feature_names_out(columns)
                    transformed_names.extend(ohe_names)
                else:
                    if isinstance(columns, list):
                        transformed_names.extend(columns)
                    else:
                        transformed_names.append(str(columns))
        else:
            transformed_names = list(feature_order)

        # Truncate or pad if mismatch
        n = len(raw_importances)
        if len(transformed_names) < n:
            transformed_names.extend([f"feature_{i}" for i in range(len(transformed_names), n)])
        transformed_names = transformed_names[:n]

        # Aggregate one-hot features back to original column names 
        aggregated = defaultdict(float)
        cat_set = set(categorical_features) if categorical_features else set()
        for fname, imp in zip(transformed_names, raw_importances):
            original = fname
            for cat in cat_set:
                if fname.startswith(cat + "_") or fname.startswith("cat__" + cat) or fname.startswith(cat):
                    original = cat
                    break
            # Also handle sklearn's "transformername__colname_value" pattern
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

        #  Sort and return top N 
        sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Normalize so the top feature is 1.0
        if sorted_features and sorted_features[0][1] > 0:
            max_imp = sorted_features[0][1]
            return [{"feature": f, "importance": round(v / max_imp, 4)} for f, v in sorted_features]

        return [{"feature": f, "importance": round(v, 4)} for f, v in sorted_features]

    except Exception as e:
        print(f"[WARN] Could not extract feature importance: {e}")
        return None


def save_model_artifacts(
    pipeline,
    outcome,
    outcome_id,
    algorithm,
    label,
    description,
    target_column,
    feature_order,
    categorical_features,
    metrics_payload,
    positive_label,
    negative_label,
    threshold,
):

    base_dir = Path("backend/models") / outcome
    model_dir = base_dir / algorithm
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model.pkl
    model_path = model_dir / "model.pkl"
    joblib.dump(pipeline, model_path)

    # Save meta.json
    meta = {
        "outcome": outcome,
        "outcome_id": outcome_id,
        "algorithm": algorithm,
        "target_column": target_column,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "feature_order": feature_order,
        "categorical_features": categorical_features,
        "threshold": float(round(threshold, 2)),
    }

    meta_path = model_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Extract feature importance and add to metrics payload
    fi = _extract_feature_importance(pipeline, feature_order, categorical_features)
    if fi is not None:
        metrics_payload["feature_importance"] = fi
        print(f"  [OK] Feature importance extracted ({len(fi)} features)")

    # Save metrics.json
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Update registry.json
    registry_path = base_dir / "registry.json"

    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {
            "outcome": outcome,
            "outcome_id": outcome_id,
            "selection_mode": "auto",
            "selection_metric": "f1",
            "active_model": None,
            "models": [],
        }

    registry.setdefault("models", [])

    found = False
    for entry in registry["models"]:
        if entry["algorithm"] == algorithm:
            entry["trained"] = True
            entry["label"] = label
            entry["description"] = description
            found = True
            break

    if not found:
        registry["models"].append({
            "algorithm": algorithm,
            "label": label,
            "description": description,
            "trained": True
        })

    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("\nArtifacts saved:")
    print(" -", model_path)
    print(" -", meta_path)
    print(" -", metrics_path)
    print(" -", registry_path)
