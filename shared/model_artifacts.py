import json
import joblib
from pathlib import Path
from shared.charts import generate_all_chart_data


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _compute_overfitting_analysis(train_metrics, test_metrics):
    """Compute overfitting analysis from train-vs-test metric gaps."""
    if not train_metrics or not test_metrics:
        return None

    analysis = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gaps": {},
    }

    for key in METRIC_KEYS:
        train_val = float(train_metrics.get(key, 0.0))
        test_val = float(test_metrics.get(key, 0.0))
        analysis["gaps"][key] = round(train_val - test_val, 4)

    gaps = analysis["gaps"]
    max_gap = max(gaps.get(key, 0.0) for key in METRIC_KEYS)
    min_gap = min(gaps.get(key, 0.0) for key in METRIC_KEYS)
    avg_gap = sum(gaps.get(key, 0.0) for key in METRIC_KEYS) / len(METRIC_KEYS)

    # Overfitting check based on the 5% (0.05) threshold across all metrics:
    if max_gap > 0.10:
        status = "Severe Overfitting"
        severity = "critical"
        note = "Train-to-validation gap exceeds 10% on one or more metrics"
    elif max_gap > 0.05:
        status = "Overfitting"
        severity = "danger"
        note = "Train-to-validation gap exceeds the 5% threshold on one or more metrics"
    elif min_gap < -0.05:
        status = "Possible Underfitting/Split Instability"
        severity = "warning"
        note = "Validation performance exceeds train performance by more than 5% on one or more metrics"
    elif max_gap <= 0.03 and min_gap >= -0.03:
        status = "Well Generalized"
        severity = "good"
        note = "Small gaps (within 3%) across all evaluation metrics"
    else:
        status = "Mild Overfitting"
        severity = "warning"
        note = "Generalization gap is present but remains under the 5% threshold across all metrics"

    analysis["status"] = status
    analysis["severity"] = severity
    analysis["avg_gap"] = round(avg_gap, 4)
    # analysis["note"] = note

    return analysis


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
    y_test=None,
    y_prob=None,
    train_metrics=None,
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

    # Generate all chart data (feature importance + curves) and merge
    chart_data = generate_all_chart_data(pipeline, feature_order, categorical_features, y_test, y_prob)
    metrics_payload.update(chart_data)
    
    # Print status message
    if "total_features" in chart_data:
        print(f"  [OK] Feature importance extracted ({chart_data['features_shown']}/{chart_data['total_features']} features)")
    if "roc_curve" in chart_data:
        print("  [OK] ROC and PR curves generated")

    # Compute overfitting analysis
    if train_metrics is not None:
        overfitting = _compute_overfitting_analysis(train_metrics, metrics_payload.get("metrics", {}))
        if overfitting:
            metrics_payload["overfitting_analysis"] = overfitting
            print(f"  [OK] Overfitting analysis: {overfitting['status']} (avg gap: {overfitting['avg_gap']:+.4f})")

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
            "selection_metric": "recall",
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
