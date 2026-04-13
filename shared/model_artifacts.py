import json
import joblib
from pathlib import Path
from shared.charts import generate_all_chart_data


def _compute_overfitting_analysis(train_metrics, test_metrics):
    """Compute overfitting analysis by comparing train vs test metrics."""
    if not train_metrics or not test_metrics:
        return None

    analysis = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gaps": {},
    }

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        train_val = train_metrics.get(key, 0)
        test_val = test_metrics.get(key, 0)
        gap = round(train_val - test_val, 4)
        analysis["gaps"][key] = gap

    # Determine overfitting status based on F1 and Recall gaps
    f1_gap = abs(analysis["gaps"].get("f1", 0))
    recall_gap = abs(analysis["gaps"].get("recall", 0))
    avg_gap = (f1_gap + recall_gap) / 2

    if avg_gap <= 0.03:
        status = "Well Generalized"
        severity = "good"
    elif avg_gap <= 0.08:
        status = "Mild Overfitting"
        severity = "warning"
    elif avg_gap <= 0.15:
        status = "Overfitting"
        severity = "danger"
    else:
        status = "Severe Overfitting"
        severity = "critical"

    analysis["status"] = status
    analysis["severity"] = severity
    analysis["avg_gap"] = round(avg_gap, 4)

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
            print(f"  [OK] Overfitting analysis: {overfitting['status']} (avg gap: {overfitting['avg_gap']:.4f})")

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
