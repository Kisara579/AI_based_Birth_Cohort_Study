import json
import joblib
from pathlib import Path


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

    