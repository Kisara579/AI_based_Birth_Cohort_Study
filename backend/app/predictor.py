from __future__ import annotations
from typing import Any, Dict, Tuple
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import HTTPException
from .config import model_paths
from .catalog import select_best_model, get_feature_engineer


_MODEL_CACHE: Dict[str, Any] = {}


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing meta file: {meta_path.name}")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid meta JSON: {meta_path.name} ({e})")

def _load_model(model_key: str, algorithm: str | None = None) -> Any:
    cache_key = f"{model_key}:{algorithm}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_path, *_ = model_paths(model_key, algorithm=algorithm)

    if not model_path.exists():
        raise HTTPException(status_code=501, detail=f"Model not found: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_path} ({e})")

    _MODEL_CACHE[cache_key] = model
    return model


def predict_with_model_key(model_key: str, values: Dict[str, Any], algorithm: str | None = None) -> Tuple[float, str]:   
    """
    model_key = outcome key such as "lbw"
    Loads the selected algorithm from:
      backend/models/<outcome_folder>/<algorithm>/model.pkl
      backend/models/<outcome_folder>/<algorithm>/meta.json
    If algorithm is None, the active model from registry.json is used.
    Returns probability + risk label using threshold from meta.json.
    """

    if algorithm is None:
        algorithm = select_best_model(model_key)

    model_path, meta_path, _ = model_paths(model_key, algorithm=algorithm)

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Algorithm '{algorithm}' not available for outcome '{model_key}'"
    )

    model = _load_model(model_key, algorithm=algorithm)
    meta = _load_meta(meta_path)

    feature_order = meta.get("feature_order")
    if not isinstance(feature_order, list) or len(feature_order) == 0:
        raise HTTPException(status_code=500, detail=f"Meta missing valid 'feature_order' for {model_key}")

    threshold = meta.get("threshold", 0.5)
    try:
        threshold = round(float(threshold), 2)
    except Exception:
        threshold = 0.5

    # Apply outcome-specific feature engineering from the catalog (e.g. interaction terms for LBW)
    engineer_fn = get_feature_engineer(model_key)
    if engineer_fn:
        values = engineer_fn(values)

    missing = [f for f in feature_order if f not in values]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {', '.join(missing)}")

    row_dict = {f: ("" if values.get(f) is None else values.get(f)) for f in feature_order}
    X_df = pd.DataFrame([row_dict], columns=feature_order)

    categorical = set(meta.get("categorical_features", []))
    for c in categorical:
        if c in X_df.columns:
            X_df[c] = X_df[c].astype(str)

    for c in feature_order:
        if c not in categorical:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    invalid_numeric = [
        c for c in feature_order
        if c not in categorical and X_df[c].isna().any()
    ]
    if invalid_numeric:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid numeric values for: {', '.join(invalid_numeric)}"
        )

    try:
        proba = model.predict_proba(X_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model predict_proba failed for {model_key}: {e}")

    probability = float(proba)
    risk_label = "High Risk" if probability >= threshold else "Low Risk"
    return probability, risk_label



