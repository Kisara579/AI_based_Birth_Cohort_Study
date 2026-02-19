from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import HTTPException
from .config import model_paths


# cache: model_key -> loaded model object
_MODEL_CACHE: Dict[str, Any] = {}


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing meta file: {meta_path.name}")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid meta JSON: {meta_path.name} ({e})")


def _load_model(model_key: str) -> Any:
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    model_path, _ = model_paths(model_key)

    if not model_path.exists():
        # Model not present: teammate hasn't added it yet
        raise HTTPException(status_code=501, detail=f"Model not found: {model_path.name}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_path.name} ({e})")

    _MODEL_CACHE[model_key] = model
    return model


def predict_with_model_key(model_key: str, values: Dict[str, Any]) -> Tuple[float, str]:
    """
    Generic predictor for any model_key:
    - loads <model_key>_model.pkl
    - loads <model_key>_model.meta.json
    - validates required features
    - returns probability + risk label using threshold from meta
    """
    model_path, meta_path = model_paths(model_key)

    model = _load_model(model_key)
    meta = _load_meta(meta_path)

    feature_order = meta.get("feature_order")
    if not isinstance(feature_order, list) or len(feature_order) == 0:
        raise HTTPException(status_code=500, detail=f"Meta missing valid 'feature_order' for {model_key}")

    threshold = meta.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    missing = [f for f in feature_order if f not in values]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {', '.join(missing)}")

    # Build a 1-row dataframe in the exact feature order
    row_dict = {f: (0 if values.get(f) is None else values.get(f)) for f in feature_order}
    X_df = pd.DataFrame([row_dict], columns=feature_order)

    # sklearn-style predict_proba expected
    try:
        proba = model.predict_proba(X_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model predict_proba failed for {model_key}: {e}")

    probability = float(proba)
    risk_label = "High Risk" if probability >= threshold else "Low Risk"
    return probability, risk_label
