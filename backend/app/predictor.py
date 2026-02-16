from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import joblib
import json
from pathlib import Path
from fastapi import HTTPException
from .config import LBW_MODEL_PATH, LBW_META_PATH



# Cache model in memory so we don't load it every request
_LBW_MODEL: Optional[Any] = None


def _load_lbw_model() -> Optional[Any]:
    global _LBW_MODEL
    if _LBW_MODEL is not None:
        return _LBW_MODEL

    if not LBW_MODEL_PATH.exists():
        return None

    _LBW_MODEL = joblib.load(LBW_MODEL_PATH)
    return _LBW_MODEL


def _mock_predict_lbw(values: Dict[str, Any]) -> Tuple[float, str]:
    # Temporary mock predictor (NOT medical, NOT real model)
    score = 0.0

    mage = values.get("mage")
    meduc = values.get("meduc")
    cigs = values.get("cigs")
    drink = values.get("drink")
    monpre = values.get("monpre")
    npvis = values.get("npvis")

    if mage is not None and mage < 18:
        score += 0.12

    if meduc is not None and meduc < 10:
        score += 0.08

    if cigs is not None:
        score += min(0.25, float(cigs) * 0.01)

    if drink == 1:
        score += 0.12

    if monpre is not None and monpre > 3:
        score += 0.10

    if npvis is not None and npvis < 4:
        score += 0.10

    probability = max(0.05, min(0.95, 0.15 + score))
    risk_label = "High Risk" if probability >= 0.5 else "Low Risk"
    return probability, risk_label


from fastapi import HTTPException

def predict_lbw(values: Dict[str, Any]) -> Tuple[float, str]:
    """
    Rules:
    - If model file exists -> MUST use meta.json and MUST validate required features.
    - If model file does not exist -> use mock fallback.
    """
    # If no model file, allow fallback
    if not LBW_MODEL_PATH.exists():
        return _mock_predict_lbw(values)

    # Model file exists: load model (must succeed)
    model = _load_lbw_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model file exists but failed to load")

    # Meta file is REQUIRED when model exists
    meta = _load_meta(LBW_META_PATH)

    feature_order = meta.get("feature_order")
    if not isinstance(feature_order, list) or len(feature_order) == 0:
        raise HTTPException(status_code=500, detail="Meta file missing valid 'feature_order'")

    threshold = meta.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.5

    # Validate missing features STRICTLY
    missing = [f for f in feature_order if f not in values]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {', '.join(missing)}",
        )

    row = []
    for f in feature_order:
        v = values.get(f)
        row.append(0 if v is None else v)

    proba = model.predict_proba([row])[0][1]
    probability = float(proba)
    risk_label = "High Risk" if probability >= threshold else "Low Risk"
    return probability, risk_label



def _load_meta(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing meta file: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid meta JSON: {path.name} ({e})")
