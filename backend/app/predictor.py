from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import joblib

from .config import LBW_MODEL_PATH


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


def predict_lbw(values: Dict[str, Any]) -> Tuple[float, str]:
    """
    If trained model exists at backend/models/lbw_model.pkl, use it.
    Otherwise use mock scoring fallback.
    """
    model = _load_lbw_model()
    if model is None:
        return _mock_predict_lbw(values)

    # IMPORTANT: model expects features in a specific order.
    # For now we assume the training pipeline matches these feature names.
    feature_order = ["mage", "meduc", "cigs", "drink", "monpre", "npvis", "male"]

    row = []
    for f in feature_order:
        v = values.get(f)
        # If missing, default to 0 (you can change later)
        row.append(0 if v is None else v)

    # model can be sklearn pipeline or xgboost wrapper supporting predict_proba
    proba = model.predict_proba([row])[0][1]
    probability = float(proba)
    risk_label = "High Risk" if probability >= 0.5 else "Low Risk"
    return probability, risk_label
