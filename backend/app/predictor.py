from typing import Any, Dict, Tuple


def predict_lbw(values: Dict[str, Any]) -> Tuple[float, str]:
    """
    Temporary mock predictor (NOT medical, NOT real model).
    Later replace with real model .pkl loading and predict_proba.
    """
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
