from typing import Any, Dict
from pydantic import BaseModel


class PredictRequest(BaseModel):
    data: Dict[str, Any]


class PredictResponse(BaseModel):
    probability: float
    risk_label: str
