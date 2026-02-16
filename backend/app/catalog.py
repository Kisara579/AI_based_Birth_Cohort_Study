from typing import Any, Dict, List,Optional


def get_outcomes() -> List[Dict[str, Any]]:
    return [
        {
            "id": "lbw",
            "name": "Low Birth Weight",
            "short": "LBW",
            "description": "Predict risk of low birth weight based on maternal factors.",
        }
    ]


def get_schema(outcome_id: str) -> Optional[Dict[str, Any]]:
    if outcome_id == "lbw":
        return {
            "title": "Low Birth Weight Inputs",
            "sections": [
                {
                    "id": "maternal",
                    "title": "Maternal Information",
                    "fields": [
                        {
                            "name": "mage",
                            "label": "Mother age (years)",
                            "type": "number",
                            "min": 10,
                            "max": 60,
                            "required": True,
                            "placeholder": "e.g., 28",
                        },
                        {
                            "name": "meduc",
                            "label": "Mother education (years)",
                            "type": "number",
                            "min": 0,
                            "max": 25,
                            "required": True,
                            "placeholder": "e.g., 12",
                        },
                        {
                            "name": "cigs",
                            "label": "Cigarettes per day",
                            "type": "number",
                            "min": 0,
                            "max": 60,
                            "required": True,
                            "placeholder": "e.g., 0",
                        },
                        {
                            "name": "drink",
                            "label": "Alcohol use",
                            "type": "select",
                            "required": True,
                            "options": [
                                {"value": 0, "label": "No"},
                                {"value": 1, "label": "Yes"},
                            ],
                        },
                    ],
                },
                {
                    "id": "pregnancy",
                    "title": "Pregnancy Care",
                    "fields": [
                        {
                            "name": "monpre",
                            "label": "Month prenatal care started",
                            "type": "number",
                            "min": 1,
                            "max": 9,
                            "required": True,
                            "placeholder": "e.g., 2",
                        },
                        {
                            "name": "npvis",
                            "label": "Number of prenatal visits",
                            "type": "number",
                            "min": 0,
                            "max": 40,
                            "required": True,
                            "placeholder": "e.g., 8",
                        },
                        {
                            "name": "male",
                            "label": "Baby sex",
                            "type": "select",
                            "required": True,
                            "options": [
                                {"value": 1, "label": "Male"},
                                {"value": 0, "label": "Female"},
                            ],
                        },
                    ],
                },
            ],
        }

    return None
