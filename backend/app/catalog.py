from typing import Any, Dict, List, Optional

OUTCOMES: Dict[str, Dict[str, Any]] = {
    "lbw": {
        "id": "lbw",
        "name": "Low Birth Weight",
        "short": "LBW",
        "description": "Predict risk of low birth weight based on maternal factors.",
        "model_key": "lbw",
        "schema": {
            "title": "Low Birth Weight Inputs",
            "sections": [
                {
                    "id": "maternal",
                    "title": "Maternal Information",
                    "fields": [
                        {"name": "mage", "label": "Mother age (years)", "type": "number", "min": 10, "max": 60, "required": True, "placeholder": "e.g., 28"},
                        {"name": "meduc", "label": "Mother education (years)", "type": "number", "min": 0, "max": 25, "required": True, "placeholder": "e.g., 12"},
                        {"name": "cigs", "label": "Cigarettes per day", "type": "number", "min": 0, "max": 60, "required": True, "placeholder": "e.g., 0"},
                        {"name": "drink", "label": "Alcohol use", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                    ],
                },
                {
                    "id": "pregnancy",
                    "title": "Pregnancy Care",
                    "fields": [
                        {"name": "monpre", "label": "Month prenatal care started", "type": "number", "min": 1, "max": 9, "required": True, "placeholder": "e.g., 2"},
                        {"name": "npvis", "label": "Number of prenatal visits", "type": "number", "min": 0, "max": 40, "required": True, "placeholder": "e.g., 8"},
                        {"name": "male", "label": "Baby sex", "type": "select", "required": True, "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}]},
                    ],
                },
            ],
        },
    },
    "preterm": {
        "id": "preterm",
        "name": "Preterm Birth",
        "short": "PTB",
        "description": "Predict likelihood of delivery before 37 weeks.",
        "model_key": "preterm",
        "schema": {"title": "Preterm Birth Inputs", "sections": []}
    },

    "preeclampsia": {
        "id": "preeclampsia",
        "name": "Preeclampsia",
        "short": "PE",
        "description": "Predict risk of hypertensive disorder during pregnancy.",
        "model_key": "preeclampsia",
        "schema": {"title": "Preeclampsia Inputs", "sections": []}
    },

    "gdm": {
        "id": "gdm",
        "name": "Gestational Diabetes",
        "short": "GDM",
        "description": "Predict risk of diabetes during pregnancy.",
        "model_key": "gdm",
        "schema": {"title": "GDM Inputs", "sections": []}
    },

    "c_section": {
        "id": "c_section",
        "name": "C-Section Delivery",
        "short": "CS",
        "description": "Predict likelihood of cesarean delivery.",
        "model_key": "c_section",
        "schema": {"title": "C-Section Inputs", "sections": []}
    },

    "nicu": {
        "id": "nicu",
        "name": "NICU Admission",
        "short": "NICU",
        "description": "Predict probability of NICU admission.",
        "model_key": "nicu",
        "schema": {"title": "NICU Inputs", "sections": []}
    },

    "stillbirth": {
        "id": "stillbirth",
        "name": "Stillbirth",
        "short": "SB",
        "description": "Predict risk of stillbirth.",
        "model_key": "stillbirth",
        "schema": {"title": "Stillbirth Inputs", "sections": []}
    },

    "pph": {
        "id": "pph",
        "name": "Postpartum Hemorrhage",
        "short": "PPH",
        "description": "Predict risk of heavy bleeding after delivery.",
        "model_key": "pph",
        "schema": {"title": "PPH Inputs", "sections": []}
    },

    "neonatal_mortality": {
        "id": "neonatal_mortality",
        "name": "Neonatal Mortality",
        "short": "NM",
        "description": "Predict risk of neonatal death within 28 days.",
        "model_key": "neonatal_mortality",
        "schema": {"title": "Neonatal Mortality Inputs", "sections": []}
    },

    "congenital_anomaly": {
        "id": "congenital_anomaly",
        "name": "Congenital Anomaly",
        "short": "CA",
        "description": "Predict likelihood of congenital anomalies.",
        "model_key": "congenital_anomaly",
        "schema": {"title": "Congenital Anomaly Inputs", "sections": []}
    },

    "sga": {
        "id": "sga",
        "name": "Small for Gestational Age",
        "short": "SGA",
        "description": "Predict risk of small for gestational age.",
        "model_key": "sga",
        "schema": {"title": "SGA Inputs", "sections": []}
    }

}


def get_outcomes() -> List[Dict[str, Any]]:
    # Return only public fields (don’t expose schema here)
    return [
        {k: v for k, v in data.items() if k in ["id", "name", "short", "description"]}
        for data in OUTCOMES.values()
    ]


def get_schema(outcome_id: str) -> Optional[Dict[str, Any]]:
    outcome = OUTCOMES.get(outcome_id)
    if not outcome:
        return None
    return outcome.get("schema")

def get_model_key(outcome_id: str) -> Optional[str]:
    outcome = OUTCOMES.get(outcome_id)
    if not outcome:
        return None
    return outcome.get("model_key")

