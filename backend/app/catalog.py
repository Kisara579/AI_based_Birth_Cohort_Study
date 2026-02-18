from typing import Any, Dict, List, Optional

OUTCOMES: Dict[str, Dict[str, Any]] = {
    "pph": {
        "id": "pph",
        "name": "Postpartum haemorrhage  ",
        "short": "PPH",
        "description": "Predict risk of heavy bleeding after delivery.",
        "model_key": "pph",
        "schema": {"title": "PPH Inputs", "sections": []}
    },

    "preeclampsia": {
        "id": "preeclampsia",
        "name": "Preeclampsia",
        "short": "PE",
        "description": "Predict risk of hypertensive disorder during pregnancy.",
        "model_key": "preeclampsia",
        "schema": {"title": "Preeclampsia Inputs", "sections": []}
    },

    "iugr": {
        "id": "iugr",
        "name": "Intrauterine Growth Restriction",
        "short": "IUGR",
        "description": "Predict risk of intrauterine growth restriction.",
        "model_key": "iugr",
        "schema": {"title": "IUGR Inputs", "sections": []}
    },

    "gdm": {
        "id": "gdm",
        "name": "Gestational Diabetes",
        "short": "GDM",
        "description": "Predict risk of diabetes during pregnancy.",
        "model_key": "gdm",
        "schema": {"title": "GDM Inputs", "sections": []}
    },

    "iol": {
        "id": "iol",
        "name": "Women requiring induction of labor",
        "short": "IOL",
        "description": "Predict risk of intrauterine onset of labor.",
        "model_key": "iol",
        "schema": {"title": "IOL Inputs", "sections": []}
    },

    "c_section": {
        "id": "c_section",
        "name": "Caesarean Section",
        "short": "CS",
        "description": "Predict likelihood of cesarean delivery.",
        "model_key": "c_section",
        "schema": {"title": "C-Section Inputs", "sections": []}
    },

    "postpartum": {
        "id": "postpartum",
        "name": "Postpartum depression",
        "short": "PPD",
        "description": "Predict likelihood of postpartum depression.",
        "model_key": "postpartum",
        "schema": {"title": "Postpartum Depression Inputs", "sections": []}
    },


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

    
    "nicu": {
        "id": "nicu",
        "name": "Neonatal Intensive Care Unit Admission",
        "short": "NICU",
        "description": "Predict probability of NICU admission.",
        "model_key": "nicu",
        "schema": {"title": "NICU Inputs", "sections": []}
    },

    
    "feeding": {
        "id": "feeding",
        "name": "Feeding Problems",
        "short": "FP",
        "description": "Predict risk of feeding problems in newborns.",
        "model_key": "feeding",
        "schema": {"title": "Feeding Problems Inputs", "sections": []}
    }, 

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

