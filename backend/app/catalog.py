from typing import Any, Dict, List, Optional
import json
from pathlib import Path

OUTCOMES: Dict[str, Dict[str, Any]] = {
    "pph": {
        "id": "pph",
        "name": "Postpartum Haemorrhage",
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
        "name": "Women Requiring Induction Of Labor",
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

    "ppd": {
        "id": "ppd",
        "name": "Postpartum Depression",
        "short": "PPD",
        "description": "Predict likelihood of postpartum depression.",
        "model_key": "ppd",
        "schema": {
            "title": "Postpartum Depression Inputs", 
            "sections": [
                {
                    "fields":[
                        {"name": "Age", "label": "Mother age (years)", "type": "number", "min": 10, "max": 60, "required": True, "placeholder": "e.g., 28"},
                        {"name": "Education Level", "label": "Education level", "type": "select", "required": True, "options": [{"value": "Primary school", "label": "Primary school"},{"value": "High school", "label": "High school"},{"value": "College", "label": "College"},{"value": "University", "label": "University"}]},
                        {"name": "Marital status", "label": "Marital status", "type": "select", "required": True, "options": [{"value": "Married", "label": "Married"},{"value": "Divorced", "label": "Divorced"}]},
                        {"name": "Husband's education level", "label": "Husband's education level", "type": "select", "required": True, "options": [{"value": "Primary school", "label": "Primary school"},{"value": "High school", "label": "High school"},{"value": "College", "label": "College"},{"value": "University", "label": "University"}]},
                        {"name": "Occupation After Your Latest Childbirth", "label": "Occupation", "type": "select", "required": True, "options": [{"value": "Housewife", "label": "Housewife"},{"value": "Student", "label": "Student"},{"value": "Teacher", "label": "Teacher"},{"value": "Doctor", "label": "Doctor"},{"value": "Service", "label": "Service"},{"value": "Business", "label": "Business"},{"value": "Other", "label": "Other"}]},
                        {"name": "Recieved Support", "label": "Received support", "type": "select", "required": True, "options": [{"value": "Low", "label": "Low"},{"value": "Medium", "label": "Medium"},{"value": "High", "label": "High"}]},
                    ]
                }
            ]
        }
    },


    "lbw": {
        "id": "lbw",
        "name": "Low Birth Weight",
        "short": "LBW",
        "description": "Predict risk of low birth weight based on maternal and delivery factors.",
        "model_key": "lbw",
        "schema": {
            "title": "Low Birth Weight Inputs",
            "sections": [
                {
                    "fields": [
                        {"name": "MAGER", "label": "Mother age (years)", "type": "number", "min": 10, "max": 60, "required": True, "placeholder": "e.g., 28"},
                        {"name": "MEDUC", "label": "Mother education level", "type": "select", "required": True, "options": [{"value": 1, "label": "No formal education / Primary"},{"value": 2, "label": "Grade 6–11"},{"value": 3, "label": "Passed O/L"},{"value": 4, "label": "Passed A/L"},{"value": 5, "label": "Diploma / Associate"},{"value": 6, "label": "Degree"},{"value": 7, "label": "Postgraduate"},{"value": 8, "label": "Doctorate"},{"value": 9, "label": "Unknown"}]},
                        {"name": "BMI", "label": "Body Mass Index (BMI)", "type": "number", "min": 10, "max": 80, "required": True, "placeholder": "e.g., 23.4"},
                        {"name": "CIG_0", "label": "Cigarettes before pregnancy (per day)", "type": "number", "min": 0, "max": 60, "required": True, "placeholder": "e.g., 0"},
                        {"name": "PREVIS", "label": "Number of prenatal visits", "type": "number", "min": 0, "max": 50, "required": True, "placeholder": "e.g., 10"},
                        {"name": "SEX", "label": "Sex", "type": "select", "required": True, "options": [{"value":"M","label":"Male"}, {"value":"F","label":"Female"}]}
                    ],
                }
            ],
        },
    },



    "pd": {
        "id": "pd",
        "name": "Premature Delivery",
        "short": "PD",
        "description": "Predict likelihood of delivery before 37 weeks.",
        "model_key": "pd",
        "schema": {
            "title": "Premature Delivery Inputs",
            "sections": [
                {
                    "fields": [
                        {"name": "MAGE", "label": "Mother age", "type": "number", "min": 10, "max": 60, "required": True, "placeholder": "e.g., 28"},
                        {"name": "CARDIAC", "label": "Cardiac disease", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "ACLUNG", "label": "Acute or chronic lung disease", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "DIABETES", "label": "Diabetes", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HYPERCH", "label": "Chronic hypertension", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HYPERPR", "label": "Pregnancy hypertension", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "PINFANT", "label": "Previous infant over 4000 grams", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "RENAL", "label": "Renal disease", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "PRETERM", "label": "Previous preterm or small infant", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]}
                    ]
                }
            ]
        }
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
    # Return only public fields 
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


# select best algorithm 
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"

def select_best_model(outcome: str):

    outcome_dir = MODELS_ROOT / outcome
    registry_path = outcome_dir / "registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found for outcome: {outcome}")

    registry = json.loads(registry_path.read_text())

    if registry.get("selection_mode") != "auto":
        return registry.get("active_model")

    metric_name = registry.get("selection_metric", "f1")

    best_model = None
    best_score = -1

    for model in registry.get("models", []):

        if not model.get("trained"):
            continue

        algo = model["algorithm"]

        metrics_path = outcome_dir / algo / "metrics.json"

        if not metrics_path.exists():
            continue

        metrics_data = json.loads(metrics_path.read_text())

        score = metrics_data["metrics"].get(metric_name)

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_model = algo

    if best_model is None:
        raise RuntimeError("No valid trained model found")

    # update registry
    registry["active_model"] = best_model
    registry_path.write_text(json.dumps(registry, indent=2))

    return best_model