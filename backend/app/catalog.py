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
                        {"name": "Recieved Support", "label": "Received support", "type": "select", "required": True, "options": [{"value": "1", "label": "Low"},{"value": "2", "label": "Medium"},{"value": "3", "label": "High"}]},
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
        "schema": {
            "title": "NICU Inputs", 
            "sections": [
                {
                    "fields": [
                        {"name": "Insulin_treatment_during_pregnancy", "label": "Insulin treatment during pregnancy", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Pregnancy_complications_hypertensive_disorders", "label": "Hypertensive disorders during pregnancy", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Pregnancy_complications_preterm_delivery", "label": "Preterm delivery complications", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Abnormal_OGTT_results", "label": "Abnormal OGTT results", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Insulin_x_Preterm", "label": "Insulin & Preterm Delivery", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Hypertension_x_OGTT", "label": "Hypertension & Abnormal OGTT", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Insulin_x_Hypertension", "label": "Insulin & Hypertension", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "Risk_score", "label": "Risk Score (0-4)", "type": "number", "min": 0, "max": 4, "required": True, "placeholder": "e.g., 2"}
                    ]
                }
            ]
        }
    },

    
    "feeding": {
        "id": "feeding",
        "name": "Feeding Problems",
        "short": "FP",
        "description": "Predict risk of feeding problems in newborns.",
        "model_key": "feeding",
        "schema": {
            "title": "Feeding Problems Inputs", 
            "sections": [
                {
                    "fields":[
                        {"name": "maternal_age", "label": "Maternal age (years)", "type": "number", "min": 20, "max": 53, "required": True, "placeholder": "e.g., 28"},
                        {"name": "maternal_education", "label": "Maternal education level", "type": "select", "required": True, "options": [
                            {"value": 1, "label": "Primary school"},
                            {"value": 2, "label": "High school"},
                            {"value": 3, "label": "College"},
                            {"value": 4, "label": "Vocational / Associate"},
                            {"value": 5, "label": "University (Undergraduate)"},
                            {"value": 6, "label": "Postgraduate University"},
                            {"value": 99, "label": "Unknown / Prefer not to say"}
                        ]},
                        {"name": "child_birthweight_kgs", "label": "Child birthweight (kg)", "type": "number", "min": 0.5, "max": 6.0, "required": True, "placeholder": "e.g., 3.2"},
                        {"name": "child_medical_complications", "label": "Child medical complications at birth", "type": "select", "required": True, "options": [
                            {"value": 2, "label": "No"},
                            {"value": 1, "label": "Yes"}
                        ]},
                        {"name": "delivery_method", "label": "Delivery method", "type": "select", "required": True, "options": [
                            {"value": 1, "label": "Spontaneous Vaginal"},
                            {"value": 2, "label": "Planned Cesarean"},
                            {"value": 3, "label": "Emergency Cesarean"},
                            {"value": 4, "label": "Instrumental Vaginal (Forceps/Vacuum)"},
                            {"value": 5, "label": "Other"},
                            {"value": 99, "label": "Unknown"}
                        ]},
                        {"name": "breastfeeding_duration_months", "label": "Breastfeeding duration (months)", "type": "number", "min": 0, "max": 82, "required": True, "placeholder": "e.g., 6"},
                        {"name": "MH_postnatal_depression", "label": "Maternal postnatal depression", "type": "select", "required": True, "options": [
                            {"value": 0, "label": "No"},
                            {"value": 1, "label": "Yes"}
                        ]},
                        {"name": "MH_postnatal_anxiety", "label": "Maternal postnatal anxiety", "type": "select", "required": True, "options": [
                            {"value": 0, "label": "No"},
                            {"value": 1, "label": "Yes"}
                        ]}
                    ]
                }
            ]}
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

BLOCKED_SEVERITIES = {"danger", "critical"}
SEVERITY_RANK = {
    "good": 2,
    "warning": 1,
}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_selection_score(metrics: Dict[str, Any], severity: str) -> float:
    """Score candidate models with a recall-sensitive composite metric."""
    f1 = _to_float(metrics.get("f1", 0.0))
    recall = _to_float(metrics.get("recall", 0.0))
    roc_auc = _to_float(metrics.get("roc_auc", 0.0))

    warning_penalty = 0.01 if severity == "warning" else 0.0
    return (0.55 * f1) + (0.35 * recall) + (0.10 * roc_auc) - warning_penalty

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
    best_rank = None
    best_context: Dict[str, Any] = {}
    evaluated = 0
    blocked: List[Dict[str, Any]] = []

    for model in registry.get("models", []):

        if not model.get("trained"):
            continue

        algo = model["algorithm"]
        metrics_path = outcome_dir / algo / "metrics.json"

        if not metrics_path.exists():
            continue

        metrics_data = json.loads(metrics_path.read_text())
        metrics = metrics_data.get("metrics", {})
        score = metrics.get(metric_name)

        if score is None:
            continue

        primary_score = _to_float(score)

        overfitting = metrics_data.get("overfitting_analysis", {})
        severity = overfitting.get("severity", "good")
        if severity in BLOCKED_SEVERITIES:
            blocked.append({
                "algorithm": algo,
                "severity": severity,
                "status": overfitting.get("status"),
            })
            continue

        composite_score = _compute_selection_score(metrics, severity)
        gaps = overfitting.get("gaps", {})
        recall_gap_pos = max(_to_float(gaps.get("recall", 0.0)), 0.0)
        f1_gap_pos = max(_to_float(gaps.get("f1", 0.0)), 0.0)

        severity_rank = SEVERITY_RANK.get(severity, 0)

        # Tie-breaker order: generalization severity > configured metric > composite > recall > smaller positive gaps > precision.
        rank = (
            severity_rank,
            round(primary_score, 6),
            round(composite_score, 6),
            round(_to_float(metrics.get("recall", 0.0)), 6),
            round(-recall_gap_pos, 6),
            round(-f1_gap_pos, 6),
            round(_to_float(metrics.get("precision", 0.0)), 6),
            algo,
        )

        evaluated += 1
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_model = algo
            best_context = {
                "algorithm": algo,
                "selection_metric": metric_name,
                "selection_metric_value": round(primary_score, 6),
                "composite_score": round(composite_score, 6),
                "severity": severity,
                "status": overfitting.get("status", "Unknown"),
            }

    if best_model is None:
        if blocked:
            raise RuntimeError(f"No safe trained model found for outcome '{outcome}' (all eligible models are blocked by overfitting severity)")
        raise RuntimeError("No valid trained model found")

    # update registry
    registry["active_model"] = best_model
    registry["selection_details"] = {
        "method": "composite_with_safety_gate_and_generalization_preference",
        "severity_priority": ["good", "warning"],
        "blocked_severities": sorted(BLOCKED_SEVERITIES),
        "evaluated_models": evaluated,
        "blocked_models": blocked,
        "selected": best_context,
    }
    registry_path.write_text(json.dumps(registry, indent=2))

    return best_model