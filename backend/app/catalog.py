from typing import Any, Dict, List, Optional, Callable
import json
from pathlib import Path

# Medical condition columns used for aggregate risk score in LBW feature engineering
_LBW_MEDICAL_CONDITIONS = [
    "ANEMIA", "CARDIAC", "ACLUNG", "DIABETES", "HERPES", "HYDRAM",
    "HEMOGLOB", "HYPERCH", "HYPERPR", "ECLAMP", "CERVIX", "PINFANT",
    "PRETERM", "RENAL", "RHSEN", "UTERINE",
]


def _engineer_lbw_features(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the engineered features for the LBW outcome that were created
    during training.  This must mirror exactly the feature-engineering cell
    in the training notebooks so the model receives the inputs it expects.
    """
    v = dict(values)  # shallow copy – don't mutate caller's dict

    def _num(key: str) -> float:
        """Safely cast a value to float, defaulting to 0."""
        try:
            return float(v.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    # (a) Interaction features between key medical conditions
    v.setdefault("HYPER_ECLAMP",   _num("HYPERPR") * _num("ECLAMP"))
    v.setdefault("HYPER_CHRONIC",  _num("HYPERCH") * _num("HYPERPR"))
    v.setdefault("DIABETES_HYPER", _num("DIABETES") * _num("HYPERPR"))
    v.setdefault("PRETERM_CERVIX", _num("PRETERM") * _num("CERVIX"))

    # (b) Aggregate medical risk score
    risk_score = sum(_num(c) for c in _LBW_MEDICAL_CONDITIONS)
    v.setdefault("MEDICAL_RISK_SCORE", risk_score)
    v.setdefault("HAS_ANY_CONDITION",  int(risk_score > 0))
    v.setdefault("HIGH_RISK",          int(risk_score >= 2))

    # (c) Polynomial / combined features
    v.setdefault("WEEKS_SQUARED",   _num("WEEKS") ** 2)
    v.setdefault("CIG_DRINK_COMBO", _num("CIGNUM") + _num("DRINKNUM"))

    return v


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
        "schema": {"title": "Intrauterine Growth Restriction Inputs", "sections": []}
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
                        {"name": "Occupation After Your Latest Childbirth", "label": "Occupation After Latest Childbirth ", "type": "select", "required": True, "options": [{"value": "Housewife", "label": "Housewife"},{"value": "Student", "label": "Student"},{"value": "Teacher", "label": "Teacher"},{"value": "Doctor", "label": "Doctor"},{"value": "Service", "label": "Service"},{"value": "Business", "label": "Business"},{"value": "Other", "label": "Other"}]},
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
        "feature_engineer": _engineer_lbw_features,
        "schema": {
            "title": "Low Birth Weight Inputs",
            "sections": [
                {
                    "fields": [
                        {"name": "SEX", "label": "Baby Sex", "type": "select", "required": True, "options": [{"value": 1, "label": "Male"}, {"value": 2, "label": "Female"}]},
                        {"name": "MARITAL", "label": "Mother's Marital Status", "type": "select", "required": True, "options": [{"value": 1, "label": "Married"}, {"value": 2, "label": "Unmarried"}]},
                        {"name": "MAGE", "label": "Mother's Age (years)", "type": "number", "min": 11, "max": 53, "required": True, "placeholder": "e.g., 28"},
                        {"name": "FAGE", "label": "Father's Age (years)", "type": "number", "min": 14, "max": 74, "required": True, "placeholder": "e.g., 30"},
                        {"name": "MEDUC", "label": "Mother's Education Level", "type": "select", "required": True, "options": [{"value": 0, "label": "No formal education"}, {"value": 5, "label": "Primary School"}, {"value": 10, "label": "Grade 6-10"}, {"value": 11, "label": "Passed O/L"}, {"value": 13, "label": "Passed A/L"}, {"value": 15, "label": "Diploma / Associate"}, {"value": 16, "label": "Degree"}, {"value": 17, "label": "Postgraduate / Doctorate"}]},
                        {"name": "FEDUC", "label": "Father's Education Level", "type": "select", "required": True, "options": [{"value": 0, "label": "No formal education"}, {"value": 5, "label": "Primary School"}, {"value": 10, "label": "Grade 6-10"}, {"value": 11, "label": "Passed O/L"}, {"value": 13, "label": "Passed A/L"}, {"value": 15, "label": "Diploma / Associate"}, {"value": 16, "label": "Degree"}, {"value": 17, "label": "Postgraduate / Doctorate"}]},
                        {"name": "GAINED", "label": "Weight Gained during Pregnancy (lbs)", "type": "number", "min": 0, "max": 98, "required": True, "placeholder": "e.g., 30"},
                        {"name": "VISITS", "label": "Total Number of Prenatal Visits", "type": "number", "min": 0, "max": 49, "required": True, "placeholder": "e.g., 12"},
                        {"name": "WEEKS", "label": "Gestation Period (weeks)", "type": "number", "min": 18, "max": 45, "required": True, "placeholder": "e.g., 38"},
                        {"name": "TOTALP", "label": "Total Previous Pregnancies", "type": "number", "min": 1, "max": 20, "required": True, "placeholder": "e.g., 1"},
                        {"name": "BDEAD", "label": "Previous Children Born Dead", "type": "number", "min": 0, "max": 6, "required": True, "placeholder": "e.g., 0"},
                        {"name": "TERMS", "label": "Previous Pregnancies Terminated", "type": "number", "min": 0, "max": 16, "required": True, "placeholder": "e.g., 0"},
                        {"name": "LOUTCOME", "label": "Months Since Last Pregnancy Outcome", "type": "number", "min": 1, "max": 9, "required": True, "placeholder": "e.g., 2"},
                        {"name": "RACEMOM", "label": "Mother's Race", "type": "select", "required": True, "options": [{"value": 1, "label": "White"}, {"value": 2, "label": "Black"}, {"value": 3, "label": "American Indian"}, {"value": 4, "label": "Chinese"}, {"value": 5, "label": "Japanese"}, {"value": 6, "label": "Hawaiian"}, {"value": 7, "label": "Filipino"}, {"value": 8, "label": "Other Asian/Pacific Islander"}]},
                        {"name": "RACEDAD", "label": "Father's Race", "type": "select", "required": True, "options": [{"value": 1, "label": "White"}, {"value": 2, "label": "Black"}, {"value": 3, "label": "American Indian"}, {"value": 4, "label": "Chinese"}, {"value": 5, "label": "Japanese"}, {"value": 6, "label": "Hawaiian"}, {"value": 7, "label": "Filipino"}, {"value": 8, "label": "Other Asian/Pacific Islander"}]},
                        {"name": "HISPMOM", "label": "Mother's Hispanic Origin", "type": "select", "required": True, "options": [{"value": "M", "label": "Mexican"}, {"value": "P", "label": "Puerto Rican"}, {"value": "C", "label": "Cuban"}, {"value": "O", "label": "Other Hispanic"}, {"value": "N", "label": "Non-Hispanic"}, {"value": "U", "label": "Unknown"}]},
                        {"name": "HISPDAD", "label": "Father's Hispanic Origin", "type": "select", "required": True, "options": [{"value": "M", "label": "Mexican"}, {"value": "P", "label": "Puerto Rican"}, {"value": "C", "label": "Cuban"}, {"value": "O", "label": "Other Hispanic"}, {"value": "N", "label": "Non-Hispanic"}, {"value": "U", "label": "Unknown"}]},
                        {"name": "CIGNUM", "label": "Cigarettes Smoked Per Day", "type": "number", "min": 0, "max": 60, "required": True, "placeholder": "e.g., 0"},
                        {"name": "DRINKNUM", "label": "Alcoholic Drinks Per Day", "type": "number", "min": 0, "max": 40, "required": True, "placeholder": "e.g., 0"},
                        {"name": "ANEMIA", "label": "Mother Has Anemia", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "CARDIAC", "label": "Mother Has Cardiac Disease", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "ACLUNG", "label": "Mother Has Acute or Chronic Lung Disease", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "DIABETES", "label": "Mother Has Diabetes", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HERPES", "label": "Mother Has Genital Herpes", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HYDRAM", "label": "Hydramnios/Oligohydramnios (Amniotic fluid disorder)", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HEMOGLOB", "label": "Hemoglobinopathy (Blood disorder)", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HYPERCH", "label": "Chronic Hypertension", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "HYPERPR", "label": "Pregnancy-associated Hypertension", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "ECLAMP", "label": "Eclampsia", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "CERVIX", "label": "Incompetent Cervix", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "PINFANT", "label": "Previous Infant Born 4000+ grams", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "PRETERM", "label": "Previous Preterm or Small-for-Gestational-Age Infant", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "RENAL", "label": "Mother Has Renal (Kidney) Disease", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "RHSEN", "label": "Rh Sensitization", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
                        {"name": "UTERINE", "label": "Uterine Bleeding during Pregnancy", "type": "select", "required": True, "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]}
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
                        {"name": "GEST", "label": "Gestation Period at Admission (weeks)", "type": "number", "min": 20, "max": 35, "required": True, "placeholder": "e.g., 30"},
                        {"name": "DILATE", "label": "Cervical Dilation (cm)", "type": "number", "min": 0, "max": 8, "required": True, "placeholder": "e.g., 2"},
                        {"name": "EFFACE", "label": "Cervical Effacement (%)", "type": "number", "min": 0, "max": 100, "required": True, "placeholder": "e.g., 50"},
                        {"name": "CONSIS", "label": "Cervical Consistency", "type": "select", "required": True, "options": [{"value": 1, "label": "Firm"}, {"value": 2, "label": "Medium"}, {"value": 3, "label": "Soft"}]},
                        {"name": "CONTR", "label": "Uterine Contractions", "type": "select", "required": True, "options": [{"value": 1, "label": "None"}, {"value": 2, "label": "Irregular / Mild"}, {"value": 3, "label": "Regular / Strong"}]},
                        {"name": "MEMBRAN", "label": "Amniotic Membranes Status", "type": "select", "required": True, "options": [{"value": 1, "label": "Intact"}, {"value": 2, "label": "Ruptured"}, {"value": 3, "label": "Unknown"}]},
                        {"name": "AGE", "label": "Mother's Age (years)", "type": "number", "min": 15, "max": 42, "required": True, "placeholder": "e.g., 28"},
                        {"name": "STRAT", "label": "Socio-economic Stratum", "type": "select", "required": True, "options": [{"value": 1, "label": "1 (Low)"}, {"value": 2, "label": "2 (Lower-Middle)"}, {"value": 3, "label": "3 (Upper-Middle)"}, {"value": 4, "label": "4 (High)"}]},
                        {"name": "GRAVID", "label": "Gravidity (Total Number of Pregnancies)", "type": "number", "min": 0, "max": 13, "required": True, "placeholder": "e.g., 1"},
                        {"name": "PARIT", "label": "Parity (Number of Previous Births)", "type": "number", "min": 0, "max": 7, "required": True, "placeholder": "e.g., 0"},
                        {"name": "DIAB", "label": "Mother has Diabetes?", "type": "select", "required": True, "options": [{"value": 1, "label": "Yes"}, {"value": 2, "label": "No"}, {"value": 9, "label": "Unknown"}]},
                        {"name": "TRANSF", "label": "Patient Transferred from Another Facility?", "type": "select", "required": True, "options": [{"value": 1, "label": "Yes"}, {"value": 2, "label": "No"}]},
                        {"name": "GEMEL", "label": "Is this a Multiple Pregnancy? (Twins, etc.)", "type": "select", "required": True, "options": [{"value": 1, "label": "Singleton"}, {"value": 2, "label": "Multiple (Twins/Triplets+)"}]}
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


def get_feature_engineer(outcome_id: str) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    outcome = OUTCOMES.get(outcome_id)
    if not outcome:
        return None
    return outcome.get("feature_engineer")



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

        overfitting = metrics_data.get("overfitting_analysis", {})
        severity = overfitting.get("severity", "good")
        if severity in BLOCKED_SEVERITIES:
            blocked.append({
                "algorithm": algo,
                "severity": severity,
                "status": overfitting.get("status"),
            })
            continue

        # Clinical priority metric hierarchy: 
        # 1. Recall (Sensitivity)
        # 2. F1-score
        # 3. ROC-AUC
        # 4. Precision
        # 5. Accuracy
        recall = _to_float(metrics.get("recall", 0.0))
        f1 = _to_float(metrics.get("f1", 0.0))
        roc_auc = _to_float(metrics.get("roc_auc", 0.0))
        precision = _to_float(metrics.get("precision", 0.0))
        accuracy = _to_float(metrics.get("accuracy", 0.0))

        # We also prefer models that are well-generalized (good > warning)
        severity_rank = SEVERITY_RANK.get(severity, 0)

        rank = (
            severity_rank,
            round(recall, 6),
            round(f1, 6),
            round(roc_auc, 6),
            round(precision, 6),
            round(accuracy, 6),
            algo,
        )

        evaluated += 1
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_model = algo
            best_context = {
                "algorithm": algo,
                "selection_metric": "recall",
                "selection_metric_value": round(recall, 6),
                "f1": round(f1, 6),
                "roc_auc": round(roc_auc, 6),
                "precision": round(precision, 6),
                "accuracy": round(accuracy, 6),
                "severity": severity,
                "status": overfitting.get("status", "Unknown"),
            }

    if best_model is None:
        if blocked:
            raise RuntimeError(f"No safe trained model found for outcome '{outcome}' (all eligible models are blocked by overfitting severity)")
        raise RuntimeError("No valid trained model found")

    # update registry
    registry["active_model"] = best_model
    registry["selection_metric"] = "recall"
    registry["selection_details"] = {
        "method": "clinical_priority_hierarchy_with_safety_gate",
        "severity_priority": ["good", "warning"],
        "blocked_severities": sorted(BLOCKED_SEVERITIES),
        "evaluated_models": evaluated,
        "blocked_models": blocked,
        "selected": best_context,
    }
    registry_path.write_text(json.dumps(registry, indent=2))

    return best_model