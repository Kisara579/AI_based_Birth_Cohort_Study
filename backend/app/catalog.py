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
        "description": "Predict risk of low birth weight based on maternal and delivery factors.",
        "model_key": "lbw",
        "schema": {
            "title": "Low Birth Weight Inputs",
            "sections": [
                {
                    "id": "demographics",
                    "title": "Demographics",
                    "fields": [
                        {"name": "MAGER", "label": "Mother age (years)", "type": "number", "min": 10, "max": 60, "required": True, "placeholder": "e.g., 28"},
                        {"name": "FAGECOMB", "label": "Father age (years)", "type": "number", "min": 10, "max": 80, "required": True, "placeholder": "e.g., 30"},
                        {"name": "MEDUC", "label": "Mother education level", "type": "select", "required": True, "options": [{"value": 1, "label": "No formal education / Primary"},{"value": 2, "label": "Grade 6–11"},{"value": 3, "label": "Passed O/L"},{"value": 4, "label": "Passed A/L"},{"value": 5, "label": "Diploma / Associate"},{"value": 6, "label": "Degree"},{"value": 7, "label": "Postgraduate"},{"value": 8, "label": "Doctorate"},{"value": 9, "label": "Unknown"}]},
                        {"name": "FEDUC", "label": "Father education level", "type": "select", "required": True, "options": [{"value": 1, "label": "No formal education / Primary"},{"value": 2, "label": "Grade 6–11"},{"value": 3, "label": "Passed O/L"},{"value": 4, "label": "Passed A/L"},{"value": 5, "label": "Diploma / Associate"},{"value": 6, "label": "Degree"},{"value": 7, "label": "Postgraduate"},{"value": 8, "label": "Doctorate"},{"value": 9, "label": "Unknown"}]},
                        {"name": "DMAR","label": "Marital Status","type": "select","required": True,"options": [{"value": "1", "label": "Married"},{"value": "2", "label": "Not Married"},{"value": "", "label": "Unknown"}]},
                        {"name": "MBSTATE_REC", "label": "Mother nativity (coded)", "type": "number", "required": True, "placeholder": "e.g., 1"}
                    ],
                },
                {
                    "id": "maternal_health",
                    "title": "Maternal Health",
                    "fields": [
                        {"name": "BMI", "label": "Body Mass Index (BMI)", "type": "number", "min": 10, "max": 80, "required": True, "placeholder": "e.g., 23.4"},
                        {"name": "PWgt_R", "label": "Pre-pregnancy weight (coded)", "type": "number", "required": True, "placeholder": "e.g., 120"},
                        {"name": "M_Ht_In", "label": "Mother height (inches)", "type": "number", "min": 40, "max": 84, "required": True, "placeholder": "e.g., 62"},
                        {"name": "WTGAIN", "label": "Weight gain during pregnancy (lbs)", "type": "number", "min": 0, "max": 100, "required": True, "placeholder": "e.g., 25"},
                        {"name": "CIG_0", "label": "Cigarettes before pregnancy (per day)", "type": "number", "min": 0, "max": 60, "required": True, "placeholder": "e.g., 0"},
                        {"name": "NO_INFEC", "label": "No infections reported (coded)", "type": "number", "required": True, "placeholder": "e.g., 0/1"},
                        {"name": "NO_MMORB", "label": "No maternal morbidity reported (coded)", "type": "number", "required": True, "placeholder": "e.g., 0/1"},
                        {"name": "NO_RISKS", "label": "No risk factors reported (coded)", "type": "number", "required": True, "placeholder": "e.g., 0/1"},
                        {"name":"RF_CESAR","label":"Previous cesarean","type":"select","required":True,"options":[{"value":"Y","label":"Yes"},{"value":"N","label":"No"}]},

                        {"name": "RF_CESARN", "label": "Number of previous cesareans", "type": "number", "min": 0, "max": 10, "required": True, "placeholder": "e.g., 0"}
                    ],
                },
                {
                    "id": "prenatal_care",
                    "title": "Prenatal Care",
                    "fields": [
                        {"name": "PRECARE", "label": "Month prenatal care began (coded)", "type": "number", "min": 0, "max": 9, "required": True, "placeholder": "e.g., 2"},
                        {"name": "PREVIS", "label": "Number of prenatal visits", "type": "number", "min": 0, "max": 50, "required": True, "placeholder": "e.g., 10"},
                        {"name": "ATTEND", "label": "Attendant at birth (coded)", "type": "number", "required": True, "placeholder": "e.g., 1"},
                        {"name": "BFACIL", "label": "Birth facility / place (coded)", "type": "number", "required": True, "placeholder": "e.g., 1"},
                        {"name":"LD_INDL","label":"Induction of labor","type":"select","required":True,"options":[{"value":"Y","label":"Yes"},{"value":"N","label":"No"}]}

                    ],
                },
                {
                    "id": "history",
                    "title": "Pregnancy History",
                    "fields": [
                        {"name": "PRIORLIVE", "label": "Prior births now living", "type": "number", "min": 0, "max": 20, "required": True, "placeholder": "e.g., 0"},
                        {"name": "PRIORDEAD", "label": "Prior births now dead", "type": "number", "min": 0, "max": 20, "required": True, "placeholder": "e.g., 0"},
                        {"name": "PRIORTERM", "label": "Prior other terminations", "type": "number", "min": 0, "max": 20, "required": True, "placeholder": "e.g., 0"},
                        {"name": "ILLB_R", "label": "Interval since last live birth (code)", "type": "number", "required": True, "placeholder": "e.g., 0"},
                        {"name": "ILOP_R", "label": "Interval since last other pregnancy outcome (coded)", "type": "number", "required": True, "placeholder": "e.g., 0"},
                        {"name": "ILP_R", "label": "Interval since last pregnancy (coded)", "type": "number", "required": True, "placeholder": "e.g., 0"}
                    ],
                },
                {
                    "id": "admin",
                    "title": "Administrative / Coverage",
                    "fields": [
                        {"name": "PAY", "label": "Payment source for delivery (coded)", "type": "number", "required": True, "placeholder": "e.g., 1"},
                        {"name": "PAY_REC", "label": "Payment recode (coded)", "type": "number", "required": True, "placeholder": "e.g., 1"},
                        {"name": "RESTATUS", "label": "Residence status (coded)", "type": "number", "required": True, "placeholder": "e.g., 1"}
                    ],
                },
                {
                    "id": "baby",
                    "title": "child Information",
                    "fields": [
                        {"name": "SEX", "label": "sex", "type": "select", "required": True, "options": [{"value":"M","label":"Male"}, {"value":"F","label":"Female"}]}
                    ],
                }
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

