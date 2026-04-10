/**
 * Universal Patient Assessment Schema
 * 
 * This merges ALL fields from the LBW and PD models into a single unified form.
 * Fields that are conceptually the same (e.g. mother age) appear ONCE under a
 * single universal key name.  The predictionMapper.js file then maps these
 * universal names to each model's specific expected variable name.
 * 
 * ===================================================================
 * TEAMMATES: When adding a new model, if it needs a field that's not
 * already here, add a new entry to the correct section below.
 * ===================================================================
 */

export const universalSchema = {
  title: "Universal Patient Assessment",
  sections: [
    /* ------------------------------------------------------------------ */
    /*  SECTION 1 — Demographics                                          */
    /* ------------------------------------------------------------------ */
    {
      id: "demographics",
      title: "Demographics",
      fields: [
        { name: "motherAge", label: "Mother's age (years)", type: "number", min: 10, max: 60, required: true, placeholder: "e.g., 28" },
        { name: "fatherAge", label: "Father's age (years)", type: "number", min: 10, max: 90, required: true, placeholder: "e.g., 30" },
        { name: "maritalStatus", label: "Marital status", type: "select", required: true, options: [
          { value: "married", label: "Married" },
          { value: "unmarried", label: "Not Married / Unmarried" }
        ]},
        { name: "motherEducation", label: "Mother's education level", type: "select", required: true, options: [
          { value: 1, label: "No formal education / Primary" },
          { value: 2, label: "Grade 6–11" },
          { value: 3, label: "Passed O/L" },
          { value: 4, label: "Passed A/L" },
          { value: 5, label: "Diploma / Associate" },
          { value: 6, label: "Degree" },
          { value: 7, label: "Postgraduate" },
          { value: 8, label: "Doctorate" },
          { value: 9, label: "Unknown" }
        ]},
        { name: "fatherEducation", label: "Father's education level", type: "select", required: true, options: [
          { value: 1, label: "No formal education / Primary" },
          { value: 2, label: "Grade 6–11" },
          { value: 3, label: "Passed O/L" },
          { value: 4, label: "Passed A/L" },
          { value: 5, label: "Diploma / Associate" },
          { value: 6, label: "Degree" },
          { value: 7, label: "Postgraduate" },
          { value: 8, label: "Doctorate" },
          { value: 9, label: "Unknown" }
        ]},
        { name: "babySex", label: "Sex of baby", type: "select", required: true, options: [
          { value: "male", label: "Male" },
          { value: "female", label: "Female" }
        ]},
        { name: "motherNativity", label: "Mother nativity (coded)", type: "number", required: true, placeholder: "e.g., 1" },
        { name: "motherRace", label: "Mother's race", type: "select", required: true, options: [
          { value: 0, label: "Unknown" },
          { value: 1, label: "Other Non-White" },
          { value: 2, label: "White" },
          { value: 3, label: "Black" },
          { value: 4, label: "American Indian" },
          { value: 5, label: "Chinese" },
          { value: 6, label: "Japanese" },
          { value: 7, label: "Hawaiian" },
          { value: 8, label: "Filipino" },
          { value: 9, label: "Other Asian" }
        ]},
        { name: "fatherRace", label: "Father's race", type: "select", required: true, options: [
          { value: 0, label: "Unknown" },
          { value: 1, label: "Other Non-White" },
          { value: 2, label: "White" },
          { value: 3, label: "Black" },
          { value: 4, label: "American Indian" },
          { value: 5, label: "Chinese" },
          { value: 6, label: "Japanese" },
          { value: 7, label: "Hawaiian" },
          { value: 8, label: "Filipino" },
          { value: 9, label: "Other Asian" }
        ]},
        { name: "motherHispanic", label: "Mother Hispanic group", type: "select", required: true, options: [
          { value: "C", label: "Cuban" },
          { value: "M", label: "Mexican" },
          { value: "N", label: "No" },
          { value: "O", label: "Colombian" },
          { value: "P", label: "Peruvian" },
          { value: "S", label: "Salvadoran" },
          { value: "U", label: "Guatemalan" }
        ]},
        { name: "fatherHispanic", label: "Father Hispanic group", type: "select", required: true, options: [
          { value: "C", label: "Cuban" },
          { value: "M", label: "Mexican" },
          { value: "N", label: "No" },
          { value: "O", label: "Colombian" },
          { value: "P", label: "Peruvian" },
          { value: "S", label: "Salvadoran" },
          { value: "U", label: "Guatemalan" }
        ]}
      ]
    },

    /* ------------------------------------------------------------------ */
    /*  SECTION 2 — Maternal Health                                       */
    /* ------------------------------------------------------------------ */
    {
      id: "maternal_health",
      title: "Maternal Health",
      fields: [
        { name: "bmi", label: "Body Mass Index (BMI)", type: "number", min: 10, max: 80, required: true, placeholder: "e.g., 23.4" },
        { name: "prePregnancyWeight", label: "Pre-pregnancy weight (coded)", type: "number", required: true, placeholder: "e.g., 120" },
        { name: "motherHeight", label: "Mother's height (inches)", type: "number", min: 40, max: 84, required: true, placeholder: "e.g., 62" },
        { name: "weightGained", label: "Weight gained during pregnancy (lbs)", type: "number", min: 0, max: 100, required: true, placeholder: "e.g., 25" },
        { name: "noInfections", label: "No infections reported (0 = has infections, 1 = none)", type: "select", required: true, options: [
          { value: 0, label: "Has infections" },
          { value: 1, label: "No infections" }
        ]},
        { name: "noMaternalMorbidity", label: "No maternal morbidity reported", type: "select", required: true, options: [
          { value: 0, label: "Has morbidity" },
          { value: 1, label: "No morbidity" }
        ]},
        { name: "noRiskFactors", label: "No risk factors reported", type: "select", required: true, options: [
          { value: 0, label: "Has risk factors" },
          { value: 1, label: "No risk factors" }
        ]}
      ]
    },

    /* ------------------------------------------------------------------ */
    /*  SECTION 3 — Lifestyle                                             */
    /* ------------------------------------------------------------------ */
    {
      id: "lifestyle",
      title: "Lifestyle Factors",
      fields: [
        { name: "cigarettesPerDay", label: "Cigarettes per day (average)", type: "number", min: 0, max: 60, required: true, placeholder: "e.g., 0" },
        { name: "drinksPerDay", label: "Drinks per day (average)", type: "number", min: 0, max: 20, required: true, placeholder: "e.g., 0" }
      ]
    },

    /* ------------------------------------------------------------------ */
    /*  SECTION 4 — Prenatal Care & Delivery                              */
    /* ------------------------------------------------------------------ */
    {
      id: "prenatal_care",
      title: "Prenatal Care & Delivery",
      fields: [
        { name: "prenatalCareMonth", label: "Month prenatal care began", type: "number", min: 0, max: 9, required: true, placeholder: "e.g., 2" },
        { name: "prenatalVisits", label: "Number of prenatal visits", type: "number", min: 0, max: 50, required: true, placeholder: "e.g., 10" },
        { name: "attendantAtBirth", label: "Attendant at birth (coded)", type: "number", required: true, placeholder: "e.g., 1" },
        { name: "birthFacility", label: "Birth facility / place (coded)", type: "number", required: true, placeholder: "e.g., 1" },
        { name: "inductionOfLabor", label: "Induction of labor", type: "select", required: true, options: [
          { value: "yes", label: "Yes" },
          { value: "no", label: "No" }
        ]},
        { name: "previousCesarean", label: "Previous cesarean", type: "select", required: true, options: [
          { value: "yes", label: "Yes" },
          { value: "no", label: "No" }
        ]},
        { name: "previousCesareanCount", label: "Number of previous cesareans", type: "number", min: 0, max: 10, required: true, placeholder: "e.g., 0" }
      ]
    },

    /* ------------------------------------------------------------------ */
    /*  SECTION 5 — Pregnancy History                                     */
    /* ------------------------------------------------------------------ */
    {
      id: "pregnancy_history",
      title: "Pregnancy History",
      fields: [
        { name: "totalPregnancies", label: "Total pregnancies", type: "number", min: 0, max: 20, required: true, placeholder: "e.g., 2" },
        { name: "priorLiveBirths", label: "Prior births now living", type: "number", min: 0, max: 20, required: true, placeholder: "e.g., 0" },
        { name: "priorDeadBirths", label: "Prior births now dead", type: "number", min: 0, max: 20, required: true, placeholder: "e.g., 0" },
        { name: "priorTerminations", label: "Other terminations", type: "number", min: 0, max: 20, required: true, placeholder: "e.g., 0" },
        { name: "intervalLastLiveBirth", label: "Interval since last live birth (coded)", type: "number", required: true, placeholder: "e.g., 0" },
        { name: "intervalLastPregnancyOutcome", label: "Interval since last pregnancy outcome (coded)", type: "number", required: true, placeholder: "e.g., 0" },
        { name: "intervalLastPregnancy", label: "Interval since last pregnancy (coded)", type: "number", required: true, placeholder: "e.g., 0" },
        { name: "lastDeliveryOutcome", label: "Outcome of last delivery (coded)", type: "number", min: 0, max: 10, required: true, placeholder: "e.g., 1" },
        { name: "previousPreterm", label: "Previous preterm or small infant", type: "select", required: true, options: [
          { value: 0, label: "No" },
          { value: 1, label: "Yes" }
        ]},
        { name: "previousLargeInfant", label: "Previous infant over 4000g", type: "select", required: true, options: [
          { value: 0, label: "No" },
          { value: 1, label: "Yes" }
        ]}
      ]
    },

    /* ------------------------------------------------------------------ */
    /*  SECTION 6 — Maternal Conditions                                   */
    /* ------------------------------------------------------------------ */
    {
      id: "maternal_conditions",
      title: "Maternal Conditions",
      fields: [
        { name: "anemia", label: "Anemia", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "cardiac", label: "Cardiac disease", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "lungDisease", label: "Acute or chronic lung disease", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "diabetes", label: "Diabetes", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "herpes", label: "Genital herpes", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "hydramnios", label: "Hydramnios / Oligohydramnios", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "hemoglobinopathy", label: "Hemoglobinopathy", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "chronicHypertension", label: "Chronic hypertension", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "pregnancyHypertension", label: "Pregnancy hypertension", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "eclampsia", label: "Eclampsia", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "incompetentCervix", label: "Incompetent cervix", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "renalDisease", label: "Renal disease", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "rhSensitization", label: "Rh sensitization", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
        { name: "uterineBleeding", label: "Uterine bleeding", type: "select", required: true, options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] }
      ]
    },

    /* ------------------------------------------------------------------ */
    /*  SECTION 7 — Administrative / Coverage                             */
    /* ------------------------------------------------------------------ */
    {
      id: "admin",
      title: "Administrative / Coverage",
      fields: [
        { name: "paymentSource", label: "Payment source for delivery (coded)", type: "number", required: true, placeholder: "e.g., 1" },
        { name: "paymentRecode", label: "Payment recode (coded)", type: "number", required: true, placeholder: "e.g., 1" },
        { name: "residenceStatus", label: "Residence status (coded)", type: "number", required: true, placeholder: "e.g., 1" }
      ]
    }
  ]
};


/* ====================================================================
 * DEMO PATIENT PROFILES — for presentation autofill
 * ====================================================================
 * These contain every key from the universal schema so every field
 * gets populated when a demo button is clicked.
 * ==================================================================== */

export const demoPatient1 = {
  // Demographics
  motherAge: 25,
  fatherAge: 28,
  maritalStatus: "married",
  motherEducation: 6,
  fatherEducation: 6,
  babySex: "female",
  motherNativity: 1,
  motherRace: 2,
  fatherRace: 2,
  motherHispanic: "N",
  fatherHispanic: "N",
  // Maternal Health
  bmi: 22.0,
  prePregnancyWeight: 130,
  motherHeight: 64,
  weightGained: 28,
  noInfections: 1,
  noMaternalMorbidity: 1,
  noRiskFactors: 1,
  // Lifestyle
  cigarettesPerDay: 0,
  drinksPerDay: 0,
  // Prenatal Care
  prenatalCareMonth: 2,
  prenatalVisits: 12,
  attendantAtBirth: 1,
  birthFacility: 1,
  inductionOfLabor: "no",
  previousCesarean: "no",
  previousCesareanCount: 0,
  // Pregnancy History
  totalPregnancies: 1,
  priorLiveBirths: 0,
  priorDeadBirths: 0,
  priorTerminations: 0,
  intervalLastLiveBirth: 0,
  intervalLastPregnancyOutcome: 0,
  intervalLastPregnancy: 0,
  lastDeliveryOutcome: 1,
  previousPreterm: 0,
  previousLargeInfant: 0,
  // Maternal Conditions
  anemia: 0,
  cardiac: 0,
  lungDisease: 0,
  diabetes: 0,
  herpes: 0,
  hydramnios: 0,
  hemoglobinopathy: 0,
  chronicHypertension: 0,
  pregnancyHypertension: 0,
  eclampsia: 0,
  incompetentCervix: 0,
  renalDisease: 0,
  rhSensitization: 0,
  uterineBleeding: 0,
  // Admin
  paymentSource: 1,
  paymentRecode: 1,
  residenceStatus: 1
};

export const demoPatient2 = {
  // Demographics
  motherAge: 39,
  fatherAge: 44,
  maritalStatus: "unmarried",
  motherEducation: 2,
  fatherEducation: 2,
  babySex: "male",
  motherNativity: 1,
  motherRace: 3,
  fatherRace: 3,
  motherHispanic: "N",
  fatherHispanic: "N",
  // Maternal Health
  bmi: 35.2,
  prePregnancyWeight: 210,
  motherHeight: 60,
  weightGained: 8,
  noInfections: 0,
  noMaternalMorbidity: 0,
  noRiskFactors: 0,
  // Lifestyle
  cigarettesPerDay: 12,
  drinksPerDay: 3,
  // Prenatal Care
  prenatalCareMonth: 6,
  prenatalVisits: 3,
  attendantAtBirth: 1,
  birthFacility: 1,
  inductionOfLabor: "yes",
  previousCesarean: "yes",
  previousCesareanCount: 2,
  // Pregnancy History
  totalPregnancies: 5,
  priorLiveBirths: 2,
  priorDeadBirths: 1,
  priorTerminations: 2,
  intervalLastLiveBirth: 888,
  intervalLastPregnancyOutcome: 888,
  intervalLastPregnancy: 888,
  lastDeliveryOutcome: 1,
  previousPreterm: 1,
  previousLargeInfant: 0,
  // Maternal Conditions
  anemia: 1,
  cardiac: 0,
  lungDisease: 0,
  diabetes: 1,
  herpes: 0,
  hydramnios: 1,
  hemoglobinopathy: 0,
  chronicHypertension: 1,
  pregnancyHypertension: 1,
  eclampsia: 0,
  incompetentCervix: 1,
  renalDisease: 0,
  rhSensitization: 0,
  uterineBleeding: 1,
  // Admin
  paymentSource: 2,
  paymentRecode: 2,
  residenceStatus: 1
};
