/**
 * Maps the universal form data → model-specific payloads.
 *
 * Each model has its own variable names.  This file centralises the
 * mapping so the UI only deals with human-readable universal keys.
 *
 * ===================================================================
 * TEAMMATES: When you add a new model, create a new function like
 *   buildLbwPayload / buildPdPayload below and register it in
 *   MODEL_BUILDERS at the bottom of this file.
 * ===================================================================
 */

// ─── LBW (Low Birth Weight) ────────────────────────────────────────
function buildLbwPayload(u) {
  return {
    MAGER:      u.motherAge,
    FAGECOMB:   u.fatherAge,
    MEDUC:      Number(u.motherEducation),
    FEDUC:      Number(u.fatherEducation),
    DMAR:       u.maritalStatus === "married" ? "1" : "2",
    MBSTATE_REC: u.motherNativity,
    SEX:        u.babySex === "male" ? "M" : "F",
    // Maternal health
    BMI:        u.bmi,
    PWgt_R:     u.prePregnancyWeight,
    M_Ht_In:    u.motherHeight,
    WTGAIN:     u.weightGained,
    CIG_0:      u.cigarettesPerDay,
    NO_INFEC:   Number(u.noInfections),
    NO_MMORB:   Number(u.noMaternalMorbidity),
    NO_RISKS:   Number(u.noRiskFactors),
    RF_CESAR:   u.previousCesarean === "yes" ? "Y" : "N",
    RF_CESARN:  u.previousCesareanCount,
    // Prenatal care
    PRECARE:    u.prenatalCareMonth,
    PREVIS:     u.prenatalVisits,
    ATTEND:     u.attendantAtBirth,
    BFACIL:     u.birthFacility,
    LD_INDL:    u.inductionOfLabor === "yes" ? "Y" : "N",
    // Pregnancy history
    PRIORLIVE:  u.priorLiveBirths,
    PRIORDEAD:  u.priorDeadBirths,
    PRIORTERM:  u.priorTerminations,
    ILLB_R:     u.intervalLastLiveBirth,
    ILOP_R:     u.intervalLastPregnancyOutcome,
    ILP_R:      u.intervalLastPregnancy,
    // Admin
    PAY:        u.paymentSource,
    PAY_REC:    u.paymentRecode,
    RESTATUS:   u.residenceStatus,
  };
}

// ─── PD (Premature Delivery) ───────────────────────────────────────
function buildPdPayload(u) {
  return {
    SEX:      u.babySex === "male" ? 1 : 2,
    MARITAL:  u.maritalStatus === "married" ? 1 : 2,
    FAGE:     u.fatherAge,
    MAGE:     u.motherAge,
    FEDUC:    Number(u.fatherEducation),
    MEDUC:    Number(u.motherEducation),
    RACEMOM:  Number(u.motherRace),
    RACEDAD:  Number(u.fatherRace),
    HISPMOM:  u.motherHispanic,
    HISPDAD:  u.fatherHispanic,
    // Pregnancy & history
    GAINED:   u.weightGained,
    VISITS:   u.prenatalVisits,
    TOTALP:   u.totalPregnancies,
    BDEAD:    u.priorDeadBirths,
    TERMS:    u.priorTerminations,
    LOUTCOME: u.lastDeliveryOutcome,
    PRETERM:  Number(u.previousPreterm),
    PINFANT:  Number(u.previousLargeInfant),
    // Lifestyle
    CIGNUM:   u.cigarettesPerDay,
    DRINKNUM: u.drinksPerDay,
    // Conditions
    ANEMIA:   Number(u.anemia),
    CARDIAC:  Number(u.cardiac),
    ACLUNG:   Number(u.lungDisease),
    DIABETES: Number(u.diabetes),
    HERPES:   Number(u.herpes),
    HYDRAM:   Number(u.hydramnios),
    HEMOGLOB: Number(u.hemoglobinopathy),
    HYPERCH:  Number(u.chronicHypertension),
    HYPERPR:  Number(u.pregnancyHypertension),
    ECLAMP:   Number(u.eclampsia),
    CERVIX:   Number(u.incompetentCervix),
    RENAL:    Number(u.renalDisease),
    RHSEN:    Number(u.rhSensitization),
    UTERINE:  Number(u.uterineBleeding),
  };
}

// ─── Registry of model builders ────────────────────────────────────
// Key = outcome id as returned by /outcomes
// Value = function(universalData) → payload object
//
// TEAMMATES: register your builder here once you create it above.
const MODEL_BUILDERS = {
  lbw: buildLbwPayload,
  pd:  buildPdPayload,
  // pph:          buildPphPayload,
  // preeclampsia: buildPreeclampsiaPayload,
  // iugr:         buildIugrPayload,
  // gdm:          buildGdmPayload,
  // iol:          buildIolPayload,
  // c_section:    buildCSectionPayload,
  // postpartum:   buildPostpartumPayload,
  // nicu:         buildNicuPayload,
  // feeding:      buildFeedingPayload,
};

/**
 * Build a per-model payload map.
 *
 * @param {object[]} outcomes  – array from /outcomes  [{id, name, …}, …]
 * @param {object}   universal – form values keyed by universal field name
 * @returns {object} { outcomeId: payloadObj, … }
 */
export function buildAllPayloads(outcomes, universal) {
  const result = {};
  for (const o of outcomes) {
    const builder = MODEL_BUILDERS[o.id];
    if (builder) {
      result[o.id] = builder(universal);
    }
    // If no builder registered yet, the outcome simply won't get a payload
    // and the dashboard will show "Model not configured yet" gracefully.
  }
  return result;
}
