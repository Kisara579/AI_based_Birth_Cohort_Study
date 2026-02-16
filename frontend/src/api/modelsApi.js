import {apiGet,apiPost} from "./client.js";

export function fetchOutcomes() {
    return apiGet("/outcomes");
}


export function fetchSchema(outcomeId){
    return apiGet(`/schema/${outcomeId}`);
}


export function predictOutcome(outcomeId, formValues) {
  return apiPost(`/predict/${outcomeId}`, { data: formValues });
}
