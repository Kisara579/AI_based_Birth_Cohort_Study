import { apiGet, apiPost } from "./client.js";

export function fetchOutcomes() {
  return apiGet("/outcomes");
}

export function fetchSchema(outcomeId) {
  return apiGet(`/schema/${outcomeId}`);
}

export function fetchModels(outcomeId) {
  return apiGet(`/models/${outcomeId}`);
}

export function fetchMetrics(outcomeId, model) {
  const qs = model ? `?model=${encodeURIComponent(model)}` : "";
  return apiGet(`/metrics/${outcomeId}${qs}`);
}

export function predictOutcome(outcomeId, formValues, model) {
  const qs = model ? `?model=${encodeURIComponent(model)}` : "";
  return apiPost(`/predict/${outcomeId}${qs}`, { data: formValues });
}

export async function predictAllOutcomes(allOutcomes, universalPayload) {
  const promises = allOutcomes.map((outcome) => {
    return predictOutcome(outcome.id, universalPayload[outcome.id])
      .then((res) => ({ id: outcome.id, name: outcome.name, success: true, ...res }))
      .catch((err) => ({ id: outcome.id, name: outcome.name, success: false, error: err.message }));
  });
  return Promise.all(promises);
}
