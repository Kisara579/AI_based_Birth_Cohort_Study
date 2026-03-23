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