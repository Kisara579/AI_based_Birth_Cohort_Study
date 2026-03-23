const API_BASE =
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function extractDetail(rawText) {
  if (!rawText) return "";

  try {
    const parsed = JSON.parse(rawText);
    if (parsed?.detail) return String(parsed.detail);
    if (parsed?.error) return String(parsed.error);
    return rawText;
  } catch {
    return rawText;
  }
}

function friendlyErrorMessage(method, path, status, detail) {
  const text = (detail || "").toLowerCase();

  if (text.includes("metrics not found")) {
    return "Metrics are not available for this outcome yet. Train the model first.";
  }

  if (text.includes("model not found")) {
    return "Prediction model is not available for this outcome yet. Train the model first.";
  }

  if (text.includes("schema not found")) {
    return "Input form is not available for this outcome yet.";
  }

  if (text.includes("unknown outcome")) {
    return "This outcome is not configured in the system yet.";
  }

  if (text.includes("missing required features")) {
    return "Some required input values are missing. Please review the form and try again.";
  }

  if (status >= 500) {
    return "Something went wrong on the server. Please try again.";
  }

  if (status === 404) {
    return "Requested data was not found.";
  }

  return `${method} ${path} failed: ${status}${detail ? ` ${detail}` : ""}`;
}

export async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const detail = extractDetail(text);
    throw new Error(friendlyErrorMessage("GET", path, res.status, detail));
  }

  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const detail = extractDetail(text);
    throw new Error(friendlyErrorMessage("POST", path, res.status, detail));
  }

  return res.json();
}