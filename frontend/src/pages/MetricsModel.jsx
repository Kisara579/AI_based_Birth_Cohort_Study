import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import { fetchMetrics, fetchModels } from "../api/modelsApi.js";
import "./MetricsModel.css";

export default function MetricsModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [modelsInfo, setModelsInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      try {
        const info = await fetchModels(outcomeId);
        if (cancelled) return;

        setModelsInfo(info);
        setSelectedModel(info?.active || "");
      } catch (e) {
        if (cancelled) return;

        setModelsInfo({ active: "", models: [] });
        setSelectedModel("");
      }
    }

    loadModels();
    return () => {
      cancelled = true;
    };
  }, [outcomeId]);

  useEffect(() => {
    let cancelled = false;

    async function loadMetrics() {
      setErr("");
      setData(null);

      try {
        const payload = await fetchMetrics(
          outcomeId,
          selectedModel || undefined,
        );
        if (!cancelled) setData(payload);
      } catch (e) {
        if (!cancelled) setErr(e.message || "Failed to load metrics");
      }
    }

    loadMetrics();

    return () => {
      cancelled = true;
    };
  }, [outcomeId, selectedModel]);

  const availableModels = useMemo(() => modelsInfo?.models || [], [modelsInfo]);

  if (err) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Model Metrics</span>
        </nav>
        <ErrorBanner message={err} />
        <div className="metricsBackWrap">
          <Button variant="ghost" onClick={() => navigate(-1)}>
            ← Back
          </Button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Model Metrics</span>
        </nav>
        <LoadingSkeleton />
      </div>
    );
  }

  const metrics = data.metrics || {};
  const report = data.classification_report || null;
  const cm = data.confusion_matrix || null;

  const reportRowsOrder = ["0", "1", "macro avg", "weighted avg"];

  const HIGHLIGHT = ["accuracy", "roc_auc", "f1", "precision", "recall", "auc"];
  const statEntries = Object.entries(metrics).filter(([k]) =>
    HIGHLIGHT.some((h) => k.toLowerCase().includes(h)),
  );

  const fmtKey = (k) =>
    k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  const fmtVal = (v) =>
    typeof v === "number"
      ? v <= 1
        ? (v * 100).toFixed(1) + "%"
        : v.toFixed(4)
      : String(v);

  const currentModelLabel =
    availableModels.find((m) => m.algorithm === selectedModel)?.label ||
    selectedModel;

  return (
    <div>
      <nav className="breadcrumb">
        <a href="/" className="breadcrumbLink">
          Overview
        </a>
        <span className="breadcrumbSep">›</span>
        <span className="breadcrumbCurrent">Model Metrics</span>
      </nav>

      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">Model Metrics</h2>
          <p className="pageSubtitle">
            Outcome: <b>{outcomeId.toUpperCase()}</b>
          </p>
        </div>
        <Button variant="ghost" onClick={() => navigate(-1)}>
          ← Back
        </Button>
      </div>

      {availableModels.length > 0 && (
        <Card
          title="Model Selection"
          subtitle="Choose algorithm for metrics view"
        >
          <div className="modelSelectGrid">
            <label htmlFor="model-select" className="formLabel">
              Algorithm
            </label>
            <select
              id="model-select"
              className="formInput"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {availableModels.map((m) => (
                <option
                  key={m.algorithm}
                  value={m.algorithm}
                  disabled={m.trained === false}
                >
                  {m.label}
                  {m.trained === false ? " (not trained)" : ""}
                </option>
              ))}
            </select>
          </div>
        </Card>
      )}

      <Card>
        <h2>Model Statistics</h2>
        {statEntries.length > 0 && (
          <div className="statGrid">
            {statEntries.map(([k, v]) => (
              <div key={k} className="statCard statCardAccent">
                <span className="statCardValue">{fmtVal(v)}</span>
                <span className="statCardLabel">{fmtKey(k)}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {report && (
        <Card
          title="Classification Report"
          subtitle={`Per-class precision/recall/F1 • ${currentModelLabel}`}
        >
          <table className="table">
            <thead>
              <tr>
                <th align="left">Class</th>
                <th align="left">Precision</th>
                <th align="left">Recall</th>
                <th align="left">F1</th>
                <th align="left">Support</th>
              </tr>
            </thead>
            <tbody>
              {reportRowsOrder.map((label) => {
                const row = report[label];
                if (!row) return null;

                return (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{(row["precision"] ?? 0).toFixed(4)}</td>
                    <td>{(row["recall"] ?? 0).toFixed(4)}</td>
                    <td>{(row["f1-score"] ?? 0).toFixed(4)}</td>
                    <td>{row["support"] ?? "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      )}

      {cm && (
        <Card
          title="Confusion Matrix"
          subtitle={`Actual (rows) vs Predicted (columns) • ${currentModelLabel}`}
        >
          <div className="cmWrap">
            <div className="cmTopLabel">Predicted →</div>

            <div className="cmGrid">
              <div className="cmCorner">Actual ↓</div>

              <div className="cmHead">0</div>
              <div className="cmHead cmHeadRight">1</div>

              <div className="cmHead">0</div>
              <div className="cmCell cmDiag">{cm[0][0]}</div>
              <div className="cmCell cmCellRight">{cm[0][1]}</div>

              <div className="cmHead cmRowLast">1</div>
              <div className="cmCell cmRowLast">{cm[1][0]}</div>
              <div className="cmCell cmCellRight cmRowLast cmDiag">
                {cm[1][1]}
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
