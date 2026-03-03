import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";

export default function MetricsModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setErr("");
    setData(null);

    fetch(`http://127.0.0.1:8000/metrics/${outcomeId}`)
      .then(async (r) => {
        const payload = await r.json();
        if (!r.ok) throw new Error(payload.detail || "Failed to load metrics");
        return payload;
      })
      .then(setData)
      .catch((e) => setErr(e.message));
  }, [outcomeId]);

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
        <div style={{ marginTop: 12 }}>
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

  // Pick key metrics to show as highlight stat cards
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

  return (
    <div>
      {/* Breadcrumb */}
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

      {/* Highlight stat cards */}
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

      {/* Summary metrics */}
      <Card title="Summary Metrics" subtitle="Overall performance on test set">
        <table className="table">
          <thead>
            <tr>
              <th align="left">Metric</th>
              <th align="left">Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([k, v]) => (
              <tr key={k}>
                <td>{fmtKey(k)}</td>
                <td>{typeof v === "number" ? v.toFixed(4) : String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Classification report */}
      {report && (
        <Card
          title="Classification Report"
          subtitle="Per-class precision/recall/F1"
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

      {/* Confusion matrix */}
      {cm && (
        <Card
          title="Confusion Matrix"
          subtitle="Actual (rows) vs Predicted (columns)"
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
