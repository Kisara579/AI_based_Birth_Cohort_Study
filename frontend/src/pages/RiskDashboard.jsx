import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import { predictAllOutcomes, fetchModels } from "../api/modelsApi.js";

import "./RiskDashboard.css";

export default function RiskDashboard() {
  const { state } = useLocation();
  const navigate = useNavigate();

  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!state?.payloads || !state?.outcomes) {
      navigate("/predict");
      return;
    }

    let cancelled = false;

    async function runPredictions() {
      setLoading(true);

      const configured = state.outcomes.filter((o) => state.payloads[o.id]);
      const unconfigured = state.outcomes.filter((o) => !state.payloads[o.id]);

      try {
        // Fetch predictions AND model info in parallel
        const [responses, ...modelInfos] = await Promise.all([
          predictAllOutcomes(configured, state.payloads),
          ...configured.map((o) =>
            fetchModels(o.id).catch(() => null)
          ),
        ]);

        // Build a lookup: outcomeId -> active algorithm label
        const modelLookup = {};
        configured.forEach((o, i) => {
          const info = modelInfos[i];
          if (info) {
            const activeAlgo = info.active_model;
            const match = (info.models || []).find((m) => m.algorithm === activeAlgo);
            modelLookup[o.id] = match?.label || activeAlgo || "Unknown";
          }
        });

        // Merge model name into each response
        const enriched = responses.map((r) => ({
          ...r,
          modelName: modelLookup[r.id] || null,
        }));

        const unconfiguredResults = unconfigured.map((o) => ({
          id: o.id,
          name: o.name,
          success: false,
          error: "Model mapping not configured yet.",
        }));

        if (!cancelled) {
          setResults([...enriched, ...unconfiguredResults]);
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    runPredictions();
    return () => { cancelled = true; };
  }, [state, navigate]);

  if (loading) {
    return (
      <div>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Analyzing Patient Data…</h2>
            <p className="pageSubtitle">
              Running global risk assessment across all models.
            </p>
          </div>
        </div>
        <LoadingSkeleton lines={10} />
      </div>
    );
  }

  // Stats
  const successful = results.filter((r) => r.success);
  const highRisks  = successful.filter((r) => r.probability >= 0.7);
  const modRisks   = successful.filter((r) => r.probability >= 0.4 && r.probability < 0.7);
  const lowRisks   = successful.filter((r) => r.probability < 0.4);
  const errored    = results.filter((r) => !r.success);

  return (
    <div>
      {/* Breadcrumb */}
      <nav className="breadcrumb">
        <a href="/" className="breadcrumbLink">Overview</a>
        <span className="breadcrumbSep">›</span>
        <button
          className="breadcrumbBtn"
          onClick={() => navigate(-1)}
        >
          Predict
        </button>
        <span className="breadcrumbSep">›</span>
        <span className="breadcrumbCurrent">Results</span>
      </nav>

      {/* Header */}
      <div className="pageHeader">
        <div>
          <h1 className="pageTitle">Patient Risk Dashboard</h1>
          <p className="pageSubtitle">
            Simultaneous assessment across {results.length} outcomes.
          </p>
        </div>
        <Button variant="ghost" onClick={() => navigate(-1)}>
          ← Edit Patient Data
        </Button>
      </div>

      {/* Summary stats */}
      <div className="dashboardSummary">
        <div className={`summaryStat ${highRisks.length > 0 ? "critical" : "safe"}`}>
          <span className="statValue">{highRisks.length}</span>
          <span className="statLabel">High Risk</span>
        </div>
        <div className={`summaryStat ${modRisks.length > 0 ? "warning" : "safe"}`}>
          <span className="statValue">{modRisks.length}</span>
          <span className="statLabel">Moderate</span>
        </div>
        <div className="summaryStat safe">
          <span className="statValue">{lowRisks.length}</span>
          <span className="statLabel">Low Risk</span>
        </div>
        {errored.length > 0 && (
          <div className="summaryStat neutral">
            <span className="statValue">{errored.length}</span>
            <span className="statLabel">Pending</span>
          </div>
        )}
      </div>

      {errored.length > 0 && (
        <ErrorBanner
          title={`${errored.length} models could not predict`}
          message="Some models are missing trained algorithms."
        />
      )}

      {/* Grid */}
      <div className="dashboardGrid">
        {results.map((r) => (
          <ResultCard key={r.id} result={r} />
        ))}
      </div>
    </div>
  );
}

/* Single result card  */
function ResultCard({ result }) {
  if (!result.success) {
    return (
      <div className="riskCard pendingCard">
        <div className="riskCardHeader">
          <h3 className="riskCardTitle">{result.name}</h3>
          <span className="riskBadge pending">Pending</span>
        </div>
        <p className="pendingText">{result.error}</p>
      </div>
    );
  }

  const prob = result.probability;
  let statusClass, label;

  if (prob >= 0.7) {
    statusClass = "highRisk";
    label = "High Risk";
  } else if (prob >= 0.4) {
    statusClass = "moderateRisk";
    label = "Moderate Risk";
  } else {
    statusClass = "lowRisk";
    label = "Low Risk";
  }

  return (
    <div className={`riskCard ${statusClass}`}>
      <div className="riskCardHeader">
        <h3 className="riskCardTitle">{result.name}</h3>
        <span className={`riskBadge ${statusClass}`}>{label}</span>
      </div>

      <div className="riskCardBody">
        <div className="probabilityCircle">
          <svg viewBox="0 0 36 36" className="circularChart">
            <path
              className="circleBg"
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <path
              className="circle"
              strokeDasharray={`${prob * 100}, 100`}
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            />
            <text x="18" y="20.35" className="percentage">
              {(prob * 100).toFixed(1)}%
            </text>
          </svg>
        </div>

        {result.modelName && (
          <div className="riskCardDetails">
            <div className="detailRow">
              <span className="detailLabel">Algorithm</span>
              <span className="detailValue">{result.modelName}</span>
            </div>
            <div className="detailRow">
              <span className="detailLabel">Risk Level</span>
              <span className="detailValue">{label}</span>
            </div>
            <div className="detailRow">
              <span className="detailLabel">Probability</span>
              <span className="detailValue">{(prob * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
