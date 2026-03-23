import { useEffect, useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import { fetchMetrics, fetchModels, fetchOutcomes } from "../api/modelsApi.js";
import "./MetricsModel.css";

export default function MetricsModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const [outcomes, setOutcomes] = useState([]);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [modelsData, setModelsData] = useState([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  const outcome = useMemo(
    () => outcomes.find((item) => item.id === outcomeId),
    [outcomes, outcomeId],
  );

  const outcomeDisplayName = outcome?.name || outcomeId.toUpperCase();

  useEffect(() => {
    let cancelled = false;

    async function loadAllMetrics() {
      setLoading(true);
      setErr("");
      setModelsData([]);

      try {
        const [info, outcomesData] = await Promise.all([
          fetchModels(outcomeId),
          fetchOutcomes(),
        ]);
        if (cancelled) return;

        setOutcomes(outcomesData);
        setModelsInfo(info);

        const availableModels = (info?.models || []).filter(
          (m) => m.trained !== false,
        );

        if (availableModels.length === 0) {
          setModelsData([]);
          setLoading(false);
          return;
        }

        const metricsResults = await Promise.all(
          availableModels.map(async (model) => {
            try {
              const payload = await fetchMetrics(outcomeId, model.algorithm);
              return {
                algorithm: model.algorithm,
                label: model.label || model.algorithm,
                trained: model.trained,
                isActive: info?.active_model === model.algorithm,
                data: payload,
                error: "",
              };
            } catch (e) {
              return {
                algorithm: model.algorithm,
                label: model.label || model.algorithm,
                trained: model.trained,
                isActive: info?.active_model === model.algorithm,
                data: null,
                error: e.message || "Failed to load metrics",
              };
            }
          }),
        );

        if (cancelled) return;

        const sorted = [...metricsResults].sort((a, b) => {
          if (a.isActive && !b.isActive) return -1;
          if (!a.isActive && b.isActive) return 1;
          return 0;
        });

        setModelsData(sorted);
      } catch (e) {
        if (!cancelled) {
          setErr(e.message || "Failed to load models");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadAllMetrics();

    return () => {
      cancelled = true;
    };
  }, [outcomeId]);

  const fmtKey = (k) =>
    k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  const fmtVal = (v) =>
    typeof v === "number"
      ? v <= 1
        ? (v * 100).toFixed(1) + "%"
        : v.toFixed(4)
      : String(v);

  const pageSubtitle = useMemo(() => {
    const activeLabel =
      modelsData.find((m) => m.isActive)?.label ||
      modelsInfo?.models?.find((m) => m.algorithm === modelsInfo?.active_model)
        ?.label ||
      modelsInfo?.active_model ||
      "None";

    const hasActive = modelsData.some((m) => m.isActive);

    return {
      outcome: outcomeDisplayName,
      activeLabel: hasActive ? activeLabel : null,
    };
  }, [outcomeDisplayName, modelsData, modelsInfo]);

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

        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Model Metrics</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcomeDisplayName}</b>
            </p>
          </div>
          <Button variant="ghost" onClick={() => navigate(-1)}>
            ← Back
          </Button>
        </div>

        <ErrorBanner message={err} />
      </div>
    );
  }

  if (loading) {
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
            <p className="pageSubtitle">Loading metrics...</p>
          </div>
        </div>

        <LoadingSkeleton />
      </div>
    );
  }

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
            <span className="pageSubtitleOutcome">
              Outcome: <b>{pageSubtitle.outcome}</b>
            </span>
            {pageSubtitle.activeLabel && (
              <>
                <span className="pageSubtitleSep">·</span>
                <span className="pageSubtitleActive">
                  Active Model: <b>{pageSubtitle.activeLabel}</b>
                </span>
              </>
            )}
          </p>
        </div>
        <Button variant="ghost" onClick={() => navigate(-1)}>
          ← Back
        </Button>
      </div>

      {modelsData.length === 0 ? (
        <ErrorBanner message="No trained models found for this outcome." />
      ) : (
        <div className="modelMetricsStack">
          {modelsData.map((modelItem) => {
            const data = modelItem.data || {};
            const metrics = data.metrics || {};
            const report = data.classification_report || null;
            const cm = data.confusion_matrix || null;

            const reportRowsOrder = ["0", "1", "macro avg", "weighted avg"];
            const HIGHLIGHT = [
              "accuracy",
              "roc_auc",
              "f1",
              "precision",
              "recall",
              "auc",
            ];

            const statEntries = Object.entries(metrics).filter(([k]) =>
              HIGHLIGHT.some((h) => k.toLowerCase().includes(h)),
            );

            return (
              <Card
                key={modelItem.algorithm}
                className={`modelMetricsCard ${modelItem.isActive ? "modelMetricsCardActive" : ""}`}
              >
                <div className="modelHeader">
                  <div className="modelHeaderLeft">
                    <div className="modelTitleRow">
                      <h3 className="modelTitle">
                        {modelItem.label} Classification
                      </h3>
                      <span
                        className={`modelBadge ${modelItem.isActive ? "modelBadgeActive" : "modelBadgeAvailable"}`}
                      >
                        {modelItem.isActive ? "Active Model" : "Available"}
                      </span>
                    </div>

                    <p className="modelTarget">
                      Target Outcome: <b>{outcomeDisplayName}</b>
                    </p>
                  </div>

                  {/* <div className="modelHeaderRight">
                    <div className="modelHeaderMeta">
                      <span className="modelHeaderMetaLabel">Status</span>
                      <span
                        className={`statusBadge ${modelItem.isActive ? "statusBadgeDeployed" : "statusBadgeIdle"}`}
                      >
                        {modelItem.isActive ? "Deployed" : "Available"}
                      </span>
                    </div>
                  </div> */}
                </div>

                {modelItem.error ? (
                  <ErrorBanner message={modelItem.error} />
                ) : (
                  <>
                    {statEntries.length > 0 && (
                      <div className="metricsGrid">
                        {statEntries.map(([k, v]) => (
                          <div key={k} className="metricCard">
                            <span className="metricValue">{fmtVal(v)}</span>
                            <span className="metricLabel">{fmtKey(k)}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="metricsBottomGrid">
                      {report && (
                        <div className="metricsPanel">
                          <h4 className="sectionTitle">
                            Classification Report
                          </h4>
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
                                    <td>
                                      {(row["precision"] ?? 0).toFixed(4)}
                                    </td>
                                    <td>{(row["recall"] ?? 0).toFixed(4)}</td>
                                    <td>{(row["f1-score"] ?? 0).toFixed(4)}</td>
                                    <td>{row["support"] ?? "-"}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {cm && Array.isArray(cm) && cm.length >= 2 && (
                        <div className="metricsPanel">
                          <h4 className="sectionTitle">Confusion Matrix</h4>

                          <div className="cmWrap">
                            <div className="cmTopLabel">Predicted →</div>

                            <div className="cmGrid">
                              <div className="cmCorner">Actual ↓</div>

                              <div className="cmHead">0</div>
                              <div className="cmHead cmHeadRight">1</div>

                              <div className="cmHead">0</div>
                              <div className="cmCell cmDiag">{cm[0][0]}</div>
                              <div className="cmCell cmCellRight">
                                {cm[0][1]}
                              </div>

                              <div className="cmHead cmRowLast">1</div>
                              <div className="cmCell cmRowLast">{cm[1][0]}</div>
                              <div className="cmCell cmCellRight cmRowLast cmDiag">
                                {cm[1][1]}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
