import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import DynamicForm from "../components/forms/DynamicForm.jsx";
import { fetchOutcomes, fetchSchema, fetchModels } from "../api/modelsApi.js";
import "./PredictModel.css";

export default function PredictModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const initialValues = location.state?.formValues || null;

  const [outcomes, setOutcomes] = useState([]);
  const [schema, setSchema] = useState(null);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomes, outcomeId],
  );

  const outcomeDisplayName = outcome?.name || outcomeId.toUpperCase();

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setErr("");
      try {
        const [outs, sch, modelInfo] = await Promise.all([
          fetchOutcomes(),
          fetchSchema(outcomeId),
          fetchModels(outcomeId),
        ]);

        if (sch?.error) throw new Error(sch.error);

        if (!cancelled) {
          setOutcomes(outs);
          setSchema(sch);
          setModelsInfo(modelInfo);
        }
      } catch (e) {
        if (!cancelled) setErr(e.message || "Failed to load schema");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [outcomeId]);

  function handleSubmit(values) {
    navigate(`/results/${outcomeId}`, {
      state: { formValues: values },
    });
  }
  if (loading) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Predict</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Predict</h2>
            <p className="pageSubtitle">Loading outcome form...</p>
          </div>
        </div>
        <LoadingSkeleton lines={6} />
      </div>
    );
  }

  if (err) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Predict</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Predict</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcomeDisplayName}</b>
            </p>
          </div>
          <Button variant="ghost" onClick={() => navigate("/")}>
            ← Back
          </Button>
        </div>
        <ErrorBanner title="Could not load prediction form" message={err} />
      </div>
    );
  }

  if (!outcome) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Predict</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Predict</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcomeDisplayName}</b>
            </p>
          </div>
          <Button variant="ghost" onClick={() => navigate("/")}>
            ← Back
          </Button>
        </div>
        <ErrorBanner
          title="Unknown outcome"
          message={`No outcome found for id: "${outcomeId}"`}
        />
      </div>
    );
  }

  // const availableModels = modelsInfo?.models || [];

  return (
    <div>
      {/* Breadcrumb */}
      <nav className="breadcrumb">
        <a href="/" className="breadcrumbLink">
          Overview
        </a>
        <span className="breadcrumbSep">›</span>
        <span className="breadcrumbCurrent">Predict</span>
      </nav>

      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">Predict</h2>
          <p className="pageSubtitle">
            Outcome: <b>{outcomeDisplayName}</b>
          </p>
        </div>
        <Button variant="ghost" onClick={() => navigate("/")}>
          ← Back
        </Button>
      </div>

      <Card
        title="Prediction Model"
        subtitle="Active model selected via clinical metric priority hierarchy (Recall > F1-score > ROC-AUC > Precision > Accuracy)"
        className="predictModelPanel"
      >
        <div className="predictModelInfo">
          <div className="predictModelRow">
            <span className="predictModelLabel">Active Algorithm</span>
            <span className="predictModelValue">
              {modelsInfo?.models?.find(
                (m) => m.algorithm === modelsInfo?.active_model,
              )?.label ||
                modelsInfo?.active_model ||
                "Not available"}
            </span>
          </div>
        </div>
      </Card>

      <Card title={schema?.title || "Inputs"} className="predictFormCard">
        <DynamicForm
          schema={schema}
          initialValues={initialValues}
          onSubmit={handleSubmit}
        />
      </Card>
    </div>
  );
}
