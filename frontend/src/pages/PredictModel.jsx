import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import DynamicForm from "../components/forms/DynamicForm.jsx";
import { fetchOutcomes, fetchSchema, fetchModels } from "../api/modelsApi.js";

export default function PredictModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const [outcomes, setOutcomes] = useState([]);
  const [schema, setSchema] = useState(null);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomes, outcomeId],
  );

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
          setSelectedModel(modelInfo?.active || "");
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
      state: { formValues: values, selectedModel },
    });
  }

  if (loading) {
    return (
      <div>
        <h1>Predict</h1>
        <LoadingSkeleton lines={6} />
      </div>
    );
  }

  if (err) {
    return (
      <div>
        <h1>Predict</h1>
        <ErrorBanner title="Could not load prediction form" message={err} />
        <div style={{ marginTop: 14 }}>
          <Link to="/">
            <Button variant="secondary">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }

  if (!outcome) {
    return (
      <div>
        <h1>Predict</h1>
        <ErrorBanner
          title="Unknown outcome"
          message={`No outcome found for id: "${outcomeId}"`}
        />
        <div style={{ marginTop: 14 }}>
          <Link to="/">
            <Button variant="secondary">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }

  const availableModels = modelsInfo?.models || [];

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
          subtitle="Choose algorithm for prediction"
        >
          <div style={{ display: "grid", gap: 10 }}>
            <label htmlFor="predict-model-select" className="formLabel">
              Algorithm
            </label>
            <select
              id="predict-model-select"
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

      <Card title={schema?.title || "Inputs"}>
        <DynamicForm schema={schema} onSubmit={handleSubmit} />
      </Card>
    </div>
  );
}
