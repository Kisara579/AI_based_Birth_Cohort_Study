import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import { fetchOutcomes, predictOutcome } from "../api/modelsApi.js";

export default function ResultsModel() {
  const { outcomeId } = useParams();
  const location = useLocation();

  // outcomes
  const [outcomes, setOutcomes] = useState([]);
  const [loadingOutcomes, setLoadingOutcomes] = useState(true);
  const [outcomesErr, setOutcomesErr] = useState("");

  // prediction
  const [predLoading, setPredLoading] = useState(false);
  const [predErr, setPredErr] = useState("");
  const [prediction, setPrediction] = useState(null);

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomes, outcomeId]
  );

  const formValues = location.state?.formValues;

  // load outcomes (to display name/short)
  useEffect(() => {
    let cancelled = false;

    async function loadOutcomes() {
      setLoadingOutcomes(true);
      setOutcomesErr("");
      try {
        const outs = await fetchOutcomes();
        if (!cancelled) setOutcomes(outs);
      } catch (e) {
        if (!cancelled) setOutcomesErr(e.message || "Failed to load outcomes");
      } finally {
        if (!cancelled) setLoadingOutcomes(false);
      }
    }

    loadOutcomes();
    return () => {
      cancelled = true;
    };
  }, []);

  // call backend prediction when we have inputs
  useEffect(() => {
    let cancelled = false;

    async function runPrediction() {
      if (!formValues) return;

      setPredLoading(true);
      setPredErr("");
      setPrediction(null);

      try {
        const res = await predictOutcome(outcomeId, formValues);

        // backend might return {"error": "..."}
        if (res?.error) throw new Error(res.error);

        if (!cancelled) setPrediction(res);
      } catch (e) {
        if (!cancelled) setPredErr(e.message || "Prediction failed");
      } finally {
        if (!cancelled) setPredLoading(false);
      }
    }

    runPrediction();

    return () => {
      cancelled = true;
    };
  }, [outcomeId, formValues]);

  // ui statuses

  if (loadingOutcomes) {
    return (
      <div>
        <h1>Results</h1>
        <LoadingSkeleton lines={5} />
      </div>
    );
  }

  if (outcomesErr) {
    return (
      <div>
        <h1>Results</h1>
        <ErrorBanner title="Could not load outcomes" message={outcomesErr} />
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
        <h1>Results</h1>
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

  // If user opened results directly (no form values passed)
  if (!formValues) {
    return (
      <div>
        <h1>Results</h1>
        <p className="muted">
          Outcome: <b>{outcome.name}</b> ({outcome.short})
        </p>

        <ErrorBanner
          title="No input data"
          message="Please run a prediction first. This page expects form inputs from the Predict screen."
        />

        <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
          <Link to={`/predict/${outcome.id}`}>
            <Button variant="primary">Go to Predict</Button>
          </Link>

          <Link to="/">
            <Button variant="ghost">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }

  // Prediction loading / error
  if (predLoading) {
    return (
      <div>
        <h1>Results</h1>
        <p className="muted">
          Outcome: <b>{outcome.name}</b> ({outcome.short})
        </p>
        <LoadingSkeleton lines={5} />
      </div>
    );
  }

  if (predErr) {
    return (
      <div>
        <h1>Results</h1>
        <p className="muted">
          Outcome: <b>{outcome.name}</b> ({outcome.short})
        </p>

        <ErrorBanner title="Prediction failed" message={predErr} />

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 14 }}>
          <Link to={`/predict/${outcome.id}`}>
            <Button variant="secondary">Edit Inputs</Button>
          </Link>

          <Link to="/">
            <Button variant="ghost">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }

  const prob = prediction?.probability ?? 0;
  const riskLabel = prediction?.risk_label ?? "Unknown";

  return (
    <div>
      <h1>Results</h1>
      <p className="muted">
        Outcome: <b>{outcome.name}</b> ({outcome.short})
      </p>

      <Card title="Prediction Result" subtitle="From backend API">
        <div style={{ display: "grid", gap: 10 }}>
          <div>
            <div className="muted">Predicted probability</div>
            <div style={{ fontSize: 28, fontWeight: 800 }}>
              {(prob * 100).toFixed(1)}%
            </div>
          </div>

          <div>
            <div className="muted">Risk category</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{riskLabel}</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 14 }}>
          <Link to={`/predict/${outcome.id}`}>
            <Button variant="secondary">Edit Inputs</Button>
          </Link>

          <Link to="/">
            <Button variant="ghost">Back to Overview</Button>
          </Link>
        </div>
      </Card>

      <Card
        title="Submitted Inputs"
        subtitle="Debug only; later we remove"
        className="mt16"
      >
        <pre className="codeBlock">{JSON.stringify(formValues, null, 2)}</pre>
      </Card>
    </div>
  );
}
