import { useMemo } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import { outcomes } from "../data/mockOutcomes.js";

function mockPredictLBW(values) {
  // Very simple fake scoring just for UI testing
  // (not medical, not real model)
  let score = 0;

  if (values.mage !== null && values.mage < 18) score += 0.12;
  if (values.meduce !== null && values.meduce < 10) score += 0.08; // if typo later, ignore
  if (values.meduc !== null && values.meduc < 10) score += 0.08;

  if (values.cigs !== null) score += Math.min(0.25, values.cigs * 0.01);
  if (values.drink === 1) score += 0.12;
  if (values.monpre !== null && values.monpre > 3) score += 0.10;
  if (values.npvis !== null && values.npvis < 4) score += 0.10;

  // clamp into 0.05 - 0.95
  return Math.max(0.05, Math.min(0.95, 0.15 + score));
}

export default function ResultsModel() {
  const { outcomeId } = useParams();
  const location = useLocation();

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomeId]
  );

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

  const formValues = location.state?.formValues;

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

  // temporary: outcome-specific mock predictor
  const prob =
    outcomeId === "lbw" ? mockPredictLBW(formValues) : 0.25;

  const riskLabel = prob >= 0.5 ? "High Risk" : "Low Risk";

  return (
    <div>
      <h1>Results</h1>
      <p className="muted">
        Outcome: <b>{outcome.name}</b> ({outcome.short})
      </p>

      <Card title="Prediction Result" subtitle="Frontend mock (temporary)">
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
        subtitle="What you entered (for debugging now, later remove)"
        className="mt16"
      >
        <pre className="codeBlock">{JSON.stringify(formValues, null, 2)}</pre>
      </Card>
    </div>
  );
}
