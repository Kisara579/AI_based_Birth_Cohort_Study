import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import { outcomes } from "../data/mockOutcomes.js";

function fakeProbabilityFromId(id) {
  // stable-ish fake number between 0.10 and 0.90
  let sum = 0;
  for (let i = 0; i < id.length; i++) sum += id.charCodeAt(i);
  const raw = (sum % 80) / 100; // 0.00 - 0.79
  return Math.min(0.9, Math.max(0.1, raw + 0.1));
}

export default function ResultsModel() {
  const { outcomeId } = useParams();

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

  const prob = fakeProbabilityFromId(outcome.id);
  const riskLabel = prob >= 0.5 ? "High Risk" : "Low Risk";

  return (
    <div>
      <h1>Results</h1>
      <p className="muted">
        Outcome: <b>{outcome.name}</b> ({outcome.short})
      </p>

      <Card title="Prediction Result" subtitle="Temporary mock result">
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
            <Button variant="secondary">Back to Predict</Button>
          </Link>

          <Link to="/">
            <Button variant="ghost">Back to Overview</Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}
