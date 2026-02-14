import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import { outcomes } from "../data/mockOutcomes.js";

export default function PredictModel() {
  const { outcomeId } = useParams();

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomeId]
  );

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

  return (
    <div>
      <h1>Predict</h1>
      <p className="muted">
        Outcome: <b>{outcome.name}</b> ({outcome.short})
      </p>

      <Card title="Patient Inputs" subtitle="Dynamic form will be added next">
        <p className="muted">
          For now, this is a placeholder. Next step we will generate inputs from
          <code> mockSchemas.js</code>.
        </p>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 14 }}>
          <Link to="/">
            <Button variant="secondary">Back</Button>
          </Link>

          <Link to={`/results/${outcome.id}`}>
            <Button variant="primary">Go to Results</Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}
