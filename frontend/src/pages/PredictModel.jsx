import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import DynamicForm from "../components/forms/DynamicForm.jsx";
import { outcomes } from "../data/mockOutcomes.js";
import { schemasByOutcome } from "../data/mockSchemas.js";

export default function PredictModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomeId]
  );

  const schema = schemasByOutcome[outcomeId];

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

  if (!schema) {
    return (
      <div>
        <h1>Predict</h1>
        <ErrorBanner
          title="Schema not defined"
          message={`No form schema defined for outcome "${outcomeId}".`}
        />
        <div style={{ marginTop: 14 }}>
          <Link to="/">
            <Button variant="secondary">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }

  function handleSubmit(values) {
    console.log("Form values:", values);

    // Later this will call backend
    navigate(`/results/${outcomeId}`, { state: { formValues: values } });
  }

  return (
    <div>
      <h1>Predict</h1>
      <p className="muted">
        Outcome: <b>{outcome.name}</b> ({outcome.short})
      </p>

      <Card title={schema.title}>
        <DynamicForm schema={schema} onSubmit={handleSubmit} />
      </Card>
    </div>
  );
}
