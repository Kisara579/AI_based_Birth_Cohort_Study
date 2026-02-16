import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import DynamicForm from "../components/forms/DynamicForm.jsx";
import { fetchOutcomes, fetchSchema } from "../api/modelsApi.js";

export default function PredictModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const [outcomes, setOutcomes] = useState([]);
  const [schema, setSchema] = useState(null);
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
        // We load outcomes too so we can show name/short on this page
        const [outs, sch] = await Promise.all([
          fetchOutcomes(),
          fetchSchema(outcomeId),
        ]);

        // backend returns {"error":"Schema not found"} sometimes
        if (sch?.error) throw new Error(sch.error);

        if (!cancelled) {
          setOutcomes(outs);
          setSchema(sch);
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
    // Later we will POST to backend /predict
    navigate(`/results/${outcomeId}`, { state: { formValues: values } });
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

  return (
    <div>
      <h1>Predict</h1>
      <p className="muted">
        Outcome: <b>{outcome.name}</b> ({outcome.short})
      </p>

      <Card title={schema?.title || "Inputs"}>
        <DynamicForm schema={schema} onSubmit={handleSubmit} />
      </Card>
    </div>
  );
}
