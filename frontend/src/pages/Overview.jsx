import { useEffect, useState } from "react";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import OutcomeCard from "../components/models/OutcomeCard.jsx";
import { fetchOutcomes } from "../api/modelsApi.js";

export default function Overview() {
  const [outcomes, setOutcomes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setErr("");
      try {
        const data = await fetchOutcomes();
        if (!cancelled) setOutcomes(data);
      } catch (e) {
        if (!cancelled) setErr(e.message || "Failed to load outcomes");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <h1>Overview</h1>
      <p className="muted">Select an outcome to start a prediction.</p>

      {loading ? (
        <LoadingSkeleton lines={6} />
      ) : err ? (
        <ErrorBanner title="Could not load outcomes" message={err} />
      ) : (
        <div className="grid">
          {outcomes.map((o) => (
            <OutcomeCard key={o.id} outcome={o} />
          ))}
        </div>
      )}
    </div>
  );
}
