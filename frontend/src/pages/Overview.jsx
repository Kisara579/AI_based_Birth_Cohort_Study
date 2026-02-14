import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import OutcomeCard from "../components/models/OutcomeCard.jsx";
import { outcomes } from "../data/mockOutcomes.js";

export default function Overview() {
  return (
    <div>
      <h1>Overview</h1>
      <p className="muted">Choose an outcome, run predictions, and see.</p>

      <Card title="Welcome" subtitle="This is a reusable Card component">
        <p className="muted">
          Next we will build the Outcomes grid (11 outcomes).
        </p>
      </Card>

      <Card title="Buttons" subtitle="Reusable button variants">
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Button variant="primary">Primary</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="ghost">Ghost</Button>
          <Button disabled>Disabled</Button>
        </div>
      </Card>

      <Card title="Loading Skeleton" subtitle="Reusable loading state">
        <LoadingSkeleton lines={4} />
      </Card>

      <Card title="Error example">
        <ErrorBanner
          title="API not reachable"
          message="Please start backend server or check network."
          onRetry={() => alert("Retry clicked")}
        />
      </Card>

      <div className="grid">
        {outcomes.map((o) => (
          <OutcomeCard key={o.id} outcome={o} />
        ))}
      </div>
    </div>
  );
}
