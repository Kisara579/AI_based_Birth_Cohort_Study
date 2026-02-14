import { useNavigate } from "react-router-dom";
import Card from "../common/Card.jsx";
import Button from "../common/Button.jsx";

export default function OutcomeCard({ outcome }) {
  const navigate = useNavigate();

  return (
    <Card
      className="outcomeCard"
      title={`${outcome.name}`}
      subtitle={outcome.description}
    >
      <div className="outcomeFooter">
        <span className="muted">{outcome.short}</span>
        <Button variant="primary" onClick={() => navigate(`/predict/${outcome.id}`)}>
          Predict
        </Button>
      </div>
    </Card>
  );
}
