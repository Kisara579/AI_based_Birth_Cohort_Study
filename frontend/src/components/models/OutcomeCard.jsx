import { useNavigate } from "react-router-dom";
import Card from "../common/Card.jsx";
import Button from "../common/Button.jsx";

function outcomeStyle() {
  return {
    iconClass: "outcomeIconBlue",
    tagClass: "tagBlue",
    icon: (
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  };
}

export default function OutcomeCard({ outcome }) {
  const navigate = useNavigate();
  const { iconClass, tagClass, icon } = outcomeStyle();

  return (
    <Card className="outcomeCard">
      {/* icon */}
      <div className={`outcomeIconWrap ${iconClass}`}>{icon}</div>

      {/* title + description */}
      <div style={{ marginBottom: 8 }}>
        <h2 className="cardTitle">{outcome.name}</h2>
        {outcome.description && (
          <p className="cardSubtitle">{outcome.description}</p>
        )}
      </div>

      {/* footer*/}
      <div className="outcomeFooter">
        <span className={`tag ${tagClass}`}>{outcome.short}</span>

        <div className="outcomeActions">
          <Button
            variant="ghost"
            onClick={() => navigate(`/metrics/${outcome.id}`)}
          >
            Metrics
          </Button>
          <Button
            variant="primary"
            onClick={() => navigate(`/predict/${outcome.id}`)}
          >
            Predict
          </Button>
        </div>
      </div>
    </Card>
  );
}
