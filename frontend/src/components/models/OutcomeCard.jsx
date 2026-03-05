import { useNavigate } from "react-router-dom";
import Button from "../common/Button.jsx";

const SVG_ICON = (
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
);

export default function OutcomeCard({ outcome }) {
  const navigate = useNavigate();

  return (
    <div className="card outcomeCard">
      {/* icon */}
      <div className="outcomeIconWrap outcomeIconBlue">{SVG_ICON}</div>

      {/* title */}
      <h2
        className="cardTitle"
        style={{
          margin: "10px 0 4px",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {outcome.name}
      </h2>

      <p
        className="cardSubtitle"
        style={{
          lineHeight: "1.5",
          minHeight: "3em",
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
          margin: 0,
        }}
      >
        {outcome.description}
      </p>

      {/* footer */}
      <div className="outcomeFooter">
        <span className="tag tagBlue">{outcome.short}</span>

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
    </div>
  );
}
