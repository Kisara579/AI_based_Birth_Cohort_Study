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

  const name = (outcome?.name ?? "").trim();
  const desc = (outcome?.description ?? "").trim();
  const short = (outcome?.short ?? "").toString().trim();

  return (
    <div className="card outcomeCard">
      <div className="outcomeTop">
        {/* icon */}
        <div className="outcomeIconWrap outcomeIconBlue">{SVG_ICON}</div>

        {/* title */}
        <h2 className="cardTitle outcomeTitle">{name}</h2>
        
        <p className="cardSubtitle outcomeDesc">{desc}</p>
      </div>

      {/* footer */}
      <div className="outcomeFooter">
        <span className="tag tagBlue">{short}</span>

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
