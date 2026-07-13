import { useNavigate } from "react-router-dom";
import Button from "../common/Button.jsx";
import "./OutcomeCard.css";

const OUTCOME_ICONS = {
  pph: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-13-7-13S5 10.7 5 15a7 7 0 0 0 7 7z" />
    </svg>
  ),
  preeclampsia: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  ),
  iugr: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a5 5 0 1 0 5 5 5 5 0 0 0-5-5zm0 10c-4.4 0-8 2.2-8 5v3h16v-3c0-2.8-3.6-5-8-5z" />
    </svg>
  ),
  gdm: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </svg>
  ),
  iol: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  ),
  c_section: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.5 17.5L3 6V3h3l11.5 11.5" />
      <path d="M13 19h8" />
    </svg>
  ),
  ppd: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 14s1.5-2 4-2 4 2 4 2" />
      <line x1="9" y1="9" x2="9.01" y2="9" />
      <line x1="15" y1="9" x2="15.01" y2="9" />
    </svg>
  ),
  lbw: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 3h12M12 3v15M12 18a3 3 0 1 1-6 0c0-1.66 1.34-3 3-3h6c1.66 0 3 1.34 3 3a3 3 0 1 1-6 0z" />
    </svg>
  ),
  pd: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
      <path d="M8 14h.01M12 14h.01M16 14h.01" />
    </svg>
  ),
  nicu: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M12 8v8M8 12h8" />
    </svg>
  ),
  feeding: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8h1a4 4 0 0 1 0 8h-1" />
      <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z" />
      <line x1="6" y1="2" x2="6" y2="4" />
      <line x1="10" y1="2" x2="10" y2="4" />
    </svg>
  ),
};

const DEFAULT_ICON = (
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
  const id = outcome?.id ?? "";
  const icon = OUTCOME_ICONS[id] || DEFAULT_ICON;

  return (
    <div className="card outcomeCard">
      <div className="outcomeTop">
        <div className="outcomeCardHeader">
          <div className="outcomeIconWrap outcomeIconBlue">{icon}</div>
          <h2 className="cardTitle outcomeTitle">{name}</h2>
        </div>
        <p className="cardSubtitle outcomeDesc">{desc}</p>
      </div>

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
