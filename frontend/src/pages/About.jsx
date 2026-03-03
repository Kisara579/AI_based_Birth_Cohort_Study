export default function About() {
  const features = [
    {
      title: "Real-time Inference",
      text: "FastAPI backend delivers low-latency predictions directly from trained scikit-learn pipelines stored on disk — no cloud dependency needed.",
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
    },
    {
      title: "Model Transparency",
      text: "Every model exposes full classification reports, per-class precision/recall/F1, and a confusion matrix so clinicians can assess reliability before use.",
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
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <line x1="3" y1="9" x2="21" y2="9" />
          <line x1="9" y1="21" x2="9" y2="9" />
        </svg>
      ),
    },
    {
      title: "Research Grade",
      text: "Built for birth-cohort research with reproducible training pipelines, versioned models, and structured data preprocessing for auditability.",
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
          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
        </svg>
      ),
    },
  ];

  return (
    <div>
      {/* hero */}
      <div className="heroSection">
        <div className="heroBadge">
          <span className="heroBadgeDot" />
          Research Dashboard
        </div>
        <h1 className="heroTitle">About This Project</h1>
        <p className="heroSubtitle">
          An AI-powered perinatal outcome prediction platform built for birth
          cohort research — combining clinical ML models with an interactive web
          interface for real-time risk assessment.
        </p>
      </div>

      <div className="heroDivider" />

      {/* tech stack */}
      <div className="sectionLabel">
        <span className="sectionTitle">Key Capabilities</span>
      </div>

      <div className="aboutGrid">
        {features.map((f) => (
          <div key={f.title} className="card">
            <div className="aboutFeatureIcon">{f.icon}</div>
            <h3 className="aboutCardTitle">{f.title}</h3>
            <p className="aboutCardText">{f.text}</p>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 32 }}>
        <div className="card">
          <h3 className="aboutCardTitle" style={{ marginBottom: 10 }}>
            Tech Stack
          </h3>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {[
              "Python 3.11",
              "FastAPI",
              "scikit-learn",
              "XGBoost",
              "pandas",
              "React 19",
              "Vite",
              "React Router",
            ].map((t) => (
              <span key={t} className="tag tagBlue">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
