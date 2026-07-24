import { useState } from "react";
import "./About.css";

export default function About() {
  const [activeReferences, setActiveReferences] = useState(null);
  const [showPriorityModal, setShowPriorityModal] = useState(false);
  const [showOverfitModal, setShowOverfitModal] = useState(false);

  const openReferences = (sectionKey) => {
    setActiveReferences(sectionKey);
  };

  const closeReferences = () => {
    setActiveReferences(null);
  };

  const features = [
    {
      title: "Real-time Inference",
      text: "FastAPI backend delivers low-latency predictions directly from trained scikit-learn pipelines stored on disk - no cloud dependency needed.",
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



  const algorithms = [
    { name: "XGBoost" },
    { name: "SVM (Support Vector Machine)" },
    { name: "Logistic Regression" },
    { name: "Random Forest" },
    { name: "Decision Tree" },
    { name: "Gradient Boosting" },
    { name: "Extremely Randomized Trees (XRT)" },
    { name: "K-Nearest Neighbors (KNN)" },
    { name: "LightGBM" },
    { name: "CatBoost" },
    { name: "Bagging Classifier" },
    { name: "Voting & Stacking Classifiers" },
  ];

  const teamMembers = [
    {
      name: "Sanjana Kaushalya",
      role: "Computer Science Undergraduate",
      description: "CS/2022/047",
      initials: "SK",
      github: "https://github.com/",
      linkedin: "https://linkedin.com/",
    },
    {
      name: "Harshani Pramodya",
      role: "Computer Science Undergraduate",
      description: "CS/2022/054",
      initials: "HP",
      github: "https://github.com",
      linkedin: "https://linkedin.com",
    },
    {
      name: "Vishmini Dinara",
      role: "Computer Science Undergraduate",
      description: "CS/2022/031",
      initials: "VD",
      github: "https://github.com",
      linkedin: "https://linkedin.com",
    },
    {
      name: "Dilsha Shehara",
      role: "Computer Science Undergraduate",
      description: "CS/2022/034",
      initials: "HP",
      github: "https://github.com",
      linkedin: "https://linkedin.com",
    },
    {
      name: "Paviru Lishara",
      role: "Computer Science Undergraduate",
      description: "CS/2022/068",
      initials: "PL",
      github: "https://github.com",
      linkedin: "https://linkedin.com",
    },
    {
      name: "Kisara Beddawala ",
      role: "Computer Science Undergraduate",
      description: "CS/2022/069",
      initials: "KB",
      github: "https://github.com",
      linkedin: "https://linkedin.com",
    },
  ];
  const overfittingReferences = [
    {
      title: "A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging",
      authors: "Bradshaw, T. J., Huemann, Z., Hu, J., & Rahmim, A.",
      year: "2023",
      journal: "Radiology: Artificial Intelligence",
      link: "https://doi.org/10.1148/ryai.220232"
    },
    {
      title: "Twelve key challenges in medical machine learning and solutions",
      authors: "Ellis, R. J., Sander, R. M., & Limon, A.",
      year: "2022",
      journal: "Intelligence-Based Medicine",
      link: "https://www.sciencedirect.com/science/article/pii/S2666521222000217"
    },
    {
      title: "Machine learning for medical imaging: methodological failures and recommendations for the future",
      authors: "Varoquaux, G., & Cheplygina, V.",
      year: "2022",
      journal: "npj Digital Medicine",
      link: "https://www.nature.com/articles/s41746-022-00592-y"
    }
  ];

  const selectionReferences = [
    {
      title: "Prediction of high-risk pregnancy based on machine learning algorithms",
      authors: "Pi, X., Wang, J., Chu, L., Zhang, G., & Zhang, W.",
      year: "2025",
      journal: "Scientific Reports",
      link: "https://pmc.ncbi.nlm.nih.gov/articles/PMC12049423/#:~:text=Six%20machine%20learning%20algorithms%E2%80%94multilayer,employed%20to%20construct%20predictive%20models."
    },
    {
      title: "Evaluation",
      authors: "Simon, G. J., & Aliferis, C.",
      year: "2024",
      journal: "Artificial Intelligence and Machine Learning in Health Care and Medical Sciences: Best Practices and Pitfalls",
      link: "https://www.ncbi.nlm.nih.gov/books/NBK610543/"
    },
    {
      title: "Evaluation of machine learning algorithms for health and wellness applications: A tutorial",
      authors: "Tohka, J., & van Gils, M.",
      year: "2021",
      journal: "Computers in Biology and Medicine",
      link: "https://www.sciencedirect.com/science/article/pii/S0010482521001189"
    },
    {
      title: "Decision Analytics and Machine Learning Model Selection for Healthcare Predictions",
      authors: "Elsevier Research Consortium",
      year: "2024",
      journal: "Decision Analytics Journal",
      link: "https://www.sciencedirect.com/science/article/pii/S2772662224001279"
    }
  ];

  return (
    <div>
      {/* hero */}
      <div className="heroSection">
        <h1 className="heroTitle">About This Project</h1>
        <p className="heroSubtitle">
          An advanced AI-driven platform designed for birth cohort research. It integrates robust clinical machine learning models with an intuitive web interface to provide real-time perinatal risk assessment and outcome prediction.
        </p>
      </div>

      <div className="heroDivider" />

      {/* capabilities */}
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



      {/* algorithms section */}
      <div className="aboutSection">
        <div className="sectionHeaderContainer">
          <span className="sectionNumber">Machine Learning</span>
          <h2 className="sectionMainTitle">Algorithms Used</h2>
          <p className="sectionDescription">
            Predictive pipelines leverage a wide array of supervised machine learning classifiers, rigorously evaluated and tuned using cohort dataset models.
          </p>
        </div>

        <div className="algorithmsGrid">
          {algorithms.map((a) => (
            <div key={a.name} className="card algoCard" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px', textAlign: 'center' }}>
              <h4 className="algoTitle" style={{ margin: 0 }}>{a.name}</h4>
            </div>
          ))}
        </div>
      </div>

      {/* validation & selection methodology section */}
      <div className="aboutSection">
        <div className="sectionHeaderContainer">
          <h2 className="sectionMainTitle">Validation & Model Selection Methodology</h2>
          <p className="sectionDescription">
            How we detect overfitting, enforce model generalization, and select the optimal clinical prediction pipeline.
          </p>
        </div>

        <div className="methodologyGrid">
          <div className="card methodCard">
            <div className="methodIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </div>
            <h3 className="methodTitle">Overfitting Detection & Validation</h3>
            <p className="methodText">
              To prevent models from memorizing training data, we employ Stratified 5-Fold Cross-Validation, ensuring consistent class ratios. We actively monitor learning curves to verify that the generalization gap between train and validation scores remains minimal.
            </p>
            <div className="methodCardActions">
              <button className="referenceBtn" onClick={() => setShowOverfitModal(true)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="btnIcon">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                More Details
              </button>
              <button className="referenceBtn" onClick={() => openReferences("overfitting")}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="btnIcon">
                  <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                References
              </button>
            </div>
          </div>

          <div className="card methodCard">
            <div className="methodIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <h3 className="methodTitle">Best Model Selection Framework</h3>
            <p className="methodText">
              From the non-overfitted candidate algorithms, the best model is selected based on a structured clinical metric hierarchy. Because maternal and neonatal outcomes require minimizing false negatives, we prioritize Recall and balanced metrics over raw correctness.
            </p>
            <div className="methodCardActions">
              <button className="referenceBtn" onClick={() => setShowPriorityModal(true)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="btnIcon">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                More Details
              </button>
              <button className="referenceBtn" onClick={() => openReferences("selection")}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="btnIcon">
                  <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                References
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* tech stack */}
      <div className="aboutTechSection">
        <div className="card">
          <h3 className="aboutCardTitle">Technology Stack</h3>
          <div className="aboutTechTags">
            {[
              "Python 3.11",
              "FastAPI",
              "Uvicorn",
              "scikit-learn",
              "XGBoost",
              "LightGBM",
              "CatBoost",
              "pandas",
              "NumPy",
              "Optuna",
              "SHAP",
              "imbalanced-learn",
              "React 19",
              "Vite",
              "React Router",
              "Chart.js",
            ].map((t) => (
              <span key={t} className="tag tagBlue">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* team section */}
      <div className="aboutSection">
        <div className="sectionHeaderContainer">
          <span className="sectionNumber">People</span>
          <h2 className="sectionMainTitle">Team Members</h2>
          <p className="sectionDescription">
            Researchers and developers behind the AI-based Birth Cohort Study outcome prediction system.
          </p>
        </div>

        <div className="teamGrid">
          {teamMembers.map((m, idx) => (
            <div key={idx} className="card teamCard">
              <div className="teamAvatar">{m.initials}</div>
              <h4 className="teamName">{m.name}</h4>
              <p className="teamRole">{m.role}</p>
              <p className="teamDescription">{m.description}</p>
              <div className="teamSocials">
                <a
                  href={m.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="socialLink"
                  title="GitHub Profile"
                >
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
                    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
                  </svg>
                </a>
                <a
                  href={m.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="socialLink"
                  title="LinkedIn Profile"
                >
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
                    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
                    <rect x="2" y="9" width="4" height="12" />
                    <circle cx="4" cy="4" r="2" />
                  </svg>
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* References Modal */}
      {activeReferences && (
        <div className="modalOverlay" onClick={closeReferences}>
          <div className="modalContent" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <h3 className="modalTitle">
                {activeReferences === "overfitting"
                  ? "Overfitting Mitigation & Validation References"
                  : "Model Selection Framework References"}
              </h3>
              <button className="modalCloseBtn" onClick={closeReferences}>&times;</button>
            </div>
             <div className="modalBody">
              {(activeReferences === "overfitting" ? overfittingReferences : selectionReferences).map((ref, idx) => (
                <div key={idx} className="refItem" style={{flexDirection: 'column', alignItems: 'stretch'}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '16px'}}>
                    <div className="refCitation">
                      <strong>{ref.title}</strong>
                      <p className="refAuthors">{ref.authors} ({ref.year}). <em>{ref.journal}</em>.</p>
                    </div>
                    <a href={ref.link} target="_blank" rel="noopener noreferrer" className="refLinkBtn">
                      View Paper
                      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginLeft: '6px'}}>
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
                      </svg>
                    </a>
                  </div>
                  {ref.evidence && (
                    <div className="refEvidence">
                      <strong>Research Evidence:</strong> {ref.evidence}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Priority Modal */}
      {showPriorityModal && (
        <div className="modalOverlay" onClick={() => setShowPriorityModal(false)}>
          <div className="modalContent priorityModalContent" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <h3 className="modalTitle">Selection Metric Priority Hierarchy</h3>
              <button className="modalCloseBtn" onClick={() => setShowPriorityModal(false)}>&times;</button>
            </div>
            <div className="modalBody">
              <p className="modalIntroText">
                To select the active predictive model, the development team uses a strict clinical metric hierarchy. This ensures that the selected model prioritizes minimizing high-risk clinical misses while maintaining optimal probability calibration and classification performance.
              </p>
              
              <div className="metricPriorityContainer modalTableContainer">
                <table className="priorityTable">
                  <thead>
                    <tr>
                      <th>Priority</th>
                      <th>Metric</th>
                      <th>Clinical Rationale & Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><span className="priorityBadge badge1">1</span></td>
                      <td>
                        <strong>Recall</strong>
                        <div className="metricSubtext">(Sensitivity)</div>
                      </td>
                      <td>
                        <div className="rationaleBrief">Missing a patient with the condition (false negative) has serious clinical consequences.</div>
                        <div className="rationaleDetail">
                          <strong>Formula:</strong> TP / (TP + FN)
                          <br />
                          <strong>Clinical Importance:</strong> In clinical prediction (e.g., preeclampsia, PPH, or NICU admission), failing to identify a patient at risk can delay critical preventative care. High recall ensures the system captures as many true positive cases as possible.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge2">2</span></td>
                      <td>
                        <strong>F1-score</strong>
                        <div className="metricSubtext">(Harmonic Mean)</div>
                      </td>
                      <td>
                        <div className="rationaleBrief">Balances Recall and Precision, especially useful when classes are imbalanced.</div>
                        <div className="rationaleDetail">
                          <strong>Formula:</strong> 2 &times; (Precision &times; Recall) / (Precision + Recall)
                          <br />
                          <strong>Clinical Importance:</strong> Adverse perinatal outcomes are relatively rare (high class imbalance). F1-score prevents selecting a model that achieves high recall simply by predicting everyone as high-risk, which would make the model clinically useless.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge3">3</span></td>
                      <td>
                        <strong>ROC-AUC</strong>
                        <div className="metricSubtext">(Discrimination)</div>
                      </td>
                      <td>
                        <div className="rationaleBrief">Measures the model's ability to distinguish between positive and negative cases across thresholds.</div>
                        <div className="rationaleDetail">
                          <strong>Definition:</strong> Area Under the Receiver Operating Characteristic curve.
                          <br />
                          <strong>Clinical Importance:</strong> Evaluates overall model discrimination power. A high ROC-AUC demonstrates that the risk scores assigned by the model are well-ordered (i.e. patients with adverse outcomes get higher scores than normal patients).
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge4">4</span></td>
                      <td>
                        <strong>Precision</strong>
                        <div className="metricSubtext">(Positive Predictive Value)</div>
                      </td>
                      <td>
                        <div className="rationaleBrief">Reduces false positive predictions.</div>
                        <div className="rationaleDetail">
                          <strong>Formula:</strong> TP / (TP + FP)
                          <br />
                          <strong>Clinical Importance:</strong> Keeping precision reasonable reduces "alarm fatigue" for clinical staff and avoids triggering unnecessary anxiety, screening procedures, or follow-ups for healthy families.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge5">5</span></td>
                      <td>
                        <strong>Accuracy</strong>
                        <div className="metricSubtext">(Overall Correctness)</div>
                      </td>
                      <td>
                        <div className="rationaleBrief">Overall correctness, but should not be the sole criterion.</div>
                        <div className="rationaleDetail">
                          <strong>Formula:</strong> (TP + TN) / (TP + TN + FP + FN)
                          <br />
                          <strong>Clinical Importance:</strong> On highly imbalanced data, accuracy is misleadingly high (e.g. 98% accuracy by predicting "no risk" everywhere). Thus, accuracy is only checked as a general baseline sanity metric.
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              
              <div className="modalDetailsSection">
                <h4 className="detailsSubtitle">Generalization & Calibration Safeguards</h4>
                <p className="detailsText">
                  Before applying this priority rank, our automated pipeline first filters out models that exhibit signs of overfitting (where the train-to-validation metric gap exceeds 5%). Ties between models are resolved by selecting the one with the superior Brier score (measuring probability calibration) to ensure the output risk percentages are reliable.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Overfit Details Modal */}
      {showOverfitModal && (
        <div className="modalOverlay" onClick={() => setShowOverfitModal(false)}>
          <div className="modalContent priorityModal" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <h3 className="modalTitle">Overfitting Categorization Rules</h3>
              <button className="modalCloseBtn" onClick={() => setShowOverfitModal(false)}>
                ✕
              </button>
            </div>
            
            <div className="modalBody">
              <p className="modalIntro">
                Models are strictly categorized based on their "Generalization Gap"—the difference between their training performance and validation performance.
              </p>
              
              <div className="priorityTableContainer">
                <table className="priorityTable">
                  <thead>
                    <tr>
                      <th>Status</th>
                      <th>Condition</th>
                      <th>Meaning</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><span className="priorityBadge badge1" style={{backgroundColor: "rgba(16,185,129,0.15)", color: "#10b981"}}>Good</span></td>
                      <td>Max Gap ≤ 3%</td>
                      <td>
                        <div className="rationaleDetail">
                          <strong>Well Generalized:</strong> Model performs consistently on both seen and unseen data.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge2" style={{backgroundColor: "rgba(245,158,11,0.15)", color: "#f59e0b"}}>Warning</span></td>
                      <td>Max Gap between 3% and 5%</td>
                      <td>
                        <div className="rationaleDetail">
                          <strong>Mild Overfitting:</strong> Slight generalization gap but remains under the safety threshold.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge3" style={{backgroundColor: "rgba(239,68,68,0.15)", color: "#ef4444"}}>Danger</span></td>
                      <td>Max Gap &gt; 5%</td>
                      <td>
                        <div className="rationaleDetail">
                          <strong>Overfitting:</strong> The model has started to memorize the training data and is losing its ability to generalize.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge4" style={{backgroundColor: "rgba(220,38,38,0.25)", color: "#dc2626"}}>Critical</span></td>
                      <td>Max Gap &gt; 10%</td>
                      <td>
                        <div className="rationaleDetail">
                          <strong>Severe Overfitting:</strong> Very unstable model. Highly restricted from being selected as the active clinical model.
                        </div>
                      </td>
                    </tr>
                    <tr>
                      <td><span className="priorityBadge badge5" style={{backgroundColor: "rgba(168,85,247,0.15)", color: "#a855f7"}}>Warning</span></td>
                      <td>Min Gap &lt; -5%</td>
                      <td>
                        <div className="rationaleDetail">
                          <strong>Underfitting / Split Instability:</strong> Validation performance is suspiciously better than training, indicating potential data leakage or an unrepresentative validation split.
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
