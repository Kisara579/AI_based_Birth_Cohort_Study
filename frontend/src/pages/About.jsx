import { useState } from "react";
import "./About.css";

export default function About() {
  const [activeReferences, setActiveReferences] = useState(null);

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

  const maternalOutcomes = [
    {
      id: "pph",
      num: "1",
      name: "Postpartum haemorrhage (PPH)",
      desc: "Heavy bleeding after delivery. Predicts high-risk patients using pre-delivery clinical features.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-13-7-13S5 10.7 5 15a7 7 0 0 0 7 7z" />
        </svg>
      ),
    },
    {
      id: "preeclampsia",
      num: "2",
      name: "Preeclampsia",
      desc: "Pregnancy-induced hypertensive disorder. Assesses risk using early maternal indicators.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
        </svg>
      ),
    },
    {
      id: "iugr",
      num: "3",
      name: "Intrauterine growth restriction (IUGR)",
      desc: "Delays in fetal growth in utero. Screens maternal risk factors to flag potential growth delays.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a5 5 0 1 0 5 5 5 5 0 0 0-5-5zm0 10c-4.4 0-8 2.2-8 5v3h16v-3c0-2.8-3.6-5-8-5z" />
        </svg>
      ),
    },
    {
      id: "gdm",
      num: "4",
      name: "Gestational diabetes",
      desc: "High glucose levels developing during pregnancy. Analyzes maternal metabolic baseline indicators.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" />
        </svg>
      ),
    },
    {
      id: "iol",
      num: "5",
      name: "Women requiring induction of labour (IOL)",
      desc: "Stimulation of uterine contractions before spontaneous labor to prevent birth complications.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
      ),
    },
    {
      id: "c_section",
      num: "6",
      name: "Caesarean section",
      desc: "Likelihood or clinical necessity of surgical delivery based on mother's health profile.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.5 17.5L3 6V3h3l11.5 11.5" />
          <path d="M13 19h8" />
        </svg>
      ),
    },
    {
      id: "ppd",
      num: "7",
      name: "Postpartum depression",
      desc: "Clinical depression occurring after birth. Screens psychosocial and demographic variables.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 14s1.5-2 4-2 4 2 4 2" />
          <line x1="9" y1="9" x2="9.01" y2="9" />
          <line x1="15" y1="9" x2="15.01" y2="9" />
        </svg>
      ),
    },
  ];

  const neonatalOutcomes = [
    {
      id: "lbw",
      num: "1",
      name: "Low birth weight",
      desc: "Birth weight below 2.5 kilograms. Analyzes maternal health and lifestyle risk factors.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 3h12M12 3v15M12 18a3 3 0 1 1-6 0c0-1.66 1.34-3 3-3h6c1.66 0 3 1.34 3 3a3 3 0 1 1-6 0z" />
        </svg>
      ),
    },
    {
      id: "pd",
      num: "2",
      name: "Premature delivery",
      desc: "Delivery occurring before 37 complete weeks of gestation, predicted via clinical history.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
          <line x1="16" y1="2" x2="16" y2="6" />
          <line x1="8" y1="2" x2="8" y2="6" />
          <line x1="3" y1="10" x2="21" y2="10" />
          <path d="M8 14h.01M12 14h.01M16 14h.01" />
        </svg>
      ),
    },
    {
      id: "nicu",
      num: "3",
      name: "Neonatal Intensive Care Unit (NICU) admission",
      desc: "Likelihood of a newborn requiring immediate intensive monitoring and medical intervention.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M12 8v8M8 12h8" />
        </svg>
      ),
    },
    {
      id: "feeding",
      num: "4",
      name: "Feeding problems",
      desc: "Difficulties in breastfeeding or formula intake in early postpartum weeks.",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 8h1a4 4 0 0 1 0 8h-1" />
          <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z" />
          <line x1="6" y1="2" x2="6" y2="4" />
          <line x1="10" y1="2" x2="10" y2="4" />
        </svg>
      ),
    },
  ];

  const algorithms = [
    {
      name: "XGBoost",
      isCore: true,
      desc: "Extreme Gradient Boosting. High-performance tree boosting method optimized for clinical tabular data classifications.",
    },
    {
      name: "SVM (Support Vector Machine)",
      isCore: true,
      desc: "Linear & kernel SVM models that compute maximum-margin decision hyperplanes for outcome boundary classification.",
    },
    {
      name: "Logistic Regression",
      isCore: true,
      desc: "Foundational probabilistic model that determines binary risk ratios using generalized linear functions.",
    },
    {
      name: "Random Forest",
      isCore: true,
      desc: "Ensemble of bagging decision trees that aggregates predictions to prevent overfitting and improve stability.",
    },
    {
      name: "Decision Tree",
      isCore: true,
      desc: "Base non-parametric model mapping features to hierarchical leaf nodes for explainable clinical decisions.",
    },
    {
      name: "Gradient Boosting",
      isCore: false,
      desc: "Sequential ensemble method that iteratively fits regression trees on residual error loss gradients.",
    },
    {
      name: "Extremely Randomized Trees (XRT)",
      isCore: false,
      desc: "Extra Trees model that increases splits randomness to reduce variance and capture non-linear relationships.",
    },
    {
      name: "K-Nearest Neighbors (KNN)",
      isCore: false,
      desc: "Instance-based classification algorithm that maps predictions to the majority vote of neighboring data points.",
    },
    {
      name: "LightGBM",
      isCore: false,
      desc: "Light Gradient Boosting Machine, utilizing leaf-wise tree growth for faster computation speed.",
    },
    {
      name: "Bagging Classifier",
      isCore: false,
      desc: "Meta-estimator fitting base classifiers (like Logistic Regression) on random bootstrapped training subsets.",
    },
    {
      name: "Voting & Stacking Classifiers",
      isCore: false,
      desc: "Hybrid meta-estimators combining predictions of multiple heterogeneous models via soft-voting and stacking.",
    },
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
      title: "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection",
      authors: "Kohavi, R.",
      year: "1995",
      journal: "International Joint Conference on Artificial Intelligence (IJCAI)",
      link: "https://dl.acm.org/doi/10.5555/1625855.1625980"
    },
    {
      title: "The Problem of Overfitting",
      authors: "Hawkins, D. M.",
      year: "2004",
      journal: "Journal of Chemical Information and Computer Sciences",
      link: "https://pubs.acs.org/doi/10.1021/ci034247d"
    },
    {
      title: "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms",
      authors: "Dietterich, T. G.",
      year: "1998",
      journal: "Neural Computation",
      link: "https://ieeexplore.ieee.org/document/6790518"
    }
  ];

  const selectionReferences = [
    {
      title: "The Precision-Recall Plot Is More Informative than the ROC Plot when Evaluating Binary Classifiers on Imbalanced Datasets",
      authors: "Saito, T., & Rehmsmeier, M.",
      year: "2015",
      journal: "PLOS ONE",
      link: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118595"
    },
    {
      title: "Predicting Good Probabilities with Supervised Learning",
      authors: "Niculescu-Mizil, A., & Caruana, R.",
      year: "2005",
      journal: "International Conference on Machine Learning (ICML)",
      link: "https://dl.acm.org/doi/10.1145/1102351.1102430"
    },
    {
      title: "Assessing the performance of prediction models: a framework for traditional and novel performance measures",
      authors: "Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., Pencina, M. J., & Kattan, M. W.",
      year: "2010",
      journal: "Epidemiology (Cambridge, Mass.)",
      link: "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3575184/"
    }
  ];

  return (
    <div>
      {/* hero */}
      <div className="heroSection">
        <div className="heroBadge">Research Dashboard</div>
        <h1 className="heroTitle">About This Project</h1>
        <p className="heroSubtitle">
          An AI-powered perinatal outcome prediction platform built for birth
          cohort research combining clinical ML models with an interactive web
          interface for real-time risk assessment.
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

      {/* outcomes section */}
      <div className="aboutSection">
        <div className="sectionHeaderContainer">
          <h2 className="sectionMainTitle">Maternal / Adverse Pregnancy Outcomes</h2>
          <p className="sectionDescription">
            Primary maternal health indicators monitored throughout the cohort study, predicting critical complications from early gestation up to immediate postpartum stages.
          </p>
        </div>

        <div className="outcomesGrid">
          {maternalOutcomes.map((o) => (
            <div key={o.id} className="card outcomeCard">
              <span className="outcomeNumber">{o.num}</span>
              <div className="outcomeCardHeader">
                <div className="outcomeIcon">{o.icon}</div>
                <h4 className="outcomeTitle">{o.name}</h4>
              </div>
              <p className="outcomeText">{o.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="aboutSection">
        <div className="sectionHeaderContainer">
          <h2 className="sectionMainTitle">Neonatal Adverse Outcomes</h2>
          <p className="sectionDescription">
            Crucial adverse outcomes in newborns monitored to assess post-delivery health, development parameters, and intensive pediatric support requirements.
          </p>
        </div>

        <div className="outcomesGrid">
          {neonatalOutcomes.map((o) => (
            <div key={o.id} className="card outcomeCard">
              <span className="outcomeNumber">{o.num}</span>
              <div className="outcomeCardHeader">
                <div className="outcomeIcon">{o.icon}</div>
                <h4 className="outcomeTitle">{o.name}</h4>
              </div>
              <p className="outcomeText">{o.desc}</p>
            </div>
          ))}
        </div>
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
            <div key={a.name} className="card algoCard">
              <div className="algoHeader">
                <h4 className="algoTitle">{a.name}</h4>
                <span className={`algoBadge ${a.isCore ? "badgeCore" : "badgeOther"}`}>
                  {a.isCore ? "Core" : "Ensemble / Other"}
                </span>
              </div>
              <p className="algoText">{a.desc}</p>
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
            <button className="referenceBtn" onClick={() => openReferences("overfitting")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="btnIcon">
                <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              References
            </button>
          </div>

          <div className="card methodCard">
            <div className="methodIcon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
            </div>
            <h3 className="methodTitle">Best Model Selection Framework</h3>
            <p className="methodText">
              From the non-overfitted candidate algorithms, the best model is selected by evaluating clinical metrics. Because neonatal datasets are highly imbalanced, we prioritize PR-AUC and F1-scores over raw accuracy, and optimize Brier scores for probability calibration.
            </p>
            <button className="referenceBtn" onClick={() => openReferences("selection")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="btnIcon">
                <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              References
            </button>
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
              "scikit-learn",
              "XGBoost",
              "pandas",
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
                <div key={idx} className="refItem">
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
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
