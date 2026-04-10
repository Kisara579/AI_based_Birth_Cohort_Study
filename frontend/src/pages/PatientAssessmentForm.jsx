import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import DynamicForm from "../components/forms/DynamicForm.jsx";
import { fetchOutcomes } from "../api/modelsApi.js";
import { universalSchema, demoPatient1, demoPatient2 } from "../utils/UniversalSchema.js";
import { buildAllPayloads } from "../utils/predictionMapper.js";

import "./PatientAssessmentForm.css";

export default function PatientAssessmentForm() {
  const navigate = useNavigate();

  const [outcomes, setOutcomes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const [formKey, setFormKey] = useState(0);
  const [initialValues, setInitialValues] = useState({});

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setErr("");
      try {
        const outs = await fetchOutcomes();
        if (!cancelled) setOutcomes(outs);
      } catch (e) {
        if (!cancelled) setErr(e.message || "Failed to load outcomes");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  /* Autofill handler  */
  function handleAutofill(demoData) {
    setInitialValues(demoData);
    setFormKey((k) => k + 1);
  }

  /* Submit handler  */
  function handleSubmit(values) {
    if (outcomes.length === 0) return;

    // Build per-model payloads from the universal form data
    const payloads = buildAllPayloads(outcomes, values);

    navigate("/results", {
      state: { payloads, outcomes },
    });
  }

  /*  Renders  */
  if (loading) {
    return (
      <div className="assessmentPage">
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Patient Assessment</h2>
            <p className="pageSubtitle">Initializing system…</p>
          </div>
        </div>
        <LoadingSkeleton lines={8} />
      </div>
    );
  }

  if (err) {
    return (
      <div className="assessmentPage">
        <div className="pageHeader">
          <div><h2 className="pageTitle">Patient Assessment</h2></div>
        </div>
        <ErrorBanner title="Could not load system" message={err} />
      </div>
    );
  }

  return (
    <div className="assessmentPage">
      {/* Breadcrumb */}
      <nav className="breadcrumb">
        <a href="/" className="breadcrumbLink">Overview</a>
        <span className="breadcrumbSep">›</span>
        <span className="breadcrumbCurrent">Predict</span>
      </nav>

      {/* Hero header */}
      <div className="assessmentHero">
        <div className="heroLeft">
          <h1 className="assessmentTitle">Patient Risk Assessment</h1>
          <p className="assessmentSubtitle">
            Complete the clinical profile below. When you click <strong>Predict All Risks</strong>,
            the system will run <strong>{outcomes.length} AI models</strong> simultaneously and
            display a comprehensive risk dashboard.
          </p>
        </div>
      </div>

      {/* Demo autofill card */}
      <div className="demoBanner">
        <div className="demoBannerInner">
          <div className="demoBannerText">
            <span className="demoBannerIcon">⚡</span>
            <span><strong>Quick Demo</strong> - Autofill the form with sample patient data</span>
          </div>
          <div className="demoBannerActions">
            <button className="demoBtn" onClick={() => handleAutofill(demoPatient1)}>
              <span className="demoBtnDot p1" />
              Patient 1
            </button>
            <button className="demoBtn" onClick={() => handleAutofill(demoPatient2)}>
              <span className="demoBtnDot p2" />
              Patient 2
            </button>
          </div>
        </div>
      </div>

      {/* Main form */}
      <Card title={universalSchema.title} className="assessmentFormCard">
        <DynamicForm
          key={formKey}
          schema={universalSchema}
          initialValues={initialValues}
          onSubmit={handleSubmit}
          submitLabel="Predict All Risks"
        />
      </Card>
    </div>
  );
}
