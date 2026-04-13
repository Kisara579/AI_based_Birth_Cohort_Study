import { useEffect, useMemo, useState, useRef } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import {
  fetchOutcomes,
  fetchModels,
  fetchSchema,
  predictOutcome,
} from "../api/modelsApi.js";
import "./ResultsModel.css";

export default function ResultsModel() {
  const { outcomeId } = useParams();
  const location = useLocation();

  // outcomes
  const [outcomes, setOutcomes] = useState([]);
  const [loadingOutcomes, setLoadingOutcomes] = useState(true);
  const [outcomesErr, setOutcomesErr] = useState("");

  const [modelsInfo, setModelsInfo] = useState(null);
  const [schema, setSchema] = useState(null);

  // prediction
  const [predLoading, setPredLoading] = useState(false);
  const [predErr, setPredErr] = useState("");
  const [prediction, setPrediction] = useState(null);

  const outcome = useMemo(
    () => outcomes.find((o) => o.id === outcomeId),
    [outcomes, outcomeId],
  );

  const outcomeDisplayName = outcome?.name || outcomeId.toUpperCase();

  const formValues = location.state?.formValues;

  // Map field.name -> field.label
  const fieldLabelMap = useMemo(() => {
    const map = {};
    for (const section of schema?.sections ?? []) {
      for (const field of section.fields ?? []) {
        map[field.name] = field.label || field.name;
      }
    }
    return map;
  }, [schema]);

  // Map field.name -> Map(optionValue -> optionLabel)
  const fieldOptionMap = useMemo(() => {
    const map = {};
    for (const section of schema?.sections ?? []) {
      for (const field of section.fields ?? []) {
        if (field.options?.length) {
          const optMap = {};
          for (const opt of field.options) {
            // store both string and original key for safe lookup
            optMap[String(opt.value)] = opt.label;
          }
          map[field.name] = optMap;
        }
      }
    }
    return map;
  }, [schema]);

  // load outcomes (to display name/short)
  useEffect(() => {
    let cancelled = false;

    async function loadBasics() {
      setLoadingOutcomes(true);
      setOutcomesErr("");
      try {
        const [outs, modelInfo, schemaData] = await Promise.all([
          fetchOutcomes(),
          fetchModels(outcomeId),
          fetchSchema(outcomeId),
        ]);
        if (!cancelled) {
          setOutcomes(outs);
          setModelsInfo(modelInfo);
          setSchema(schemaData);
        }
      } catch (e) {
        if (!cancelled) setOutcomesErr(e.message || "Failed to load outcomes");
      } finally {
        if (!cancelled) setLoadingOutcomes(false);
      }
    }

    loadBasics();
    return () => {
      cancelled = true;
    };
  }, [outcomeId]);

  // call backend prediction when we have inputs
  useEffect(() => {
    let cancelled = false;

    async function runPrediction() {
      if (!formValues) return;

      setPredLoading(true);
      setPredErr("");
      setPrediction(null);

      try {
        const effectiveModel = modelsInfo?.active;
        const res = await predictOutcome(outcomeId, formValues, effectiveModel);

        if (res?.error) throw new Error(res.error);

        if (!cancelled) setPrediction(res);
      } catch (e) {
        if (!cancelled) setPredErr(e.message || "Prediction failed");
      } finally {
        if (!cancelled) setPredLoading(false);
      }
    }

    if (formValues && !loadingOutcomes) {
      runPrediction();
    }

    return () => {
      cancelled = true;
    };
  }, [outcomeId, formValues, modelsInfo, loadingOutcomes]);

  // SVG circle gauge values (must be before early returns for hooks)
  const prob = prediction?.probability ?? 0;
  const pct = prob * 100;
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const targetOffset = circumference - (pct / 100) * circumference;

  // Animate circle fill from empty to target
  const [animatedOffset, setAnimatedOffset] = useState(circumference);
  const [displayPct, setDisplayPct] = useState(0);
  const animFrameRef = useRef(null);

  useEffect(() => {
    // Only animate when prediction is visible (not loading)
    if (!prediction || predLoading) {
      setAnimatedOffset(circumference);
      setDisplayPct(0);
      return;
    }

    // Reset then animate after a short delay
    setAnimatedOffset(circumference);
    setDisplayPct(0);

    const timer = setTimeout(() => {
      setAnimatedOffset(targetOffset);

      // Animate counter number
      const duration = 1200;
      const startTime = performance.now();
      const animate = (now) => {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // easeOutQuart
        const eased = 1 - Math.pow(1 - progress, 4);
        setDisplayPct(eased * pct);
        if (progress < 1) {
          animFrameRef.current = requestAnimationFrame(animate);
        }
      };
      animFrameRef.current = requestAnimationFrame(animate);
    }, 300);

    return () => {
      clearTimeout(timer);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [prediction, predLoading]);

  // ui statuses

  if (loadingOutcomes) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Results</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Prediction Results</h2>
            <p className="pageSubtitle">Loading outcomes...</p>
          </div>
        </div>
        <LoadingSkeleton lines={5} />
      </div>
    );
  }

  if (outcomesErr) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Results</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Prediction Results</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcomeDisplayName}</b>
            </p>
          </div>
          <Link to="/">
            <Button variant="ghost">← Back to Overview</Button>
          </Link>
        </div>
        <ErrorBanner title="Could not load outcomes" message={outcomesErr} />
      </div>
    );
  }

  if (!outcome) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Results</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Prediction Results</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcomeDisplayName}</b>
            </p>
          </div>
          <Link to="/">
            <Button variant="ghost">← Back to Overview</Button>
          </Link>
        </div>
        <ErrorBanner
          title="Unknown outcome"
          message={`No outcome found for id: "${outcomeId}"`}
        />
      </div>
    );
  }

  // If user opened results directly (no form values passed)
  if (!formValues) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Results</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Prediction Results</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcome.name}</b>
            </p>
          </div>
        </div>
        <ErrorBanner
          title="No input data"
          message="Please run a prediction first. This page expects form inputs from the Predict screen."
        />
        <div className="resultsActionRow">
          <Link to={`/predict/${outcome.id}`}>
            <Button variant="primary">Go to Predict</Button>
          </Link>
          <Link to="/">
            <Button variant="ghost">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }

  // Prediction loading / error
  if (predLoading) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Results</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Prediction Results</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcome.name}</b>
            </p>
          </div>
        </div>
        <LoadingSkeleton lines={5} />
      </div>
    );
  }

  if (predErr) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Results</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Prediction Results</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcome.name}</b>
            </p>
          </div>
        </div>
        <ErrorBanner title="Prediction unavailable" message={predErr} />

        <div className="resultsActionRow">
          <Link to={`/predict/${outcome.id}`}>
            <Button variant="secondary">Edit Inputs</Button>
          </Link>

          <Link to="/">
            <Button variant="ghost">Back to Overview</Button>
          </Link>
        </div>
      </div>
    );
  }


  const riskLabel = prediction?.risk_label ?? "Unknown";

  // Colour band
  let riskClass = "riskLow";
  if (pct >= 70) {
    riskClass = "riskHigh";
  } else if (pct >= 40) {
    riskClass = "riskMedium";
  }

  const effectiveModel = modelsInfo?.active_model || "";
  const modelLabel =
    modelsInfo?.models?.find((m) => m.algorithm === effectiveModel)?.label ||
    effectiveModel ||
    "Active Model";

  return (
    <div>
      {/* Breadcrumb */}
      <nav className="breadcrumb">
        <a href="/" className="breadcrumbLink">
          Overview
        </a>
        <span className="breadcrumbSep">›</span>
        <a href={`/predict/${outcome.id}`} className="breadcrumbLink">
          Predict
        </a>
        <span className="breadcrumbSep">›</span>
        <span className="breadcrumbCurrent">Results</span>
      </nav>

      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">Prediction Results</h2>
          <p className="pageSubtitle">
            Outcome: <b>{outcome.name}</b>{" "}
            <span className="tag tagBlue tagMl">{outcome.short}</span>
          </p>
        </div>
        <div className="resultsHeaderActions">
          <Link to={`/predict/${outcome.id}`}>
            <Button variant="secondary">Edit Inputs</Button>
          </Link>
          <Link to="/">
            <Button variant="ghost">Overview</Button>
          </Link>
        </div>
      </div>

      <Card
        title="Prediction Result"
        subtitle="Model output from backend API"
        className="resultPredictionCard"
      >
        <div className="resultHero">
          {/* Circular gauge */}
          <div className="probCircleWrap">
            <svg className="probCircleSvg" viewBox="0 0 110 110">
              <circle className="probCircleBg" cx="55" cy="55" r={radius} />
              <circle
                className={`probCircleFill stroke-${riskClass}`}
                cx="55"
                cy="55"
                r={radius}
                strokeDasharray={circumference}
                strokeDashoffset={animatedOffset}
              />
            </svg>
            <div className="probCircleText">
              <span className={`probCirclePct text-${riskClass}`}>
                {displayPct.toFixed(1)}%
              </span>
              <span className="probCircleSmall">probability</span>
            </div>
          </div>

          {/* Meta */}
          <div className="resultMeta">
            <div className="resultMetaRow">
              <span className="resultMetaLabel">Risk Category</span>
              <span className={`riskBadge ${riskClass}`}>
                {riskClass === "riskHigh"
                  ? "⚠ "
                  : riskClass === "riskMedium"
                    ? "◐ "
                    : "✓ "}
                {riskLabel}
              </span>
            </div>

            <div className="resultMetaRow">
              <span className="resultMetaLabel">Predicted Probability</span>
              <span className="resultMetaValue">{displayPct.toFixed(2)}%</span>
              <div className="progressBarWrap">
                <div
                  className={`progressBarFill bg-${riskClass}`}
                  style={{ width: `${displayPct}%` }}
                />
              </div>
            </div>

            <div className="resultMetaRow">
              <span className="resultMetaLabel">Outcome Model</span>
              <span className="resultMetaValue resultMetaValueSm">
                {outcome.name}
              </span>
            </div>

            <div className="resultMetaRow">
              <span className="resultMetaLabel">Active Algorithm Used</span>
              <span className="resultMetaValue resultMetaValueSm">
                {modelLabel}
              </span>
            </div>
          </div>
        </div>
      </Card>

      {/* Submitted inputs - clean grid instead of raw JSON */}
      <Card
        title="Submitted Inputs"
        subtitle="Values sent to the prediction model"
        className="resultInputsCard resultsSectionSpacing"
      >
        <div className="inputsGrid">
          {Object.entries(formValues).map(([k, v]) => {
            const label = fieldLabelMap[k] || k.replace(/_/g, " ");
            const optMap = fieldOptionMap[k];
            const display = optMap
              ? (optMap[String(v)] ?? String(v))
              : String(v);
            return (
              <div key={k} className="inputCell">
                <div className="inputCellKey">{label}</div>
                <div className="inputCellVal">{display}</div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
