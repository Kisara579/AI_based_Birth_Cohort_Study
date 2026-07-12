import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Card from "../components/common/Card.jsx";
import Button from "../components/common/Button.jsx";
import ErrorBanner from "../components/common/ErrorBanner.jsx";
import LoadingSkeleton from "../components/common/LoadingSkeleton.jsx";
import {
  fetchMetrics,
  fetchModels,
  fetchOutcomes,
  fetchSchema,
} from "../api/modelsApi.js";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar, Line, Doughnut } from "react-chartjs-2";
import "./MetricsModel.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Tooltip,
  Legend,
);

/* palette */
const METRIC_COLORS = {
  accuracy: { bg: "rgba(99,102,241,0.75)", border: "#818cf8" },
  f1: { bg: "rgba(16,185,129,0.75)", border: "#34d399" },
  precision: { bg: "rgba(245,158,11,0.75)", border: "#fbbf24" },
  recall: { bg: "rgba(236,72,153,0.75)", border: "#f472b6" },
  roc_auc: { bg: "rgba(56,189,248,0.75)", border: "#38bdf8" },
};

const FI_GRADIENT_COLORS = ["rgba(99,102,241,0.85)", "rgba(129,140,248,0.7)"];

/* chart configs */
const DARK_GRID = "rgba(255,255,255,0.1)";
const DARK_TICK = "rgba(255,255,255,0.85)";
const LIGHT_GRID = "rgba(15,23,42,0.1)";
const LIGHT_TICK = "#0f172a";

/* Confusion Matrix Heatmap (canvas) */
function ConfusionHeatmap({ matrix, isDarkMode }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !matrix || matrix.length < 2) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    const LABEL_W = 60;
    const LABEL_H = 40;
    const CELL = 90;
    const PAD = 6;
    const W = LABEL_W + CELL * 2 + PAD * 3;
    const H = LABEL_H + CELL * 2 + PAD * 3 + 28;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, W, H);

    // axis labels
    ctx.fillStyle = isDarkMode
      ? "rgba(255,255,255,0.85)"
      : "rgba(15,23,42,0.7)";
    ctx.font = "600 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Predicted →", LABEL_W + CELL + PAD, 16);

    ctx.save();
    ctx.translate(14, LABEL_H + CELL + PAD);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Actual →", 0, 0);
    ctx.restore();

    // class labels top
    ctx.fillStyle = isDarkMode
      ? "rgba(255,255,255,0.85)"
      : "rgba(15,23,42,0.6)";
    ctx.font = "700 13px Inter, sans-serif";
    ctx.textAlign = "center";
    const y0 = 28 + LABEL_H / 2 + 4;
    ctx.fillText("0", LABEL_W + PAD + CELL / 2, y0);
    ctx.fillText("1", LABEL_W + PAD * 2 + CELL + CELL / 2, y0);

    // class labels left
    const x0 = LABEL_W - 8;
    ctx.textAlign = "right";
    ctx.fillText("0", x0, 28 + LABEL_H + PAD + CELL / 2 + 5);
    ctx.fillText("1", x0, 28 + LABEL_H + PAD * 2 + CELL + CELL / 2 + 5);

    // compute range
    const flat = matrix.flat();
    const maxVal = Math.max(...flat, 1);

    const cells = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];

    cells.forEach(([r, c]) => {
      const val = matrix[r][c];
      const ratio = val / maxVal;
      const isDiag = r === c;

      const cx = LABEL_W + PAD + c * (CELL + PAD);
      const cy = 28 + LABEL_H + PAD + r * (CELL + PAD);

      // background
      if (isDiag) {
        const alpha = 0.15 + ratio * 0.55;
        ctx.fillStyle = `rgba(99,102,241,${alpha})`;
      } else {
        const alpha = 0.08 + ratio * 0.4;
        ctx.fillStyle = `rgba(239,68,68,${alpha})`;
      }

      // rounded rect
      const radius = 12;
      ctx.beginPath();
      ctx.moveTo(cx + radius, cy);
      ctx.lineTo(cx + CELL - radius, cy);
      ctx.quadraticCurveTo(cx + CELL, cy, cx + CELL, cy + radius);
      ctx.lineTo(cx + CELL, cy + CELL - radius);
      ctx.quadraticCurveTo(cx + CELL, cy + CELL, cx + CELL - radius, cy + CELL);
      ctx.lineTo(cx + radius, cy + CELL);
      ctx.quadraticCurveTo(cx, cy + CELL, cx, cy + CELL - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.quadraticCurveTo(cx, cy, cx + radius, cy);
      ctx.closePath();
      ctx.fill();

      // border
      ctx.strokeStyle = isDiag
        ? "rgba(99,102,241,0.35)"
        : "rgba(239,68,68,0.25)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // value
      ctx.fillStyle = isDarkMode ? "#ffffff" : "#0f172a";
      ctx.font = "800 20px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(val.toLocaleString(), cx + CELL / 2, cy + CELL / 2);
    });
  }, [matrix, isDarkMode]);

  if (!matrix || !Array.isArray(matrix) || matrix.length < 2) return null;

  return (
    <div className="heatmapWrap">
      <canvas ref={canvasRef} />
    </div>
  );
}

/* Overfitting Card */
function OverfittingCard({ analysis }) {
  if (!analysis) return null;

  return (
    <div className={`metricCard overfittingCard severity-${analysis.severity}`}>
      <span className="metricValue overfittingStatusValue">
        {analysis.status}
      </span>
      <span className="metricLabel">
        Gap: {(analysis.avg_gap).toFixed(4)}
      </span>
    </div>
  );
}

/* Main component */
export default function MetricsModel() {
  const { outcomeId } = useParams();
  const navigate = useNavigate();

  const [outcomes, setOutcomes] = useState([]);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [modelsData, setModelsData] = useState([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);
  const [expandedModels, setExpandedModels] = useState({});
  const [fieldLabelMap, setFieldLabelMap] = useState({});
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved) return saved !== "light";
    return document.body.getAttribute("data-theme") !== "light";
  });

  useEffect(() => {
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.attributeName === "data-theme") {
          setIsDarkMode(document.body.getAttribute("data-theme") !== "light");
        }
      }
    });
    observer.observe(document.body, { attributes: true });
    return () => observer.disconnect();
  }, []);

  const outcome = useMemo(
    () => outcomes.find((item) => item.id === outcomeId),
    [outcomes, outcomeId],
  );

  const outcomeDisplayName = outcome?.name || outcomeId.toUpperCase();

  useEffect(() => {
    let cancelled = false;

    async function loadAllMetrics() {
      setLoading(true);
      setErr("");
      setModelsData([]);

      try {
        const [info, outcomesData, schemaData] = await Promise.all([
          fetchModels(outcomeId),
          fetchOutcomes(),
          fetchSchema(outcomeId).catch(() => null),
        ]);
        if (cancelled) return;

        setOutcomes(outcomesData);
        setModelsInfo(info);

        /* Build field name -> label map from schema */
        const labelMap = {};
        if (schemaData && schemaData.sections) {
          for (const section of schemaData.sections) {
            for (const field of section.fields || []) {
              if (field.name && field.label) {
                labelMap[field.name] = field.label;
              }
            }
          }
        }
        setFieldLabelMap(labelMap);

        const availableModels = (info?.models || []).filter(
          (m) => m.trained !== false,
        );

        if (availableModels.length === 0) {
          setModelsData([]);
          setLoading(false);
          return;
        }

        const metricsResults = await Promise.all(
          availableModels.map(async (model) => {
            try {
              const payload = await fetchMetrics(outcomeId, model.algorithm);
              return {
                algorithm: model.algorithm,
                label: model.label || model.algorithm,
                trained: model.trained,
                isActive: info?.active_model === model.algorithm,
                data: payload,
                error: "",
              };
            } catch (e) {
              return {
                algorithm: model.algorithm,
                label: model.label || model.algorithm,
                trained: model.trained,
                isActive: info?.active_model === model.algorithm,
                data: null,
                error: e.message || "Failed to load metrics",
              };
            }
          }),
        );

        if (cancelled) return;

        const sorted = [...metricsResults].sort((a, b) => {
          if (a.isActive && !b.isActive) return -1;
          if (!a.isActive && b.isActive) return 1;
          return 0;
        });

        setModelsData(sorted);
      } catch (e) {
        if (!cancelled) {
          setErr(e.message || "Failed to load models");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadAllMetrics();

    return () => {
      cancelled = true;
    };
  }, [outcomeId]);

  const fmtKey = (k) =>
    k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  const fmtVal = (v) =>
    typeof v === "number"
      ? v <= 1
        ? (v * 100).toFixed(1) + "%"
        : v.toFixed(4)
      : String(v);

  const pageSubtitle = useMemo(() => {
    const activeLabel =
      modelsData.find((m) => m.isActive)?.label ||
      modelsInfo?.models?.find((m) => m.algorithm === modelsInfo?.active_model)
        ?.label ||
      modelsInfo?.active_model ||
      "None";

    const hasActive = modelsData.some((m) => m.isActive);

    return {
      outcome: outcomeDisplayName,
      activeLabel: hasActive ? activeLabel : null,
    };
  }, [outcomeDisplayName, modelsData, modelsInfo]);

  const toggleExpand = useCallback((algorithm) => {
    setExpandedModels((prev) => ({
      ...prev,
      [algorithm]: !prev[algorithm],
    }));
  }, []);

  /* Build comparison chart data */
  const METRIC_KEYS = ["accuracy", "f1", "precision", "recall", "roc_auc"];
  const METRIC_LABELS = [
    "Accuracy",
    "F1 Score",
    "Precision",
    "Recall",
    "AUC-ROC",
  ];

  const comparisonBarData = useMemo(() => {
    const validModels = modelsData.filter((m) => m.data?.metrics);
    if (validModels.length === 0) return null;

    const SEVERITY_RANK = { good: 2, warning: 1 }; 

    const getRank = (m) => {
      const severity = m.data?.overfitting_analysis?.severity || "unknown";
      const sRank = SEVERITY_RANK[severity] || 0;
      const metrics = m.data.metrics;
      
      return [
        sRank,
        metrics.recall ?? 0,
        metrics.f1 ?? 0,
        metrics.roc_auc ?? 0,
        metrics.precision ?? 0,
        metrics.accuracy ?? 0
      ];
    };

    const top5Models = [...validModels]
      .sort((a, b) => {
        const rankA = getRank(a);
        const rankB = getRank(b);
        for (let i = 0; i < rankA.length; i++) {
          if (rankA[i] !== rankB[i]) {
            return rankB[i] - rankA[i]; 
          }
        }
        return 0;
      })
      .slice(0, 5);

    return {
      labels: top5Models.map((m) => m.label),
      datasets: METRIC_KEYS.map((key, i) => ({
        label: METRIC_LABELS[i],
        data: top5Models.map((m) => m.data.metrics[key] ?? 0),
        backgroundColor: METRIC_COLORS[key].bg,
        borderColor: METRIC_COLORS[key].border,
        borderWidth: 1.5,
        borderRadius: 6,
        borderSkipped: false,
      })),
    };
  }, [modelsData]);

  /* Build Train vs Test comparison chart data */
  const buildTrainTestData = useCallback((overfitObj) => {
    if (!overfitObj) return null;
    const keys = ["accuracy", "f1", "precision", "recall", "roc_auc"];
    const labels = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"];
    return {
      labels,
      datasets: [
        {
          label: "Train",
          data: keys.map((k) => overfitObj.train_metrics[k] ?? 0),
          backgroundColor: "rgba(99,102,241,0.65)",
          borderColor: "#818cf8",
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: "Test",
          data: keys.map((k) => overfitObj.test_metrics[k] ?? 0),
          backgroundColor: "rgba(16,185,129,0.65)",
          borderColor: "#34d399",
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    };
  }, []);

  /* Build per-model feature importance chart */
  const MAX_FI_DISPLAY = 10;

  const buildFIData = useCallback((fi, maxCount = MAX_FI_DISPLAY) => {
    if (!fi || !Array.isArray(fi) || fi.length === 0) return null;
    const sorted = [...fi].sort((a, b) => b.importance - a.importance);
    const top = sorted.slice(0, maxCount);
    return {
      labels: top.map((f) => f.feature),
      datasets: [
        {
          label: "Importance",
          data: top.map((f) => f.importance),
          backgroundColor: FI_GRADIENT_COLORS[0],
          borderColor: FI_GRADIENT_COLORS[1],
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    };
  }, []);

  const tickColor = isDarkMode ? DARK_TICK : LIGHT_TICK;
  const gridColor = isDarkMode ? DARK_GRID : LIGHT_GRID;
  const tooltipBg = isDarkMode
    ? "rgba(15,23,42,0.92)"
    : "rgba(255,255,255,0.96)";
  const tooltipTitle = isDarkMode ? "#f8fafc" : "#0f172a";
  const tooltipBody = isDarkMode ? "#cbd5e1" : "#334155";
  const tooltipBorder = isDarkMode
    ? "rgba(255,255,255,0.15)"
    : "rgba(15,23,42,0.15)";

  /* Build ROC, PR, Threshold, and Doughnut data */
  const buildRocData = useCallback(
    (rocObj) => {
      if (!rocObj) return null;
      return {
        labels: rocObj.fpr,
        datasets: [
          {
            label: "ROC Curve",
            data: rocObj.tpr,
            borderColor: METRIC_COLORS.roc_auc.border,
            backgroundColor: METRIC_COLORS.roc_auc.bg,
            tension: 0.3,
            fill: true,
            pointRadius: 0,
            pointHoverRadius: 4,
          },
          {
            label: "Random",
            data: rocObj.fpr,
            borderColor: isDarkMode
              ? "rgba(255,255,255,0.3)"
              : "rgba(0,0,0,0.3)",
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
          },
        ],
      };
    },
    [isDarkMode],
  );

  const buildPrData = useCallback((prObj) => {
    if (!prObj) return null;
    return {
      labels: prObj.recall,
      datasets: [
        {
          label: "Precision-Recall Curve",
          data: prObj.precision,
          borderColor: METRIC_COLORS.precision.border,
          backgroundColor: METRIC_COLORS.precision.bg,
          tension: 0.3,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    };
  }, []);

  const buildThresholdData = useCallback((thrObj) => {
    if (!thrObj) return null;
    return {
      labels: thrObj.thresholds,
      datasets: [
        {
          label: "Precision",
          data: thrObj.precision,
          borderColor: METRIC_COLORS.precision.border,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          label: "Recall",
          data: thrObj.recall,
          borderColor: METRIC_COLORS.recall.border,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          label: "F1 Score",
          data: thrObj.f1,
          borderColor: METRIC_COLORS.f1.border,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    };
  }, []);

  const buildClassDistData = useCallback(
    (clsObj) => {
      if (!clsObj) return null;
      return {
        labels: clsObj.labels.map(
          (l, i) => `Class ${l} (${clsObj.percentages[i]}%)`,
        ),
        datasets: [
          {
            data: clsObj.counts,
            backgroundColor: [
              isDarkMode ? "rgba(255,255,255,0.15)" : "rgba(15,23,42,0.15)",
              "rgba(239,68,68,0.75)", // Red for positive class
            ],
            borderColor: tooltipBorder,
            borderWidth: 1,
          },
        ],
      };
    },
    [isDarkMode, tooltipBorder],
  );

  const baseBarOpts = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 900,
        easing: "easeOutQuart",
        delay: (ctx) => ctx.dataIndex * 80 + ctx.datasetIndex * 40,
      },
      plugins: {
        legend: {
          labels: { color: tickColor, font: { size: 12, weight: 600 } },
        },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipTitle,
          bodyColor: tooltipBody,
          borderColor: tooltipBorder,
          borderWidth: 1,
          padding: 12,
          cornerRadius: 10,
          callbacks: {
            label: (ctx) =>
              `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: tickColor, font: { size: 12, weight: 500 } },
          grid: { color: gridColor },
        },
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            color: tickColor,
            font: { size: 12, weight: 500 },
            callback: (v) => `${(v * 100).toFixed(0)}%`,
          },
          grid: { color: gridColor },
        },
      },
    }),
    [tickColor, gridColor, tooltipBg, tooltipTitle, tooltipBody, tooltipBorder],
  );

  const fiOpts = useMemo(
    () => ({
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 800,
        easing: "easeOutQuart",
        delay: (ctx) => ctx.dataIndex * 60,
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipTitle,
          bodyColor: tooltipBody,
          borderColor: tooltipBorder,
          borderWidth: 1,
          padding: 12,
          cornerRadius: 10,
          callbacks: {
            title: (items) => {
              if (!items.length) return "";
              const fieldName = items[0].label;
              return fieldLabelMap[fieldName] || fieldName;
            },
            label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 1.05,
          ticks: { color: tickColor, font: { size: 11, weight: 500 } },
          grid: { color: gridColor },
        },
        y: {
          ticks: { color: tickColor, font: { size: 12, weight: 600 } },
          grid: { display: false },
        },
      },
    }),
    [
      fieldLabelMap,
      tickColor,
      gridColor,
      tooltipBg,
      tooltipTitle,
      tooltipBody,
      tooltipBorder,
    ],
  );

  const lineCurveOpts = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animations: {
        x: {
          type: "number",
          easing: "linear",
          duration: (ctx) => {
            if (ctx.type !== "data") return 0;
            const len = ctx.chart.data.labels?.length || 100;
            return 1500 / len;
          },
          from: NaN,
          delay: (ctx) => {
            if (ctx.type !== "data") return 0;
            const len = ctx.chart.data.labels?.length || 100;
            return ctx.index * (1500 / len);
          },
        },
        y: {
          type: "number",
          easing: "linear",
          duration: (ctx) => {
            if (ctx.type !== "data") return 0;
            const len = ctx.chart.data.labels?.length || 100;
            return 1500 / len;
          },
          from: (ctx) => {
            if (ctx.type !== "data") return 0;
            if (ctx.index === 0) return ctx.chart.scales.y.getPixelForValue(0);
            return ctx.chart.getDatasetMeta(ctx.datasetIndex).data[ctx.index - 1].getProps(["y"], true).y;
          },
          delay: (ctx) => {
            if (ctx.type !== "data") return 0;
            const len = ctx.chart.data.labels?.length || 100;
            return ctx.index * (1500 / len);
          },
        },
      },
      plugins: {
        legend: { labels: { color: tickColor, font: { size: 12 } } },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipTitle,
          bodyColor: tooltipBody,
          borderColor: tooltipBorder,
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: 1,
          ticks: { color: tickColor, stepSize: 0.2 },
          grid: { color: gridColor },
          title: { display: true, color: tickColor, font: { weight: "bold" } },
        },
        y: {
          min: 0,
          max: 1,
          ticks: { color: tickColor, stepSize: 0.2 },
          grid: { color: gridColor },
        },
      },
    }),
    [tickColor, gridColor, tooltipBg, tooltipTitle, tooltipBody, tooltipBorder],
  );

  const thresholdOpts = useMemo(() => {
    return {
      ...lineCurveOpts,
      scales: {
        ...lineCurveOpts.scales,
        x: {
          ...lineCurveOpts.scales.x,
          title: {
            display: true,
            text: "Threshold",
            color: tickColor,
            font: { weight: "bold" },
          },
        },
      },
    };
  }, [lineCurveOpts, tickColor]);

  const rocOpts = useMemo(() => {
    return {
      ...lineCurveOpts,
      scales: {
        ...lineCurveOpts.scales,
        x: {
          ...lineCurveOpts.scales.x,
          title: {
            display: true,
            text: "False Positive Rate",
            color: tickColor,
            font: { weight: "bold" },
          },
        },
        y: {
          ...lineCurveOpts.scales.y,
          title: {
            display: true,
            text: "True Positive Rate",
            color: tickColor,
            font: { weight: "bold" },
          },
        },
      },
    };
  }, [lineCurveOpts, tickColor]);

  const prOpts = useMemo(() => {
    return {
      ...lineCurveOpts,
      scales: {
        ...lineCurveOpts.scales,
        x: {
          ...lineCurveOpts.scales.x,
          title: {
            display: true,
            text: "Recall",
            color: tickColor,
            font: { weight: "bold" },
          },
        },
        y: {
          ...lineCurveOpts.scales.y,
          title: {
            display: true,
            text: "Precision",
            color: tickColor,
            font: { weight: "bold" },
          },
        },
      },
    };
  }, [lineCurveOpts, tickColor]);

  const doughnutOpts = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1800,
        easing: "easeInOutCirc",
        animateRotate: true,
        animateScale: true,
        delay: 200,
      },
      plugins: {
        legend: {
          position: "right",
          labels: { color: tickColor, font: { size: 12 } },
        },
        tooltip: {
          backgroundColor: tooltipBg,
          titleColor: tooltipTitle,
          bodyColor: tooltipBody,
          borderColor: tooltipBorder,
          borderWidth: 1,
        },
      },
      cutout: "65%",
    }),
    [tickColor, tooltipBg, tooltipTitle, tooltipBody, tooltipBorder],
  );

  /* render helpers */
  const HIGHLIGHT = ["accuracy", "roc_auc", "f1", "precision", "recall", "auc"];
  const reportRowsOrder = ["0", "1", "macro avg", "weighted avg"];

  /* Early returns */
  if (err) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Model Metrics</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Model Metrics</h2>
            <p className="pageSubtitle">
              Outcome: <b>{outcomeDisplayName}</b>
            </p>
          </div>
          <Button variant="ghost" onClick={() => navigate(-1)}>
            ← Back
          </Button>
        </div>
        <ErrorBanner message={err} />
      </div>
    );
  }

  if (loading) {
    return (
      <div>
        <nav className="breadcrumb">
          <a href="/" className="breadcrumbLink">
            Overview
          </a>
          <span className="breadcrumbSep">›</span>
          <span className="breadcrumbCurrent">Model Metrics</span>
        </nav>
        <div className="pageHeader">
          <div>
            <h2 className="pageTitle">Model Metrics</h2>
            <p className="pageSubtitle">Loading metrics...</p>
          </div>
        </div>
        <LoadingSkeleton />
      </div>
    );
  }

  return (
    <div>
      <nav className="breadcrumb">
        <a href="/" className="breadcrumbLink">
          Overview
        </a>
        <span className="breadcrumbSep">›</span>
        <span className="breadcrumbCurrent">Model Metrics</span>
      </nav>

      <div className="pageHeader">
        <div>
          <h2 className="pageTitle">Model Metrics</h2>
          <p className="pageSubtitle">
            <span className="pageSubtitleOutcome">
              Outcome: <b>{pageSubtitle.outcome}</b>
            </span>
            {pageSubtitle.activeLabel && (
              <>
                <span className="pageSubtitleSep">·</span>
                <span className="pageSubtitleActive">
                  Active Model: <b>{pageSubtitle.activeLabel}</b>
                </span>
              </>
            )}
          </p>
        </div>
        <Button variant="ghost" onClick={() => navigate(-1)}>
          ← Back
        </Button>
      </div>

      {modelsData.length === 0 ? (
        <ErrorBanner message="No trained models found for this outcome." />
      ) : (
        <>
          {/* Comparison Section */}
          {comparisonBarData && (
            <section className="comparisonSection">
              <div className="chartPanel chartPanelFull">
                <h4 className="chartTitle">
                  <span className="chartTitleIcon">||</span>
                  Top 5 Model Performance Comparison
                </h4>
                <div className="chartWrapAuto">
                  <Bar
                    key={isDarkMode ? "dark-comp" : "light-comp"}
                    data={comparisonBarData}
                    options={baseBarOpts}
                  />
                </div>
              </div>
            </section>
          )}

          {/* Per-model Cards */}
          <div className="modelMetricsStack">
            {modelsData.map((modelItem) => {
              const data = modelItem.data || {};
              const metrics = data.metrics || {};
              const report = data.classification_report || null;
              const cm = data.confusion_matrix || null;
              const fi = data.feature_importance || null;
              const totalFeatures = data.total_features || (fi ? fi.length : 0);
              const featuresShown = data.features_shown || (fi ? fi.length : 0);

              const statEntries = Object.entries(metrics).filter(([k]) =>
                HIGHLIGHT.some((h) => k.toLowerCase().includes(h)),
              );

              const fiData = buildFIData(fi);
              const isExpanded = expandedModels[modelItem.algorithm];

              return (
                <Card
                  key={modelItem.algorithm}
                  className={`modelMetricsCard ${modelItem.isActive ? "modelMetricsCardActive" : ""}`}
                >
                  <div className="modelHeader">
                    <div className="modelHeaderLeft">
                      <div className="modelTitleRow">
                        <h3 className="modelTitle">
                          {modelItem.label} Classification
                        </h3>
                        <span
                          className={`modelBadge ${modelItem.isActive ? "modelBadgeActive" : "modelBadgeAvailable"}`}
                        >
                          {modelItem.isActive ? "Active Model" : "Available"}
                        </span>
                      </div>
                      <p className="modelTarget">
                        Target Outcome: <b>{outcomeDisplayName}</b>
                      </p>
                    </div>
                    <div className="modelHeaderRight">
                      <Button
                        size="sm"
                        variant="primary"
                        onClick={() => toggleExpand(modelItem.algorithm)}
                      >
                        {isExpanded ? "▾ Collapse" : "▸ Expand Details"}
                      </Button>
                    </div>
                  </div>

                  {modelItem.error ? (
                    <ErrorBanner message={modelItem.error} />
                  ) : (
                    <>
                      {/* Metric stat cards */}
                      {(statEntries.length > 0 || data.overfitting_analysis) && (
                        <div className="metricsGrid">
                          {statEntries.map(([k, v]) => (
                            <div key={k} className="metricCard">
                              <span className="metricValue">{fmtVal(v)}</span>
                              <span className="metricLabel">{fmtKey(k)}</span>
                            </div>
                          ))}
                          
                          {/* Overfitting Indicator as a Card */}
                          {data.overfitting_analysis && (
                            <OverfittingCard
                              analysis={data.overfitting_analysis}
                            />
                          )}
                        </div>
                      )}

                      {/* Expanded details */}
                      {isExpanded && (
                        <div className="expandedDetails">
                          {/* Feature Importance */}
                          <div className="chartPanel featureImportancePanel">
                            <h4 className="chartTitle">
                              <span className="chartTitleIcon">#</span>
                              Feature Importance — Top{" "}
                              {fiData ? fiData.labels.length : 0} of{" "}
                              {totalFeatures} Features
                            </h4>
                            {fiData ? (
                              <div className="chartWrap chartWrapFI">
                                <Bar
                                  key={isDarkMode ? "dark-fi" : "light-fi"}
                                  data={fiData}
                                  options={fiOpts}
                                />
                              </div>
                            ) : (
                              <p className="noDataMsg">
                                No feature importance data available. 
                              </p>
                            )}
                          </div>

                          {/* Train vs Test Comparison (Overfitting) */}
                          {data.overfitting_analysis && (
                            <div className="chartPanel featureImportancePanel">
                              <h4 className="chartTitle">
                                <span className="chartTitleIcon">⚖️</span>
                                Train vs Test — Generalization Analysis
                              </h4>
                              <div className="chartWrapAuto">
                                <Bar
                                  key={
                                    isDarkMode
                                      ? "dark-traintest"
                                      : "light-traintest"
                                  }
                                  data={buildTrainTestData(
                                    data.overfitting_analysis,
                                  )}
                                  options={baseBarOpts}
                                />
                              </div>
                            </div>
                          )}

                          {/* New Grid for line curves + doughnut */}
                          <div className="metricsChartsGrid">
                            <div className="chartPanel">
                              <h4 className="chartTitle">
                                <span className="chartTitleIcon">📈</span>
                                ROC Curve
                                {data.roc_curve && (
                                  <span className="metricLabel">
                                    {" "}
                                    (AUC: {data.roc_curve.auc.toFixed(3)})
                                  </span>
                                )}
                              </h4>
                              {data.roc_curve ? (
                                <div className="chartWrap chartWrapSquare">
                                  <Line
                                    key={isDarkMode ? "dark-roc" : "light-roc"}
                                    data={buildRocData(data.roc_curve)}
                                    options={rocOpts}
                                  />
                                </div>
                              ) : (
                                <p className="noDataMsg">
                                  Re-train model to generate ROC curve.
                                </p>
                              )}
                            </div>

                            <div className="chartPanel">
                              <h4 className="chartTitle">
                                <span className="chartTitleIcon">🎯</span>
                                Precision-Recall Curve
                              </h4>
                              {data.pr_curve ? (
                                <div className="chartWrap chartWrapSquare">
                                  <Line
                                    key={isDarkMode ? "dark-pr" : "light-pr"}
                                    data={buildPrData(data.pr_curve)}
                                    options={prOpts}
                                  />
                                </div>
                              ) : (
                                <p className="noDataMsg">
                                  Re-train model to generate PR curve.
                                </p>
                              )}
                            </div>

                            <div className="chartPanel">
                              <h4 className="chartTitle">
                                <span className="chartTitleIcon">⚙️</span>
                                Metrics over Thresholds
                              </h4>
                              {data.threshold_metrics ? (
                                <div className="chartWrap chartWrapSquare">
                                  <Line
                                    key={isDarkMode ? "dark-thr" : "light-thr"}
                                    data={buildThresholdData(
                                      data.threshold_metrics,
                                    )}
                                    options={thresholdOpts}
                                  />
                                </div>
                              ) : (
                                <p className="noDataMsg">
                                  Re-train model to generate Threshold curve.
                                </p>
                              )}
                            </div>

                            <div className="chartPanel">
                              <h4 className="chartTitle">
                                <span className="chartTitleIcon">📊</span>
                                Test Set Class Distribution
                              </h4>
                              {data.class_distribution ? (
                                <div className="chartWrap chartWrapSquare">
                                  <Doughnut
                                    key={
                                      isDarkMode ? "dark-dist" : "light-dist"
                                    }
                                    data={buildClassDistData(
                                      data.class_distribution,
                                    )}
                                    options={doughnutOpts}
                                  />
                                </div>
                              ) : (
                                <p className="noDataMsg">
                                  Re-train model to generate Class Distribution.
                                </p>
                              )}
                            </div>
                          </div>

                          {/* Bottom grid: report + confusion */}
                          <div className="metricsBottomGrid">
                            {report && (
                              <div className="metricsPanel metricsPanelHover">
                                <h4 className="sectionTitle">
                                  Classification Report
                                </h4>
                                <table className="table">
                                  <thead>
                                    <tr>
                                      <th align="left">Class</th>
                                      <th align="left">Precision</th>
                                      <th align="left">Recall</th>
                                      <th align="left">F1</th>
                                      <th align="left">Support</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {reportRowsOrder.map((label) => {
                                      const row = report[label];
                                      if (!row) return null;
                                      return (
                                        <tr key={label}>
                                          <td>{label}</td>
                                          <td>
                                            {(row["precision"] ?? 0).toFixed(4)}
                                          </td>
                                          <td>
                                            {(row["recall"] ?? 0).toFixed(4)}
                                          </td>
                                          <td>
                                            {(row["f1-score"] ?? 0).toFixed(4)}
                                          </td>
                                          <td>{row["support"] ?? "-"}</td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </div>
                            )}

                            {cm && Array.isArray(cm) && cm.length >= 2 && (
                              <div className="metricsPanel metricsPanelHover">
                                <h4 className="sectionTitle">
                                  Confusion Matrix
                                </h4>
                                <ConfusionHeatmap
                                  matrix={cm}
                                  isDarkMode={isDarkMode}
                                />
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
