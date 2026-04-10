import { createBrowserRouter } from "react-router-dom";
import MainLayout from "../components/layout/MainLayout.jsx";

import Overview from "../pages/Overview.jsx";
import About from "../pages/About.jsx";
import NotFound from "../pages/NotFound.jsx";
import MetricsModel from "../pages/MetricsModel.jsx";
import PatientAssessmentForm from "../pages/PatientAssessmentForm.jsx";
import RiskDashboard from "../pages/RiskDashboard.jsx";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <MainLayout />,
    children: [
      { index: true, element: <Overview /> },

      { path: "predict", element: <PatientAssessmentForm /> },
      { path: "results", element: <RiskDashboard /> },

      { path: "metrics/:outcomeId", element: <MetricsModel /> },

      { path: "about", element: <About /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
