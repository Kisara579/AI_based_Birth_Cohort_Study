import { createBrowserRouter } from "react-router-dom";
import MainLayout from "../components/layout/MainLayout.jsx";

import Overview from "../pages/Overview.jsx";
import PredictModel from "../pages/PredictModel.jsx";
import ResultsModel from "../pages/ResultsModel.jsx";
import About from "../pages/About.jsx";
import NotFound from "../pages/NotFound.jsx";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <MainLayout />,
    children: [
      { index: true, element: <Overview /> },

      { path: "predict/:outcomeId", element: <PredictModel /> },
      { path: "results/:outcomeId", element: <ResultsModel /> },

      { path: "about", element: <About /> },

      { path: "*", element: <NotFound /> },
    ],
  },
]);
