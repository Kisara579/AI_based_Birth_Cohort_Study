import { Link } from "react-router-dom";
import "./NotFound.css";

export default function NotFound() {
  return (
    <div className="notFoundWrap">
      <h1 className="notFoundTitle">404</h1>
      <p className="notFoundText">Page not found.</p>
      <Link className="notFoundLink" to="/">
        Go Home
      </Link>
    </div>
  );
}
