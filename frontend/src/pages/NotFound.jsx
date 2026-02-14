// NotFound.jsx
import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div>
      <h1>404</h1>
      <p className="muted">Page not found.</p>
      <Link className="navLink navLinkActive" to="/">
        Go Home
      </Link>
    </div>
  );
}
