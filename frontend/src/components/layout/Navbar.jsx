import { NavLink } from "react-router-dom";

const linkClass = ({ isActive }) =>
  "navLink" + (isActive ? " navLinkActive" : "");

export default function Navbar() {
  return (
    <header className="navBar">
      <div className="container navInner">
        <NavLink to="/" className="brand">
          <span className="brandText">AI Birth Cohort Study</span>
        </NavLink>

        <nav className="navLinks">
          <NavLink to="/" className={linkClass} end>
            Overview
          </NavLink>
          <NavLink to="/about" className={linkClass}>
            About
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
