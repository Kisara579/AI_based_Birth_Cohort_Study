import { NavLink } from "react-router-dom";
import ApiStatusDot from "./ApiStatusDot.jsx";
import { useEffect, useState } from "react";
import "./Navbar.css";

const linkClass = ({ isActive }) =>
  "navLink" + (isActive ? " navLinkActive" : "");

export default function Navbar() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || "dark";
  });
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
  }, [theme]);

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.body.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  }

  function closeMenu() {
    setMenuOpen(false);
  }

  return (
    <header className="navBar">
      <div className="container navInner">
        <NavLink to="/" className="brand">
          {/*icon */}
          <svg
            width="26"
            height="26"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--accent)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className="brandIcon"
          >
            <path d="M12 2a7 7 0 0 1 7 7c0 4-3 7-7 7s-7-3-7-7a7 7 0 0 1 7-7z" />
            <path d="M9 17v1a3 3 0 0 0 6 0v-1" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="10" y1="11" x2="14" y2="11" />
          </svg>
          <span className="brandText">AI Birth Cohort Study</span>
          <ApiStatusDot />
        </NavLink>

        <div className="navActions">
          <button
            type="button"
            className="themeToggle"
            onClick={toggleTheme}
            aria-label="Toggle theme"
            title={
              theme === "dark"
                ? "Switch to light mode"
                : "Switch to dark mode"
            }
          >
            {theme === "dark" ? (
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
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
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
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>

          <button
            type="button"
            className="menuToggle"
            onClick={() => setMenuOpen((value) => !value)}
            aria-label="Toggle navigation menu"
            aria-expanded={menuOpen}
            aria-controls="site-navigation"
          >
            <span />
            <span />
            <span />
          </button>
        </div>

        <nav
          id="site-navigation"
          className={`navLinks${menuOpen ? " navLinksOpen" : ""}`}
        >
          <NavLink to="/" className={linkClass} onClick={closeMenu} end>
            Overview
          </NavLink>
          <NavLink to="/about" className={linkClass} onClick={closeMenu}>
            About
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
