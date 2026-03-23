import "./Footer.css";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footerInner">
        <span>© {new Date().getFullYear()} AI Birth Cohort Study</span>
        <span className="muted">Research dashboard</span>
      </div>
    </footer>
  );
}
