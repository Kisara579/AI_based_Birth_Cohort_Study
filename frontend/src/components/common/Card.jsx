
export default function Card({ title, subtitle, children, classname = "", className = "" }) {
  return (
    <section className={`card ${classname} ${className}`.trim()}>
      {(title || subtitle) && (
        <header className="cardHeader">
          {title && <h2 className="cardTitle">{title}</h2>}
          {subtitle && <p className="cardSubtitle">{subtitle}</p>}
        </header>
      )}
      <div className="cardBody">{children}</div>
    </section>
  );
}
