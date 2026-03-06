import "./LoadingSkeleton.css";

export default function LoadingSkeleton({ lines = 3, className = "" }) {
  return (
    <div className={`skeletonWrap ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="skeletonLine" />
      ))}
    </div>
  );
}
