export default function ErrorBanner({ title = "Error", message, onRetry }) {
  return (
    <div className="errorBanner" role="alert">
      <div className="errorText">
        <strong className="errorTitle">{title}</strong>
        {message && <div className="errorMessage">{message}</div>}
      </div>

      {onRetry && (
        <button className="errorRetryBtn" onClick={onRetry} type="button">
          Retry
        </button>
      )}
    </div>
  );
}

