export default function FieldRenderer({
  field,
  value,
  error,
  onChange,
  onBlur,
}) {
  const id = `field_${field.name}`;

  return (
    <div className="formField">
      <label className="formLabel" htmlFor={id}>
        {field.label}
        {field.required ? <span className="req">*</span> : null}
      </label>

      {field.type === "number" ? (
        <input
          id={id}
          className={`formInput ${error ? "formInputError" : ""}`}
          type="number"
          inputMode="numeric"
          value={value ?? ""}
          placeholder={field.placeholder || ""}
          min={field.min}
          max={field.max}
          step={field.step ?? "any"}
          onChange={(e) => onChange(field.name, e.target.value)}
          onBlur={() => onBlur?.(field.name)}
        />
      ) : field.type === "select" ? (
        <select
          id={id}
          className={`formInput ${error ? "formInputError" : ""}`}
          value={value ?? ""}
          onChange={(e) => onChange(field.name, e.target.value)}
          onBlur={() => onBlur?.(field.name)}
        >
          <option value="" disabled>
            Select...
          </option>
          {(field.options || []).map((opt) => (
            <option key={String(opt.value)} value={String(opt.value)}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : (
        <div className="muted">Unsupported field type: {field.type}</div>
      )}

      {error ? <div className="formError">{error}</div> : null}
    </div>
  );
}
