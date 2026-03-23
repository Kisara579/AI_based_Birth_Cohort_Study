import FieldRenderer from "./FieldRenderer.jsx";
import "./FormSection.css";

export default function FormSection({
  section,
  values,
  errors,
  onChange,
  onBlur,
}) {
  return (
    <div className="formSection">
      {section.title && (
        <h3 className="formSectionTitle">{section.title}</h3>
      )}

      <div className="formGrid">
        {section.fields.map((field) => (
          <FieldRenderer
            key={field.name}
            field={field}
            value={values[field.name]}
            error={errors[field.name]}
            onChange={onChange}
            onBlur={onBlur}
          />
        ))}
      </div>
    </div>
  );
}
