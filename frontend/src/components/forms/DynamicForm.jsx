import { useEffect, useMemo, useState } from "react";
import FormSection from "./FormSection.jsx";
import Button from "../common/Button.jsx";
import "./DynamicForm.css";

// simple validators
function validateField(field, rawValue) {
  const v = rawValue;

  if (field.required) {
    const empty = v === undefined || v === null || String(v).trim() === "";
    if (empty) return "Required";
  }

  // if empty and not required 
  if (v === undefined || v === null || String(v).trim() === "") return null;

  if (field.type === "number") {
    const num = Number(v);
    if (Number.isNaN(num)) return "Must be a number";
    if (field.min !== undefined && num < field.min) return `Min ${field.min}`;
    if (field.max !== undefined && num > field.max) return `Max ${field.max}`;
  }

  if (field.type === "select") {
    // allow any option value as string, but require it to be present
    if (String(v).trim() === "") return "Required";
  }

  return null;
}

function normalizeValues(schema, values) {
  // Convert number fields to Number; keep select as number if option values are numeric-like
  const out = {};
  for (const section of schema.sections) {
    for (const field of section.fields) {
      const raw = values[field.name];

      if (raw === "" || raw === undefined || raw === null) {
        out[field.name] = null;
        continue;
      }

      if (field.type === "number") {
        out[field.name] = Number(raw);
      } else if (field.type === "select") {
        // try to cast to number if it looks numeric, else keep string
        const asNum = Number(raw);
        out[field.name] = Number.isNaN(asNum) ? raw : asNum;
      } else {
        out[field.name] = raw;
      }
    }
  }
  return out;
}

export default function DynamicForm({ schema, onSubmit, initialValues }) {
  const allFields = useMemo(() => {
    const fields = [];
    for (const s of schema.sections) fields.push(...s.fields);
    return fields;
  }, [schema]);

  const [values, setValues] = useState(() => {
    const init = {};
    for (const f of allFields) {
      init[f.name] =
        initialValues &&
        initialValues[f.name] !== undefined &&
        initialValues[f.name] !== null
          ? String(initialValues[f.name])
          : "";
    }
    return init;
  });

  useEffect(() => {
    if (initialValues) {
      setValues((prev) => {
        const next = { ...prev };
        for (const f of allFields) {
          if (
            initialValues[f.name] !== undefined &&
            initialValues[f.name] !== null
          ) {
            next[f.name] = String(initialValues[f.name]);
          }
        }
        return next;
      });
    }
  }, [initialValues, allFields]);

  const [errors, setErrors] = useState({});

  function handleChange(name, value) {
    setValues((prev) => ({ ...prev, [name]: value }));
    // live clear error on change
    setErrors((prev) => ({ ...prev, [name]: null }));
  }

  function handleBlur(name) {
    const field = allFields.find((f) => f.name === name);
    if (!field) return;
    const err = validateField(field, values[name]);
    setErrors((prev) => ({ ...prev, [name]: err }));
  }

  function validateAll() {
    const nextErrors = {};
    for (const field of allFields) {
      const err = validateField(field, values[field.name]);
      if (err) nextErrors[field.name] = err;
    }
    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  }

  function handleSubmit(e) {
    e.preventDefault();
    const ok = validateAll();
    if (!ok) return;

    const normalized = normalizeValues(schema, values);
    onSubmit?.(normalized);
  }

  return (
    <form onSubmit={handleSubmit} className="dynamicForm">
      {schema.sections.map((section) => (
        <FormSection
          key={section.id}
          section={section}
          values={values}
          errors={errors}
          onChange={handleChange}
          onBlur={handleBlur}
        />
      ))}

      <div className="formActions">
        <Button type="submit" variant="primary">
          Run Prediction
        </Button>
      </div>
    </form>
  );
}
