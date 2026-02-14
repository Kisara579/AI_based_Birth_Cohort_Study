// Schema format: one object per outcomeId
// Each field describes how to render + validate an input

export const schemasByOutcome = {
  lbw: {
    title: "Low Birth Weight Inputs",
    sections: [
      {
        id: "maternal",
        title: "Maternal Information",
        fields: [
          {
            name: "mage",
            label: "Mother age (years)",
            type: "number",
            min: 10,
            max: 60,
            required: true,
            placeholder: "e.g., 28",
          },
          {
            name: "meduc",
            label: "Mother education (years)",
            type: "number",
            min: 0,
            max: 25,
            required: true,
            placeholder: "e.g., 12",
          },
          {
            name: "cigs",
            label: "Cigarettes per day",
            type: "number",
            min: 0,
            max: 60,
            required: true,
            placeholder: "e.g., 0",
          },
          {
            name: "drink",
            label: "Alcohol use",
            type: "select",
            required: true,
            options: [
              { value: 0, label: "No" },
              { value: 1, label: "Yes" },
            ],
          },
        ],
      },
      {
        id: "pregnancy",
        title: "Pregnancy Care",
        fields: [
          {
            name: "monpre",
            label: "Month prenatal care started",
            type: "number",
            min: 1,
            max: 9,
            required: true,
            placeholder: "e.g., 2",
          },
          {
            name: "npvis",
            label: "Number of prenatal visits",
            type: "number",
            min: 0,
            max: 40,
            required: true,
            placeholder: "e.g., 8",
          },
          {
            name: "male",
            label: "Baby sex",
            type: "select",
            required: true,
            options: [
              { value: 1, label: "Male" },
              { value: 0, label: "Female" },
            ],
          },
        ],
      },
    ],
  },
};
