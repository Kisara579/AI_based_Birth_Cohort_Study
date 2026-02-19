# ✅ Teammate Guide: Clone → Train → Register Model → Test → Push PR

---

## 📁 0. Project Structure (Understand First)

After cloning, the project looks like this:

```
AI_based_Birth_Cohort_Study/
│
├── backend/
│   ├── app/
│   │   ├── catalog.py      ← You will add your outcome schema here
│   │   ├── main.py         ← DO NOT EDIT
│   │   ├── predictor.py    ← DO NOT EDIT
│   │   └── config.py       ← DO NOT EDIT
│   ├── models/             ← You will place your .pkl + .meta.json here
│   └── requirements.txt
│
├── frontend/               ← DO NOT EDIT
│
├── training/               ← Create a subfolder here for your training script
│   └── low_birth_weight/   ← Example already exists
│
└── instruction_guide.md
```

> **Already working example:** The `lbw` (Low Birth Weight) outcome has a fully working model in `backend/models/lbw_model.pkl`. Use it to verify your setup works before adding your own.

---

## 🧠 1. Your Responsibility

You are assigned one outcome (e.g. `preterm`, `gdm`, `nicu`). Your job end-to-end:

1. Create a training script under `training/<your_outcome>/`
2. Train your ML model as a full sklearn Pipeline
3. Save the model as `<model_key>_model.pkl`
4. Create a meta file `<model_key>_model.meta.json`
5. Place both files into `backend/models/`
6. Register your outcome + schema in `backend/app/catalog.py`
7. Restart backend and test via Swagger + UI
8. Open a Pull Request

**You do NOT modify:**

- `frontend/`
- `backend/app/predictor.py`
- `backend/app/config.py`
- `backend/app/main.py`

---

## 0.1 Prerequisites — Install These First

Before cloning, make sure you have all of the following installed:

### Python 3.9 or higher

Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)

**Verify:**

```bash
python --version
# Expected: Python 3.9.x or higher
# On Mac/Linux try: python3 --version
```

> ⚠️ During installation on Windows, check **"Add Python to PATH"**.

### Node.js 18 or higher (for frontend)

Download from [https://nodejs.org/](https://nodejs.org/) — choose the **LTS** version.

**Verify:**

```bash
node --version   # Expected: v18.x or higher
npm --version    # Expected: 9.x or higher
```

### Git

Download from [https://git-scm.com/downloads](https://git-scm.com/downloads)

**Verify:**

```bash
git --version
```

---

## 1 Clone the repo

```bash
git clone <REPO_URL>
cd AI_based_Birth_Cohort_Study
```

---

## 2 Create Python Virtual Environment (.venv)

A virtual environment keeps project dependencies isolated from the rest of your machine. You only need to do this **once** after cloning.

> All commands below must be run from inside the **project root** folder (`AI_based_Birth_Cohort_Study/`).

### Windows PowerShell

```powershell
python -m venv .venv
```

### Windows Git Bash / Command Prompt

```bash
python -m venv .venv
```

### Mac / Linux

```bash
python3 -m venv .venv
```

This creates a `.venv/` folder in the project root. It is already in `.gitignore` so it will never be committed.

**Verify it was created:**

```bash
# You should see .venv/ listed
ls      # Mac/Linux/Git Bash
dir     # Windows Command Prompt
```

---

## 3 Activate the Virtual Environment

You must activate it **every time you open a new terminal**.

### Windows PowerShell

> ⚠️ If you get a "scripts cannot be run" error, run this **once** in PowerShell as Administrator, then close and reopen PowerShell:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Git Bash

```bash
source .venv/Scripts/activate
```

### Mac / Linux

```bash
source .venv/bin/activate
```

After activation your terminal prompt will show `(.venv)` at the start — this confirms it is active.

---

## 4 Run Backend (FastAPI)

> Make sure `.venv` is activated (you see `(.venv)` in the prompt) before running these commands.
> Run from the **project root** (`AI_based_Birth_Cohort_Study/`).

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --port 8000
```

### Verify backend is running

Open these URLs in your browser:

- `http://127.0.0.1:8000/health` → should return `{"ok": true, ...}`
- `http://127.0.0.1:8000/outcomes` → should return a list of all outcomes
- `http://127.0.0.1:8000/docs` → interactive Swagger UI

---

## 5 Run Frontend (React/Vite)

Open a **new terminal** (keep backend running in the first terminal):

```bash
cd frontend
npm install
npm run dev
```

Open:

- `http://localhost:5173`

> ⚠️ Make sure backend is running on port 8000 first, otherwise predictions in the UI will fail.

---

---

# ✅ Part B — Add Your Model

---

## 7. Naming Rules (CRITICAL)

The backend loads models by `model_key`. Your file names **must** follow this exact pattern:

```
backend/models/<model_key>_model.pkl
backend/models/<model_key>_model.meta.json
```

⚠️ If the name is wrong → backend throws `501` or `500` errors.

### Current `catalog.py` outcome ↔ model_key mapping

| outcome_id     | model_key      | Required model file       | Required meta file              |
| -------------- | -------------- | ------------------------- | ------------------------------- |
| `pph`          | `pph`          | `pph_model.pkl`           | `pph_model.meta.json`           |
| `preeclampsia` | `preeclampsia` | `preeclampsia_model.pkl`  | `preeclampsia_model.meta.json`  |
| `iugr`         | `iugr`         | `iugr_model.pkl`          | `iugr_model.meta.json`          |
| `gdm`          | `gdm`          | `gdm_model.pkl`           | `gdm_model.meta.json`           |
| `iol`          | `iol`          | `iol_model.pkl`           | `iol_model.meta.json`           |
| `c_section`    | `c_section`    | `c_section_model.pkl`     | `c_section_model.meta.json`     |
| `postpartum`   | `postpartum`   | `postpartum_model.pkl`    | `postpartum_model.meta.json`    |
| `lbw`          | `lbw`          | `lbw_model.pkl` ✅ exists | `lbw_model.meta.json` ✅ exists |
| `preterm`      | `preterm`      | `preterm_model.pkl`       | `preterm_model.meta.json`       |
| `nicu`         | `nicu`         | `nicu_model.pkl`          | `nicu_model.meta.json`          |
| `feeding`      | `feeding`      | `feeding_model.pkl`       | `feeding_model.meta.json`       |

> If you register a brand-new outcome in `catalog.py`, the `model_key` you set there must be used in both file names.

---

## 8. Create Your Training Script

Create a folder under `training/` for your outcome and write your training script there:

```
training/
└── preterm/
    └── training_preterm.py
```

### Recommended pattern — always use a full sklearn Pipeline

```python
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1. Load and prepare data
df = pd.read_csv("path/to/your/data.csv")
X = df[["age", "bmi", "bp", "glucose"]]   # ← your feature columns
y = df["label"]                             # ← your target column

# 2. Build a full pipeline (preprocessing + model in one object)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss"))
])

# 3. Train
pipeline.fit(X, y)

# 4. (Optional) Quick sanity check
test_input = pd.DataFrame([{"age": 28, "bmi": 23, "bp": 120, "glucose": 90}])
print(pipeline.predict_proba(test_input))

# 5. Save — use exact model_key name
joblib.dump(pipeline, "preterm_model.pkl")
print("Saved: preterm_model.pkl")
```

> ⚠️ **Always save the full Pipeline**, not just the XGBoost model object. The backend passes a DataFrame directly to `predict_proba()`. If preprocessing is not inside the pipeline, prediction will fail.

**Run the script from the project root (with `.venv` active):**

```bash
# Windows PowerShell / Git Bash
python training/preterm/training_preterm.py

# Mac/Linux
python3 training/preterm/training_preterm.py
```

---

## 9. Test Model Before Moving to Backend

Before copying anything to `backend/models/`, verify in your training script:

```python
test_input = pd.DataFrame([{"age": 28, "bmi": 23, "bp": 120, "glucose": 90}])
proba = pipeline.predict_proba(test_input)
print(proba)  # Should print a 2-column array like [[0.72, 0.28]]
```

If `predict_proba` fails here → fix your pipeline first. Do not move a broken model to `backend/models/`.

---

## 10. Feature Alignment Rule (MOST COMMON MISTAKE)

These three things **must match exactly** — same names, same order:

| Location                     | What must match                     |
| ---------------------------- | ----------------------------------- |
| Training `X` column names    | `feature_order` in meta.json        |
| `feature_order` in meta.json | `"name"` field of each schema field |
| Schema field `"name"` values | Keys sent in the API request body   |

If any one of these three differs → backend returns **400 Missing required features**.

**Example of a correct alignment:**

Training columns:

```python
X = df[["age", "bmi", "bp", "glucose"]]
```

Meta `feature_order`:

```json
["age", "bmi", "bp", "glucose"]
```

Schema `"name"` fields in `catalog.py`:

```python
{"name": "age", ...},
{"name": "bmi", ...},
{"name": "bp", ...},
{"name": "glucose", ...},
```

---

## 11. Create Meta File (REQUIRED)

Create a `.meta.json` file with the same base name as your model:

```
backend/models/preterm_model.meta.json
```

```json
{
  "feature_order": ["age", "bmi", "bp", "glucose"],
  "threshold": 0.5,
  "model_name": "Preterm XGBoost v1",
  "trained_by": "Your Name",
  "date": "2026-02-19"
}
```

**Rules:**

- `feature_order` must match your training feature names **exactly** (same spelling, same order)
- `threshold` controls the High/Low risk label — `0.5` is standard
- Backend returns **400** if any feature from `feature_order` is missing in the request
- Backend returns **500** if the meta file is missing

---

## 12. Place Files into `backend/models/`

Copy both files into the models folder:

```
backend/models/preterm_model.pkl
backend/models/preterm_model.meta.json
```

Your `backend/models/` folder should look like:

```
backend/models/
├── lbw_model.pkl             ← already exists
├── lbw_model.meta.json       ← already exists
├── preterm_model.pkl         ← you added this
└── preterm_model.meta.json   ← you added this
```

---

## 13. Register Outcome + Schema in `catalog.py`

Edit `backend/app/catalog.py`. Most outcomes are already registered with an empty schema. Find your outcome and fill in the `"sections"` inside `"schema"`:

```python
"preterm": {
    "id": "preterm",
    "name": "Preterm Birth",
    "short": "PTB",
    "description": "Predict likelihood of delivery before 37 weeks.",
    "model_key": "preterm",
    "schema": {
        "title": "Preterm Birth Inputs",
        "sections": [
            {
                "id": "maternal",
                "title": "Maternal Information",
                "fields": [
                    {
                        "name": "age",         # ← must match feature_order
                        "label": "Age (years)",
                        "type": "number",
                        "min": 10,
                        "max": 60,
                        "required": True
                    },
                    {
                        "name": "bmi",         # ← must match feature_order
                        "label": "BMI",
                        "type": "number",
                        "min": 10,
                        "max": 60,
                        "required": True
                    },
                    {
                        "name": "bp",          # ← must match feature_order
                        "label": "Blood pressure",
                        "type": "number",
                        "min": 60,
                        "max": 200,
                        "required": True
                    },
                    {
                        "name": "glucose",     # ← must match feature_order
                        "label": "Glucose (mg/dL)",
                        "type": "number",
                        "min": 50,
                        "max": 300,
                        "required": True
                    }
                ]
            }
        ]
    }
}
```

> **Field types supported:** `"number"`, `"select"` (add `"options"` array for select fields).

---

## 14. (Optional but Recommended) Save Model Metrics

You may also save a metrics file alongside your model:

```
backend/models/preterm_metrics.json
```

```json
{
  "accuracy": 0.89,
  "precision": 0.72,
  "recall": 0.65,
  "f1_score": 0.68,
  "roc_auc": 0.91
}
```

This will later be exposed via `GET /model-metrics/preterm` to display charts in the UI.

---

## 15. Restart Backend

Always restart after adding model files or editing `catalog.py`:

```bash
# Press CTRL+C to stop the running server, then:
cd backend
uvicorn app.main:app --reload --port 8000
```

> If you are in the project root (not inside `backend/`), run `cd backend` first.

---

## 16. Test Your Outcome

### 16.1 Check outcome appears

Open `http://127.0.0.1:8000/outcomes` — your outcome must be in the list.

### 16.2 Check schema

Open `http://127.0.0.1:8000/schema/preterm` — must return your schema JSON.

### 16.3 Test predict in Swagger

Open `http://127.0.0.1:8000/docs`, use `POST /predict/{outcome_id}` with `outcome_id = preterm`.

Body:

```json
{
  "data": {
    "age": 25,
    "bmi": 22,
    "bp": 120,
    "glucose": 90
  }
}
```

Expected:

```json
{
  "probability": 0.xx,
  "risk_label": "Low Risk" or "High Risk"
}
```

### Common Errors

- **501 Model not found** → wrong file name, see naming table in section 7
- **500 Missing meta file** → meta file missing or wrong name
- **400 Missing required features** → schema `"name"` values don't match `feature_order`

### 16.4 Test in Web UI

Open `http://localhost:5173` → Overview → click your outcome card → Predict → fill form → Submit → Results page displays the prediction.

---

## 17. Git Workflow

> Run from the **project root** (`AI_based_Birth_Cohort_Study/`). If you are inside `backend/`, run `cd ..` first.

```bash
git checkout -b add-preterm-model
git add backend/models/ backend/app/catalog.py training/preterm/
git commit -m "Add preterm model, meta, schema, training script"
git push origin add-preterm-model
```

Open a Pull Request on GitHub.

---

## ✅ What NOT to Edit

| File / Folder              | Rule                                               |
| -------------------------- | -------------------------------------------------- |
| `frontend/`                | ❌ Do NOT touch                                    |
| `backend/app/main.py`      | ❌ Do NOT touch                                    |
| `backend/app/predictor.py` | ❌ Do NOT touch                                    |
| `backend/app/config.py`    | ❌ Do NOT touch                                    |
| `backend/models/`          | ✅ Add your `.pkl` and `.meta.json` only           |
| `backend/app/catalog.py`   | ✅ Fill in your outcome's `"schema"` sections only |
| `training/<your_outcome>/` | ✅ Create and add your training script             |

---

## ✅ Quick Checklist Before PR

### Environment setup works

- [ ] `python --version` shows 3.9+
- [ ] `node --version` shows 18+
- [ ] `.venv` created and activated (`(.venv)` in prompt)
- [ ] `pip install -r backend/requirements.txt` completed without errors
- [ ] `http://127.0.0.1:8000/health` returns `{"ok": true}`
- [ ] `http://localhost:5173` loads the UI

### Training done

- [ ] Training script created in `training/<your_outcome>/`
- [ ] Full sklearn Pipeline saved (not raw model object)
- [ ] `predict_proba(test_input)` verified inside training script before moving files

### Files in place

- [ ] `backend/models/<model_key>_model.pkl` exists
- [ ] `backend/models/<model_key>_model.meta.json` exists
- [ ] `feature_order` in meta exactly matches training `X` column names
- [ ] Schema `"name"` fields exactly match `feature_order` names

### Backend working

- [ ] Backend restarted after adding files
- [ ] `http://127.0.0.1:8000/outcomes` lists your outcome
- [ ] `http://127.0.0.1:8000/schema/<outcome_id>` returns your schema
- [ ] Swagger `POST /predict/<outcome_id>` returns `probability` + `risk_label`
- [ ] UI shows prediction on Results page

---

## ⚠️ Common Mistakes

| Error                                | Cause                             | Fix                                                                   |
| ------------------------------------ | --------------------------------- | --------------------------------------------------------------------- |
| `501 Model not found`                | Wrong file name or wrong folder   | File must be `backend/models/<model_key>_model.pkl`                   |
| `500 Missing meta file`              | Meta file missing or wrong name   | Create `backend/models/<model_key>_model.meta.json`                   |
| `400 Missing required features`      | Schema `"name"` ≠ `feature_order` | Align all three: training columns = feature_order = schema names      |
| `500 predict_proba failed`           | Saved raw model, not Pipeline     | Always `joblib.dump(pipeline, ...)` the full Pipeline                 |
| `scripts cannot be run` (PowerShell) | Execution policy blocked          | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as admin    |
| `python` not found                   | Python not on PATH                | Reinstall Python and check "Add to PATH"                              |
| `pip install` fails with SSL error   | Corporate proxy / antivirus       | Try `pip install --trusted-host pypi.org -r backend/requirements.txt` |
| `npm: command not found`             | Node.js not installed             | Install from [https://nodejs.org](https://nodejs.org)                 |
| Frontend shows "API offline"         | Backend not running               | Start backend first on port 8000                                      |
