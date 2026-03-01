# AI-Based Maternal & Neonatal Risk Prediction System

A comprehensive machine learning system for predicting maternal and neonatal health outcomes using clinical and demographic data. This system employs 11 independent predictive models to assess risk factors across pregnancy, delivery, and postpartum stages.

## Table of Contents

- [Overview](#overview)
- [Predicted Outcomes](#predicted-outcomes)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Models](#training-models)
- [Adding New Models](#adding-new-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

## Overview

This system provides evidence-based risk prediction for maternal and neonatal outcomes using supervised machine learning algorithms. The models are trained on clinical birth cohort data and utilize features including maternal demographics, medical history, laboratory values, and prenatal care indicators.

The system is designed for research purposes and to support clinical decision-making by identifying high-risk cases that may require enhanced monitoring or intervention. Each model outputs a risk probability and classification, enabling stratified care pathways.

## Predicted Outcomes

The system predicts the following 11 maternal and neonatal health outcomes:

### Maternal Health Outcomes

1. **Gestational Diabetes** - Risk of developing diabetes during pregnancy
2. **Preeclampsia** - Risk of pregnancy-induced hypertension with proteinuria
3. **Maternal Anemia** - Risk of hemoglobin deficiency during pregnancy
4. **Cesarean Section Risk** - Likelihood of requiring surgical delivery
5. **Postpartum Hemorrhage** - Risk of excessive bleeding after delivery

### Neonatal Health Outcomes

6. **Low Birth Weight** - Risk of infant birth weight < 2,500 grams
7. **Premature Delivery** - Risk of delivery before 37 weeks gestation
8. **Neonatal ICU Admission** - Likelihood of requiring intensive care after birth
9. **Fetal Growth Restriction** - Risk of impaired intrauterine growth
10. **Stillbirth Risk** - Risk of fetal death after 20 weeks gestation
11. **Neonatal Respiratory Distress** - Risk of breathing complications at birth

## Features

- **11 Independent Prediction Models** - Each outcome has a dedicated trained model
- **Multiple ML Algorithms** - Support for XGBoost, Random Forest, Logistic Regression, and more
- **Class Imbalance Handling** - SMOTE oversampling and class weight adjustments
- **Threshold Optimization** - Optimal decision thresholds for each model
- **Cross-Validation** - Stratified k-fold validation for robust evaluation
- **RESTful API Backend** - FastAPI-based inference server
- **Interactive Frontend** - React-based dashboard for visualization
- **Modular Architecture** - Easy to extend with additional models
- **Comprehensive Evaluation** - Multiple performance metrics and reports

## Project Structure

```
AI_BASED_BIRTH_COHORT_STUDY/
│
├── backend/                              # Backend API server
│   ├── app/
│   │   ├── main.py                      # FastAPI application entry point
│   │   ├── predictor.py                 # Model inference logic
│   │   ├── schemas.py                   # Pydantic data validation schemas
│   │   └── config.py                    # Configuration settings
│   ├── models/                          # Trained model artifacts (.pkl, .joblib)
│   │   ├── low_birth_weight_model.pkl
│   │   ├── premature_delivery_model.pkl
│   │   ├── gestational_diabetes_model.pkl
│   │   └── ...
│   └── requirements.txt                 # Python dependencies
│
├── data/                                 # Dataset storage
│   ├── low_birth_weight/
│   │   ├── raw/                         # Original data files
│   │   └── processed/                   # Preprocessed data
│   ├── premature_delivery/
│   │   ├── raw/
│   │   └── processed/
│   ├── gestational_diabetes/
│   ├── preeclampsia/
│   ├── neonatal_icu_admission/
│   ├── maternal_anemia/
│   ├── cesarean_section/
│   ├── fetal_growth_restriction/
│   ├── stillbirth/
│   ├── postpartum_hemorrhage/
│   └── neonatal_respiratory_distress/
│
├── training/                             # Model training scripts
│   ├── training_low_birth_weight.py
│   ├── training_premature_delivery.py
│   ├── training_gestational_diabetes.py
│   ├── training_preeclampsia.py
│   ├── training_neonatal_icu.py
│   ├── training_maternal_anemia.py
│   ├── training_cesarean_section.py
│   ├── training_fetal_growth_restriction.py
│   ├── training_stillbirth.py
│   ├── training_postpartum_hemorrhage.py
│   ├── training_neonatal_respiratory_distress.py
│   └── utils.py                         # Shared training utilities
│
├── frontend/                             # React dashboard application
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.js
│   ├── package.json
│   └── README.md
│
├── notebooks/                            # Jupyter notebooks for EDA
│   ├── exploratory_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_comparison.ipynb
│
├── reports/                              # Generated evaluation reports
│   ├── performance_metrics.pdf
│   ├── feature_importance.png
│   └── confusion_matrices.png
│
├── .gitignore                            # Git ignore rules
└── README.md                             # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 14+ and npm (for frontend)
- Git

### Backend Setup

1. **Clone the repository**

```bash
git clone https://github.com/kisara579/AI_based_Birth_Cohort_Study.git
cd AI_based_Birth_Cohort_Study
```

2. **Create and activate virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install Python dependencies**

```bash
pip install -r backend/requirements.txt
```

### Frontend Setup

1. **Navigate to frontend directory**

```bash
cd frontend
```

2. **Install Node.js dependencies**

```bash
npm install
```

## Usage

### Starting the Backend Server

1. **Activate virtual environment** (if not already activated)

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Start FastAPI server**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Access API documentation**

Open browser to `http://localhost:8000/docs` for interactive Swagger UI documentation.

### Starting the Frontend Dashboard

1. **Navigate to frontend directory**

```bash
cd frontend
```

2. **Start React development server**

```bash
npm start
```

3. **Access dashboard**

Open browser to `http://localhost:3000` to view the application.

### Making Predictions

**Using the API directly:**

```bash
curl -X POST "http://localhost:8000/predict/low_birth_weight" \
     -H "Content-Type: application/json" \
     -d '{
       "maternal_age": 28,
       "gestational_age": 35,
       "maternal_weight": 65,
       "previous_pregnancies": 1,
       "smoking": 0,
       "medical_conditions": 0
     }'
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.72,
  "risk_level": "High",
  "confidence": 0.85
}
```

## Training Models

### Prerequisites for Training

Ensure you have prepared your dataset in CSV format with appropriate features and target variable.

### Training Workflow

1. **Prepare your data**

Place raw data in the appropriate folder:

```
data/<outcome_name>/raw/dataset.csv
```

2. **Run training script**

```bash
python training/training_<outcome_name>.py
```

Example:

```bash
python training/training_low_birth_weight.py
```

### Training Script Structure

Each training script follows this structure:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_data(file_path):
    """Load dataset from CSV"""
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    """Preprocess features and target"""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y

def train_model(X_train, y_train):
    """Train XGBoost classifier"""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=3  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    return y_pred, y_pred_proba

def save_model(model, scaler, features, model_name):
    """Save trained model artifacts"""
    os.makedirs('backend/models', exist_ok=True)
    joblib.dump(model, f'backend/models/{model_name}_model.pkl')
    joblib.dump(scaler, f'backend/models/{model_name}_scaler.pkl')
    joblib.dump(features, f'backend/models/{model_name}_features.pkl')

def main():
    # Load and preprocess data
    df = load_data('data/outcome_name/raw/dataset.csv')
    X, y = preprocess_data(df, target_column='outcome')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    # Train and evaluate
    model = train_model(X_train_resampled, y_train_resampled)
    evaluate_model(model, X_test_scaled, y_test)

    # Save model
    save_model(model, scaler, X.columns.tolist(), 'outcome_name')

if __name__ == "__main__":
    main()
```

### Cross-Validation

Include cross-validation in your training:

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=5, scoring='roc_auc'
)
print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Threshold Optimization

Optimize decision threshold for better performance:

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find optimal threshold (e.g., maximize F1-score)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")
```

## Adding New Models

To add a new prediction model to the system:

### Step 1: Prepare Data Structure

Create data folders:

```bash
mkdir -p data/new_outcome/raw
mkdir -p data/new_outcome/processed
```

### Step 2: Create Training Script

Create `training/training_new_outcome.py` using the template above.

### Step 3: Train the Model

```bash
python training/training_new_outcome.py
```

This will generate:

- `backend/models/new_outcome_model.pkl`
- `backend/models/new_outcome_scaler.pkl`
- `backend/models/new_outcome_features.pkl`

### Step 4: Update Backend Schemas

Edit `backend/app/schemas.py`:

```python
from pydantic import BaseModel

class NewOutcomeInput(BaseModel):
    feature1: float
    feature2: int
    feature3: str
    # Add all required features

    class Config:
        schema_extra = {
            "example": {
                "feature1": 25.5,
                "feature2": 1,
                "feature3": "value"
            }
        }

class NewOutcomeOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    confidence: float
```

### Step 5: Update Predictor Logic

Edit `backend/app/predictor.py`:

```python
def load_new_outcome_model(self):
    """Load the new outcome model"""
    model_path = os.path.join(self.models_dir, 'new_outcome_model.pkl')
    scaler_path = os.path.join(self.models_dir, 'new_outcome_scaler.pkl')
    features_path = os.path.join(self.models_dir, 'new_outcome_features.pkl')

    self.new_outcome_model = joblib.load(model_path)
    self.new_outcome_scaler = joblib.load(scaler_path)
    self.new_outcome_features = joblib.load(features_path)

def predict_new_outcome(self, data: dict):
    """Make prediction for new outcome"""
    df = pd.DataFrame([data])

    # Ensure correct feature order
    df = df[self.new_outcome_features]

    # Scale features
    df_scaled = self.new_outcome_scaler.transform(df)

    # Predict
    prediction = self.new_outcome_model.predict(df_scaled)[0]
    probability = self.new_outcome_model.predict_proba(df_scaled)[0]

    # Determine risk level
    prob_positive = probability[1]
    if prob_positive > 0.7:
        risk_level = "High"
    elif prob_positive > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction": int(prediction),
        "probability": float(prob_positive),
        "risk_level": risk_level,
        "confidence": float(max(probability))
    }
```

### Step 6: Add API Endpoint

Edit `backend/app/main.py`:

```python
@app.post("/predict/new_outcome", response_model=NewOutcomeOutput)
async def predict_new_outcome(data: NewOutcomeInput):
    """
    Predict new outcome risk

    Returns:
    - prediction: 0 (negative) or 1 (positive)
    - probability: Probability of positive class
    - risk_level: Low, Medium, or High
    - confidence: Model confidence score
    """
    result = predictor.predict_new_outcome(data.dict())
    return result
```

### Step 7: Update Frontend

Add the new prediction form and visualization components to the React frontend.

### Step 8: Test the API

```bash
curl -X POST "http://localhost:8000/predict/new_outcome" \
     -H "Content-Type: application/json" \
     -d '{"feature1": 25.5, "feature2": 1, "feature3": "value"}'
```

## Evaluation Metrics

The system evaluates models using multiple metrics appropriate for imbalanced medical datasets:

### Classification Metrics

- **Accuracy** - Overall correctness of predictions
- **Precision** - Proportion of positive predictions that are correct
- **Recall (Sensitivity)** - Proportion of actual positives correctly identified
- **F1-Score** - Harmonic mean of precision and recall
- **Specificity** - Proportion of actual negatives correctly identified

### Probabilistic Metrics

- **ROC-AUC** - Area under the receiver operating characteristic curve
- **PR-AUC** - Area under the precision-recall curve
- **Log Loss** - Logarithmic loss for probability predictions

### Clinical Relevance Metrics

- **Positive Predictive Value (PPV)** - Probability that positive prediction is correct
- **Negative Predictive Value (NPV)** - Probability that negative prediction is correct
- **Number Needed to Screen (NNS)** - Patients needed to screen to identify one true positive

### Interpretation Guidelines

| Risk Level | Probability Range | Clinical Action        |
| ---------- | ----------------- | ---------------------- |
| Low        | 0.00 - 0.40       | Standard care          |
| Medium     | 0.40 - 0.70       | Enhanced monitoring    |
| High       | 0.70 - 1.00       | Immediate intervention |

### Model Performance Targets

For clinical deployment, models should achieve:

- ROC-AUC > 0.75
- Sensitivity > 0.80 (minimize false negatives)
- Specificity > 0.70
- Positive Predictive Value > 0.60

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│                  (User Interface Layer)                     │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP/REST API
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                           │
│              (API & Business Logic Layer)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Predictor   │  │   Schemas    │  │   Config     │     │
│  │   Module     │  │  (Pydantic)  │  │   Settings   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────┬────────────────────────────────────────────┘
                 │ Model Loading
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Artifacts                           │
│                  (Inference Layer)                          │
├─────────────────────────────────────────────────────────────┤
│  • Trained Models (.pkl)                                    │
│  • Feature Scalers (.pkl)                                   │
│  • Feature Lists (.pkl)                                     │
│  • 11 Independent Models                                    │
└─────────────────────────────────────────────────────────────┘
```

### Machine Learning Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → Train/Test Split
                                                        ↓
                                            Feature Scaling (StandardScaler)
                                                        ↓
                                            Class Balancing (SMOTE)
                                                        ↓
                                            Model Training (XGBoost/RF)
                                                        ↓
                                            Threshold Optimization
                                                        ↓
                                            Cross-Validation
                                                        ↓
                                            Model Evaluation
                                                        ↓
                                            Save Model Artifacts
```

### Technology Stack

**Backend:**

- FastAPI - Modern web framework for building APIs
- Scikit-learn - Machine learning algorithms
- XGBoost - Gradient boosting framework
- Pandas - Data manipulation
- NumPy - Numerical computing
- Imbalanced-learn - Handling class imbalance
- Uvicorn - ASGI server

**Frontend:**

- React - UI library
- Axios - HTTP client
- Chart.js - Data visualization
- Material-UI - Component library

**Machine Learning:**

- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression
- SMOTE - Synthetic Minority Oversampling
- StandardScaler - Feature normalization

## Contributing

Contributions are welcome from researchers, healthcare professionals, and developers.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/new-model`)
3. **Commit your changes** (`git commit -m 'Add new prediction model'`)
4. **Push to the branch** (`git push origin feature/new-model`)
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Include docstrings for all functions and classes
- Add unit tests for new features
- Update documentation as needed
- Ensure models achieve minimum performance thresholds
- Validate on independent test datasets

### Areas for Contribution

- Additional prediction models
- Improved feature engineering techniques
- Model interpretability tools (SHAP, LIME)
- Deployment optimization
- Frontend enhancements
- Documentation improvements

## Disclaimer

**IMPORTANT: This system is designed for research and educational purposes only.**

This software is NOT a medical device and has NOT been approved by regulatory authorities (FDA, CE, etc.) for clinical use. The predictions provided by this system should NOT be used as the sole basis for medical decision-making.

### Usage Restrictions

- This system should only be used by qualified healthcare professionals
- Predictions must be interpreted in the context of comprehensive clinical assessment
- The system is not a substitute for professional medical judgment
- All clinical decisions should be made by licensed medical practitioners

### Data Privacy

- Ensure compliance with HIPAA, GDPR, and local healthcare data regulations
- Implement appropriate data anonymization and security measures
- Obtain necessary ethical approvals and informed consent
- Do not use identifiable patient data without authorization

### Liability

The developers and contributors assume no liability for any harm resulting from the use of this software. Users assume all risks associated with deployment and usage.

### Validation Required

Before any clinical deployment, the models must be:

- Validated on local population data
- Reviewed by medical ethics committees
- Tested in prospective clinical studies
- Approved by appropriate regulatory bodies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{maternal_neonatal_prediction,
  title={AI-Based Maternal & Neonatal Risk Prediction System},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/AI_based_Birth_Cohort_Study}
}
```

## Contact

For questions, suggestions, or collaboration inquiries:

- Email: your.email@institution.edu
- GitHub Issues: [Create an issue](https://github.com/kisara579/AI_based_Birth_Cohort_Study/issues)
- Project Website: [Coming Soon]

---

**Version:** 1.0.0  
**Last Updated:** February 2026  
**Status:** Active Development
