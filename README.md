# AI-Based Maternal & Neonatal Risk Prediction System

## Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 18+ and npm (for frontend)
- Git

### Backend Setup

1. **Clone the repository**

```bash
git clone https://github.com/kisara579/AI_based_Birth_Cohort_Study.git
cd AI_based_Birth_Cohort_Study
```

2. **Create and activate virtual environment**

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt (cmd)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows Git Bash
python -m venv .venv
source .venv/Scripts/activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
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
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt (cmd)
.venv\Scripts\activate.bat

# Windows Git Bash
source .venv/Scripts/activate

# Linux/Mac
source .venv/bin/activate
```

2. **Start FastAPI server**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```


### Starting the Frontend Dashboard

1. **Navigate to frontend directory**

```bash
cd frontend
```

2. **Start React development server**

```bash
npm run dev
```

3. **Access dashboard**

Open browser to `http://localhost:5173` to view the application.


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
- React Router DOM - Client-side routing
- Vite - Frontend build tool and dev server
- ESLint - Code linting

**Machine Learning:**

- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression
- Linear SVM
- StandardScaler - Feature normalization


