# CreditIQ — AI Credit Risk Intelligence Platform

Enterprise-style credit risk assessment platform built on the Home Credit Default Risk dataset. Combines a LightGBM default model, SHAP-based explainability, and a structured decision-rules engine behind a FastAPI backend and Streamlit dashboard.

**Author:** [Nailasalim](https://github.com/Nailasalim)  
**Status:** Phase 1 checkpoint — inference, explainability, and decision rules operational

---

## Project Overview

CreditIQ predicts applicant default probability, assigns risk bands, and recommends approve / review / decline outcomes. Underwriters can inspect global and applicant-level drivers, browse portfolio-derived policy rules, and see which rules fire for a given application.

The stack separates **model inference** (FastAPI) from **presentation** (Streamlit), so the same backend can serve multiple clients.

---

## Problem Statement

Home Credit serves a large, imbalanced portfolio (~8% default rate). Manual underwriting does not scale, and black-box scores alone are insufficient for regulated credit decisions.

This project addresses:

- **Default prediction** — probability of non-repayment at the portfolio-tuned threshold (0.67)
- **Transparent decisions** — risk bands, confidence, and rule-level explanations
- **Explainability** — SHAP-based feature drivers at portfolio and applicant level
- **Policy alignment** — business rules derived from model behaviour and portfolio patterns

---

## Features Implemented

| Area | Status | Description |
|------|--------|-------------|
| **Risk Prediction** | ✅ | Applicant form → `POST /decision` → score, band, recommendation |
| **Explainability** | ✅ | Global SHAP importance; individual prediction contributions |
| **Decision Rules** | ✅ | 7 structured rules (R001–R007), live evaluation, mined metrics |
| **FastAPI backend** | ✅ | `/health`, `/predict`, `/decision`, `/rules` |
| **Dashboard shell** | 🔶 | NeoStats-style dark UI; Dashboard, Data Explorer, Talk To Data, Reports are placeholders |

### Model performance (Phase 1)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7516 |
| Threshold | 0.67 |
| Accuracy | 0.862 |
| Precision | 0.257 |
| Recall | 0.374 |
| F1 | 0.305 |
| Features | 21 |

---

## Architecture Overview

```
┌─────────────────────┐     HTTP      ┌──────────────────────┐
│  Streamlit UI       │ ────────────► │  FastAPI (src/api)   │
│  ui/streamlit_app   │               │  /predict /decision  │
└─────────────────────┘               │  /rules              │
                                      └──────────┬───────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    ▼                            ▼                            ▼
            src/ml/predict.py           src/ml/rules.py              src/data/preprocessor.py
            (LightGBM inference)        (RuleCondition evaluators)   (feature engineering)
                    │                            │
                    └────────────┬───────────────┘
                                 ▼
                         models/
                    model.pkl · metrics.json
                    feature_names.json · shap_values.npy
```

**Decision flow:** raw applicant fields → feature engineering → model probability → risk band → rule evaluation → final recommendation (with conflict resolution when approve/decline rules overlap).

---

## Project Structure

```
credit_risk_prediction/
├── src/
│   ├── api/main.py           # FastAPI entry point
│   ├── data/                 # Artifact loader, preprocessor
│   ├── ml/                   # predict, rules, rule_explain
│   └── utils/config.py       # Paths, thresholds, portfolio size
├── ui/
│   ├── streamlit_app.py      # App shell & navigation
│   ├── risk_prediction_page.py
│   ├── explainability_page.py
│   └── decision_rules_page.py
├── models/                   # Trained artifacts (committed for inference)
├── documents/
│   └── phase1_findings.md    # EDA & training summary
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation Instructions

**Requirements:** Python 3.11+, Git

```powershell
# Clone the repository
git clone https://github.com/Nailasalim/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Set the project root on `PYTHONPATH` when running locally:

```powershell
$env:PYTHONPATH="."
```

Optional environment variable:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CREDIT_RISK_API_URL` | `http://127.0.0.1:8000` | Streamlit → API base URL |

---

## Running the API

From the project root:

```powershell
$env:PYTHONPATH="."
uvicorn src.api.main:app --reload --port 8000
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Default probability, risk band, decision |
| POST | `/decision` | Prediction + matched rules + recommendation |
| GET | `/rules` | Active rules catalog |

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Running the Streamlit UI

With the API running in a separate terminal:

```powershell
$env:PYTHONPATH="."
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Use **Risk Prediction** to submit an applicant, then **Explainability** or **Decision Rules** to inspect results.

---

## Current Screenshots

<!-- Add screenshots before final submission -->

| Screen | File |
|--------|------|
| Risk Prediction | `documents/screenshots/` — _pending_ |
| Explainability | `documents/screenshots/shap_summary.png` |
| Decision Rules | _pending_ |
| ROC / metrics | `documents/screenshots/roc_curve.png` |

---

## Future Work

- [ ] Dashboard KPIs and portfolio analytics
- [ ] Data Explorer and Talk To Data (NL-to-SQL)
- [ ] Reports export (PDF / audit trail)
- [ ] Retrain pipeline (`src/ml/train.py`) and CI for model promotion
- [ ] Git LFS or artifact registry for large model/SHAP files
- [ ] Docker Compose for API + UI orchestration
- [ ] Authentication and role-based access for production deployment

---

## License

Private submission checkpoint — all rights reserved unless otherwise specified by the assignment.
