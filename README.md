# Hospital Readmission Risk & Cost Prediction

A full-stack machine learning project for predicting 30-day hospital readmission risk, comparing models through a cost-aware lens, and surfacing actionable insights through a Streamlit dashboard and FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?logo=xgboost&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project tackles a practical hospital problem: identifying which patients are most likely to be readmitted within 30 days after discharge and turning that prediction into a decision-support tool for case management.

The workflow includes:

- data cleaning and feature engineering in the notebooks
- training and comparing several machine learning models
- selecting the best model using a cost-oriented framing rather than accuracy alone
- deploying a prediction API and an interactive dashboard
- generating plain-language risk notes with Groq

## Why this matters

Readmissions are expensive for hospitals and often reflect unmet follow-up needs, medication issues, or discharge instability. The goal is not only to classify patients, but to help hospitals prioritize limited follow-up resources more effectively.

The project is designed around three business decisions:

1. Which patients should receive a follow-up call
2. Which model is worth deploying for real-world use
3. Which probability threshold should trigger an alert

## Dataset

The project uses the UCI Diabetes 130-US Hospitals dataset, covering inpatient encounters from 130 US hospitals between 1999 and 2008.

Key details:

- Raw encounters: 101,766
- Retained after cleaning: 99,343
- Target: binary 30-day readmission flag
- Positive class rate: about 11.39%

The dataset includes demographic, admission, diagnosis, medication, lab, and prior-utilization features. A number of preprocessing choices were made to preserve information while keeping the modeling pipeline faithful to the underlying problem.

## Repository structure

```text
readmission-project/
├── app/
│   ├── main.py               # FastAPI backend with prediction and report endpoints
│   ├── streamlit_app.py     # Interactive Streamlit dashboard
│   ├── chatbot.py           # Chat extraction/default-filling/report generation logic
│   ├── static/              # Static assets for the frontend widget
│   └── *.pkl                # Trained model artifacts
├── data/
│   ├── diabetic_data.csv
│   ├── diabetic_data_engineered.csv
│   └── ids_mapping.csv
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb
│   └── 02_preprocessing_modeling.ipynb
├── requirements.txt
└── LICENSE
```

## Features

- Cost-aware model comparison and threshold analysis
- FastAPI-based prediction service
- Streamlit dashboard with multiple tabs for problem context, dataset exploration, model explanation, visuals, and prediction
- AI-generated risk notes and a conversational chat assistant
- Notebook-based experimentation and reproducible preprocessing

## Tech stack

- Python
- pandas, NumPy, scikit-learn
- XGBoost and CatBoost
- FastAPI
- Streamlit
- Groq for LLM-generated explanations

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd readmission-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set the Groq API key (optional but recommended)

```bash
export GROQ_API_KEY=your_key_here
```

On Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_key_here"
```

If the key is not set, the app will fall back to rule-based or default-generated text instead of failing completely.

## Running the app locally

### Start the FastAPI backend

From the project root:

```bash
cd app
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000.

### Launch the Streamlit dashboard

In a second terminal, from the project root:

```bash
streamlit run app/streamlit_app.py
```

## API endpoints

The backend exposes:

- POST /predict: returns a readmission probability and a short risk note
- POST /chat: accepts free-text input and extracts patient fields before scoring
- POST /report: generates a longer AI report for the same patient context

Example request payload for /predict:

```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[70-80)",
  "weight": "Missing",
  "time_in_hospital": 5,
  "payer_code": "Missing",
  "admission_type_desc": "Emergency",
  "discharge_disposition_desc": "Discharged to home",
  "admission_source_desc": "Emergency Room",
  "num_lab_procedures": 45,
  "num_procedures": 2,
  "num_medications": 12,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2,
  "number_diagnoses": 8,
  "total_prior_visits": 3,
  "max_glu_serum": "Missing",
  "A1Cresult": "Missing",
  "metformin": "No",
  "repaglinide": "No",
  "glimepiride": "No",
  "glipizide": "No",
  "glyburide": "No",
  "pioglitazone": "No",
  "rosiglitazone": "No",
  "insulin": "No",
  "change": "No",
  "diabetesMed": "Yes",
  "num_meds_changed": 0,
  "diag_1_group": "Circulatory",
  "diag_2_group": "Diabetes",
  "diag_3_group": "Other",
  "diabetes_related": 1
}
```

## Notes on modeling

The notebooks contain the full exploration pipeline, including:

- target definition and cleaning
- feature engineering and encoding
- comparison of several model families
- cost-based evaluation and threshold selection

The deployed app is centered on a trained model artifact and supports a user-facing explanation layer on top of that prediction.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
