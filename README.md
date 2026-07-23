# Hospital Readmission Risk & Cost Prediction

A cost-aware machine learning system that predicts 30-day hospital readmission risk and translates it into dollar-value business impact — not just a classifier, but a decision-support tool for hospital case management.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?logo=xgboost&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem

Roughly 1 in 9 patients discharged from a hospital in this dataset returns within 30 days. A readmission this soon after discharge is rarely random — it usually means a medication issue went unmanaged, a follow-up never happened, or a patient was sent home before they were stable enough to stay there. In the US, the Hospital Readmissions Reduction Program lets CMS financially penalize hospitals for excess readmissions, which turns a clinical oversight into a direct budget line.

The challenge isn't just building a classifier that outputs a probability. It's building something a case manager can actually act on *before* the patient leaves the building — and something a hospital administrator can use to justify where limited follow-up resources go.

## Business Value

A probability score on its own doesn't change behavior. This project is built around three decisions a hospital can make differently because of it:

1. **Which patients get a follow-up call.** Ranking discharges by predicted risk turns a blanket policy ("call everyone" or "call no one") into a targeted one.
2. **Which model is worth deploying.** The highest-recall model and the highest-precision model in this project are both poor business choices in isolation — one over-calls, the other misses. Model selection here is driven by a cost matrix, not a leaderboard metric.
3. **Where to set the alert threshold.** The standard 0.5 cutoff is arbitrary. Sweeping the threshold against real cost assumptions identifies the point that minimizes total dollar exposure, not the point that maximizes an abstract score.

## Dataset

| | |
|---|---|
| **Source** | [UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| **Size** | 101,766 raw encounters × 50 features, across 130 US hospitals; 99,343 encounters retained after cleaning |
| **Target** | Binary — readmitted within 30 days. The raw `readmitted` column (`<30`, `>30`, `No`) is collapsed to `1` (`<30`) vs. `0` (everything else) |
| **Class balance** | 11.39% positive class — meaningfully imbalanced, which shapes every metric and modeling decision downstream |

**Notes worth knowing before trusting the numbers:**

- 2,423 encounters coded as expired or discharged to hospice were removed *before* computing class balance — these patients cannot be readmitted by definition, and leaving them in would silently inflate the negative class.
- `weight`, `payer_code`, and `medical_specialty` are heavily missing (up to ~97%) but were **not dropped**. Missingness was tested as its own signal (it isn't predictive — readmission rate is ~11.4% either way) and then encoded as an explicit `"Missing"` category, preserving information rather than discarding rows or columns.
- `diag_1/2/3` are raw ICD-9 codes — high-cardinality by nature — grouped into 9 clinical categories (circulatory, diabetes, respiratory, etc.) following the grouping scheme from Strack et al. (2014), the paper this dataset originates from.

## Project Pipeline

| Stage | Location | Status |
|---|---|---|
| EDA & Data Cleaning | `notebooks/01_eda_feature_engineering.ipynb` | ✅ Complete |
| Feature Engineering | `notebooks/01_eda_feature_engineering.ipynb` | ✅ Complete |
| Preprocessing & Model Training (6 models) | `notebooks/02_preprocessing_modeling.ipynb` | ✅ Complete |
| Cost-Aware Model Selection | `notebooks/02_preprocessing_modeling.ipynb`, Block 10 | ✅ Complete |
| Threshold Simulation | `notebooks/02_preprocessing_modeling.ipynb`, Block 11 | ✅ Complete |
| Specialty-Specific vs. General Model Comparison | — | ⏳ Planned, not started |
| FastAPI Inference Backend | `app/main.py` | ✅ Complete |
| Streamlit Dashboard | `app/streamlit_app.py` | ✅ Complete |
| LLM-Generated Risk Notes | `app/main.py` | ✅ Complete |
| Public Deployment | — | ⏳ Not yet deployed |

## Feature Engineering

Every transformation here was a response to something found in EDA, not a default checklist item:

- **ICD-9 diagnosis grouping.** `diag_1`, `diag_2`, `diag_3` collapse from hundreds of raw codes into 9 clinical categories. Circulatory conditions turned out to be the single largest group (29.9%) — larger than diabetes itself (8.7%) — since diabetic patients are frequently admitted primarily for cardiovascular complications rather than diabetes directly.
- **Medication consolidation.** Of 23 individual medication columns, 15 were near-constant (>99% one value, almost always "No") and contributed nothing to a model. They were dropped only *after* folding their signal into a single `num_meds_changed` count, so the underlying information wasn't lost, just compressed.
- **`total_prior_visits`.** Sum of `number_outpatient`, `number_emergency`, and `number_inpatient`. Prior utilization — especially prior inpatient stays — turned out to be the strongest single predictor in the dataset: readmission rate climbs from ~8.6% at zero prior inpatient visits to ~46% at eight.
- **`diabetes_related` flag.** Captures whether diabetes appears *anywhere* in a patient's diagnoses (37.9% of encounters) versus only as the primary diagnosis (8.7%). It turned out to be a weak standalone predictor (11.6% vs. 11.0% readmission rate) but is retained for filtering and interpretability.
- **Discharge disposition filtering.** Encounters discharged to expired or hospice status are removed before modeling — a patient who died or entered hospice cannot be "readmitted," so including them would corrupt both the target distribution and any cost calculation downstream.
- **Scaling.** Applied only for Logistic Regression, which is scale-sensitive; tree-based and boosting models train on unscaled features, since splitting on raw thresholds makes scaling redundant for them.

## Models Compared

Six models were trained and evaluated identically, each handling class imbalance through the mechanism appropriate to its library (`class_weight='balanced'` for Logistic Regression, Decision Tree, and Random Forest; `scale_pos_weight` for XGBoost; CatBoost's native class-weighting) — deliberately avoiding synthetic resampling (SMOTE) to keep the real data distribution intact.

| Model | Role in the Comparison |
|---|---|
| **Logistic Regression** | Interpretable baseline; coefficient signs show the direction of each risk factor |
| **Gaussian Naive Bayes** | Standard probabilistic baseline in readmission literature — no native class-imbalance handling, a documented limitation |
| **Decision Tree** | Demonstrates the overfitting a single tree exhibits, motivating the ensemble methods below |
| **Random Forest** | Captures non-linear feature interactions, ranks feature importance |
| **XGBoost** | Gradient boosting benchmark, strong on structured clinical/tabular data |
| **CatBoost** | Used directly in the diabetes-readmission research this project is grounded in |

**Deliberately excluded: deep neural networks / LSTMs.** The literature shows these only outperform on temporal, multi-encounter patient sequences. This dataset is single-encounter and tabular, so a DNN adds no literature-backed benefit over a well-tuned ensemble here.

**Evaluation metrics:** Precision, Recall, F1, and ROC-AUC — with recall weighted as the priority metric, since a false negative (a missed readmission) is the costly error in this problem, not a false positive.

## Key Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Gaussian Naive Bayes | 0.117 | **0.970** | 0.209 | 0.596 |
| XGBoost | 0.176 | 0.571 | **0.269** | 0.644 |
| Logistic Regression | 0.181 | 0.527 | 0.269 | 0.655 |
| CatBoost | 0.186 | 0.496 | 0.271 | 0.647 |
| Decision Tree | 0.148 | 0.188 | 0.165 | 0.523 |
| Random Forest | **0.401** | 0.034 | 0.062 | **0.657** |

Accuracy alone would be actively misleading here: a model that predicts "never readmitted" for every patient scores **~89% accuracy** while catching zero real readmissions. No accuracy figures are reported for this reason — recall, precision, F1, and ROC-AUC tell the real story.

**Cost-aware model selection** reframes the choice entirely. Using illustrative costs of **$5,000 per missed readmission** and **$500 per unnecessary follow-up call**:

| Model | False Negatives | False Positives | Total Estimated Cost |
|---|---|---|---|
| **XGBoost** | 1,004 | 6,255 | **$8,147,500** |
| Logistic Regression | 1,108 | 5,589 | $8,334,500 |
| CatBoost | 1,180 | 5,075 | $8,437,500 |
| Gaussian Naive Bayes | 63 | 17,190 | $8,910,000 |
| Decision Tree | 1,901 | 2,542 | $10,776,000 |
| Random Forest | 2,262 | 118 | $11,369,000 |

**XGBoost wins on cost despite having neither the top recall nor the top precision** — the metric-leaderboard winner and the cost-leaderboard winner are different models, which is exactly the point of building the cost layer at all.

**Threshold simulation** on the winning model shows the default 0.5 cutoff isn't cost-optimal either: sweeping thresholds from 0.05 to 0.90 finds the minimum total cost at **threshold 0.40** ($7,925,000), a **~2.7% reduction** from the default threshold's $8,147,500 — achieved simply by flagging a wider net of patients as high-risk before deployment, at no model-retraining cost.

## Repository Structure

```
readmission-project/
├── app/
│   ├── main.py                     # FastAPI inference service + LLM risk-note generation
│   ├── streamlit_app.py            # Streamlit dashboard (Predict / Data Explorer / Model Insights / About)
│   ├── model.pkl                   # Trained XGBoost model (cost-optimal selection)
│   ├── scaler.pkl                  # Fitted StandardScaler
│   └── model_columns.pkl           # Expected feature/column order for inference
├── data/
│   ├── diabetic_data.csv           # Raw UCI dataset
│   ├── diabetic_data_engineered.csv
│   ├── ids_mapping.csv             # UCI's admission/discharge/source ID lookup table
│   ├── model_results.csv           # Precision / Recall / F1 / ROC-AUC per model
│   ├── cost_results.csv            # Total estimated cost per model
│   └── threshold_results.csv       # Cost vs. threshold sweep for the deployed model
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb
│   └── 02_preprocessing_modeling.ipynb
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/YomnaSabry172/readmission-project.git
cd readmission-project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Set a Groq API key for LLM-generated risk notes (a rule-based fallback runs automatically if this is unset):

```bash
export GROQ_API_KEY=your_key_here    # Windows: set GROQ_API_KEY=your_key_here
```

## Usage

**1. Start the inference API:**

```bash
cd app
uvicorn main:app --reload
```

**2. Launch the dashboard** (in a second terminal, from the project root):

```bash
streamlit run app/streamlit_app.py
```

The dashboard calls the API at `http://127.0.0.1:8000/predict` and degrades gracefully with a clear connection message if the backend isn't running.

## API

Built with **FastAPI**. Single inference endpoint:

### `POST /predict`

Accepts a patient's clinical and demographic features and returns a readmission probability with an LLM-generated risk note.

**Request body** (abridged — see `app/main.py` for the full schema):

```json
{
  "age": "[70-80)",
  "time_in_hospital": 5,
  "num_lab_procedures": 45,
  "num_medications": 12,
  "number_outpatient": 0,
  "number_emergency": 1,
  "number_inpatient": 2,
  "total_prior_visits": 3,
  "diag_1_group": "Circulatory",
  "diabetes_related": 1
}
```

**Response:**

```json
{
  "readmission_probability": 0.4123,
  "risk_note": "This patient shows a moderate readmission risk, driven primarily by two prior inpatient visits and a circulatory primary diagnosis. Consider a follow-up call within the first week post-discharge."
}
```

The risk note is generated live via Groq (`llama-3.3-70b-versatile`), grounded strictly in the model's own probability output and the patient's actual features — the LLM performs no prediction of its own, only explains a decision the trained model already made.

## Dashboard

The Streamlit app is organized into four tabs:

- **🔮 Predict** — enter a patient's details and receive a live readmission probability plus an LLM-generated risk note.
- **📊 Data Explorer** — sample rows from the engineered dataset alongside the headline EDA findings (target imbalance, strongest predictors, diagnosis distribution).
- **📈 Model Insights** — side-by-side evaluation metrics for all six models, the cost-aware ranking, and the cost-vs-threshold curve for the deployed model.
- **ℹ️ About** — project summary, modeling approach, and known limitations, written for a non-technical reviewer.

## Deployment

Not yet publicly deployed. Planned path:

- **Streamlit Community Cloud** for the dashboard (free tier, fastest to stand up for a small team project)
- **Render** or an equivalent free-tier host for the FastAPI backend, if front and back end are split across services

🔗 *Live dashboard: add link here once deployed*
🔗 *API base URL: add link here once deployed*

## Limitations

- **Cost figures are stated assumptions, not audited hospital data.** $5,000 per missed readmission and $500 per unnecessary follow-up are illustrative — a real deployment should substitute a hospital's actual penalty and intervention costs.
- **Gaussian Naive Bayes has no native class-imbalance handling**, unlike the other five models. It was trained as-is and its unusually high recall / low precision profile should be read with that in mind, not treated as a fair comparison.
- **No specialty-specific modeling yet.** Training separate models per diagnosis group (circulatory, diabetes, respiratory) and comparing against the general model is designed but not yet implemented.
- **Correlation, not causation.** Feature importance here reflects predictive association with readmission, not a validated causal mechanism — it should inform triage priorities, not clinical decisions directly.

## Future Improvements

- Implement the specialty-specific vs. general model comparison to test whether one-size-fits-all modeling is blurring together patient subgroups with genuinely different risk patterns.
- Replace assumed cost constants with real hospital financial data, including CMS penalty exposure specific to a given facility.
- Incorporate temporal, multi-encounter patient history — the point at which sequence models (LSTMs, temporal transformers) would start to earn their added complexity over the current tabular approach.
- Add model monitoring for drift, with periodic recalibration of the cost-optimal decision threshold as real-world outcomes accumulate.
- Expand the FastAPI service with authentication and batch-prediction endpoints for integration into a hospital's existing EHR workflow.

## Team

Five-person team project, built with rotating pipeline ownership — every member worked across EDA, feature engineering, modeling, the cost/novelty analysis layer, and deployment rather than specializing in a single stage.

## License

Released under the [MIT License](LICENSE).
