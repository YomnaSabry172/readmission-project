from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from groq import Groq

import chatbot

app = FastAPI()

# Lets the standalone chat_widget.html (opened from any origin, or embedded in
# any site) call this API directly from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ADD: fail loudly at startup instead of discovering it later via
# "Connection error" strings buried in every response.
if not os.environ.get("GROQ_API_KEY"):
    import logging
    logging.getLogger(__name__).warning(
        "GROQ_API_KEY is not set — extraction and AI notes will fall back "
        "to defaults/rule-based text on every request."
    )
    
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


class PatientInput(BaseModel):
    race: str
    gender: str
    age: str
    weight: str
    time_in_hospital: int
    payer_code: str
    admission_type_desc: str
    discharge_disposition_desc: str
    admission_source_desc: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    total_prior_visits: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    glimepiride: str
    glipizide: str
    glyburide: str
    pioglitazone: str
    rosiglitazone: str
    insulin: str
    change: str
    diabetesMed: str
    num_meds_changed: int
    diag_1_group: str
    diag_2_group: str
    diag_3_group: str
    diabetes_related: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    generate_report: bool = False


def _predict_from_patient_dict(patient_dict: dict) -> float:
    input_df = pd.DataFrame([patient_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    return float(model.predict_proba(input_encoded)[:, 1][0])


def generate_risk_note(prob: float, patient_dict: dict) -> str:
    risk_level = "high" if prob >= 0.5 else "moderate" if prob >= 0.3 else "low"

    prompt = f"""You are assisting a hospital case manager. A machine learning model
predicted a {risk_level} risk of 30-day readmission ({prob:.1%} probability) for a
patient with these characteristics:

- Age group: {patient_dict['age']}
- Prior inpatient visits (past year): {patient_dict['number_inpatient']}
- Prior emergency visits (past year): {patient_dict['number_emergency']}
- Total prior visits: {patient_dict['total_prior_visits']}
- Time in hospital (this stay): {patient_dict['time_in_hospital']} days
- Number of diagnoses: {patient_dict['number_diagnoses']}
- Number of medications: {patient_dict['num_medications']}
- Primary diagnosis group: {patient_dict['diag_1_group']}
- Diabetes-related: {"yes" if patient_dict['diabetes_related'] else "no"}

Write a concise, 2-3 sentence plain-language note for the case manager explaining this
patient's readmission risk level and which factors likely contributed most. Do not invent
information not given above. Do not give medical advice — only summarize the risk
context to support the case manager's own judgment."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("generate_risk_note: Groq call failed: %r", e)
        return (
            f"This patient has a {risk_level} estimated risk of 30-day readmission "
            f"({prob:.1%} predicted probability), based on their prior visit history "
            f"and clinical profile. (Note: LLM note generation unavailable — {str(e)})"
        )

@app.post("/predict")
def predict(patient: PatientInput):
    patient_dict = patient.dict()
    prob = _predict_from_patient_dict(patient_dict)
    note = generate_risk_note(prob, patient_dict)

    return {
        "readmission_probability": round(prob, 4),
        "risk_note": note
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Free-text entry point: takes whatever the user typed (plus optional prior
    turns), extracts any patient fields mentioned, fills the rest with the
    documented defaults in chatbot.DEFAULT_PATIENT, runs the same model as
    /predict, and returns a short risk note. Optionally also returns a longer
    AI-generated report when generate_report=True.
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    extracted = chatbot.extract_fields_from_text(req.message, history)
    patient_dict, filled_keys = chatbot.fill_defaults(extracted)

    prob = _predict_from_patient_dict(patient_dict)
    note = generate_risk_note(prob, patient_dict)

    result = {
        "readmission_probability": round(prob, 4),
        "risk_note": note,
        "patient_used": patient_dict,
        "fields_extracted_from_message": [k for k in extracted if extracted.get(k) is not None],
        "fields_filled_with_defaults": filled_keys,
    }

    if req.generate_report:
        result["full_report"] = chatbot.generate_full_report(prob, patient_dict, filled_keys)

    return result


class ReportRequest(BaseModel):
    patient: PatientInput
    readmission_probability: float
    fields_filled_with_defaults: list[str] = []


@app.post("/report")
def report(req: ReportRequest):
    """
    Generates the longer-form AI report for a patient/probability that's
    already been computed (e.g. from the plain /predict form flow, when the
    user clicks a separate "Generate AI report" option instead of just the
    short risk note).
    """
    patient_dict = req.patient.dict()
    full_report = chatbot.generate_full_report(
        req.readmission_probability, patient_dict, req.fields_filled_with_defaults
    )
    return {"full_report": full_report}