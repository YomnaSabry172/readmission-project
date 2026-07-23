from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from groq import Groq

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
        return (
            f"This patient has a {risk_level} estimated risk of 30-day readmission "
            f"({prob:.1%} predicted probability), based on their prior visit history "
            f"and clinical profile. (Note: LLM note generation unavailable — {str(e)})"
        )

@app.post("/predict")
def predict(patient: PatientInput):
    patient_dict = patient.dict()
    input_df = pd.DataFrame([patient_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(input_encoded)[:, 1][0]
    note = generate_risk_note(prob, patient_dict)

    return {
        "readmission_probability": round(float(prob), 4),
        "risk_note": note
    }