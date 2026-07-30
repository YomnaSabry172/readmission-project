"""
Chatbot layer for the readmission project.

Two jobs live here, kept deliberately separate from main.py's plain /predict route:

1. extract_fields_from_text()
   Takes whatever the user typed in free text (plus prior turns of the chat) and
   asks Groq to pull out any of the 30 PatientInput fields it can find. Anything
   the user never mentioned comes back as null — we never let the LLM guess a
   clinical value silently.

2. fill_defaults()
   Takes that partial dict and fills every null with a fixed, documented default
   (the same "typical patient" values already used as the Streamlit form's
   pre-filled selections/sliders, which in turn reflect the dataset's own modal/
   median values). This keeps "fill in anything suitable" honest and reproducible
   instead of leaving it up to the LLM to invent clinical numbers.

3. generate_full_report()
   A longer-form Groq call that produces a multi-section, human-readable report
   (as opposed to the 2-3 sentence risk_note main.py already generates for the
   plain form flow).
"""

import os
import json
import logging          # ADD
from groq import Groq

logger = logging.getLogger(__name__)   # ADD
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Defaults: the "typical patient" fallback for any field the user doesn't
# mention. These mirror the Streamlit form's own default selections, which
# were chosen to reflect the dataset's modal/median values (see streamlit_app.py
# tab5) — e.g. weight is missing for ~97% of records, so "Missing" is the
# realistic default rather than a guess.
# ---------------------------------------------------------------------------
DEFAULT_PATIENT = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "weight": "Missing",
    "time_in_hospital": 3,
    "payer_code": "Missing",
    "admission_type_desc": "Emergency",
    "discharge_disposition_desc": "Discharged to home",
    "admission_source_desc": "Emergency Room",
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 7,
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
}

# Fields the model needs that we always derive/compute ourselves rather than
# ask the LLM to extract or default — same rule streamlit_app.py follows for
# diabetes_related and total_prior_visits.
DERIVED_ONLY = {"total_prior_visits", "diabetes_related"}

VALID_VALUES = {
    "race": ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"],
    "gender": ["Female", "Male"],
    "age": ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
             "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
    "max_glu_serum": ["Missing", "Norm", ">200", ">300"],
    "A1Cresult": ["Missing", "Norm", ">7", ">8"],
    "metformin": ["No", "Steady", "Up", "Down"],
    "repaglinide": ["No", "Steady", "Up", "Down"],
    "glimepiride": ["No", "Steady", "Up", "Down"],
    "glipizide": ["No", "Steady", "Up", "Down"],
    "glyburide": ["No", "Steady", "Up", "Down"],
    "pioglitazone": ["No", "Steady", "Up", "Down"],
    "rosiglitazone": ["No", "Steady", "Up", "Down"],
    "insulin": ["No", "Steady", "Up", "Down"],
    "change": ["No", "Ch"],
    "diabetesMed": ["Yes", "No"],
    "diag_1_group": ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                      "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"],
    "diag_2_group": ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                      "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"],
    "diag_3_group": ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                      "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"],
}

EXTRACTABLE_FIELDS = [f for f in DEFAULT_PATIENT.keys()]

EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_patient_fields",
        "description": (
            "Record every patient field the user's message actually states or clearly "
            "implies. Omit (do not include the key at all) any field the user did not "
            "mention — never guess a clinical value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                field: (
                    {"type": "string", "enum": VALID_VALUES[field]}
                    if field in VALID_VALUES
                    else {"type": "number"}
                )
                for field in EXTRACTABLE_FIELDS
            },
            "additionalProperties": False,
        },
    },
}


def extract_fields_from_text(message: str, history: list[dict] | None = None) -> dict:
    """
    Calls Groq with tool-calling to pull out any PatientInput fields mentioned
    in the user's message (optionally with prior chat turns for context).
    Returns a partial dict — only fields the user actually stated.
    """
    history = history or []
    messages = [
        {
            "role": "system",
            "content": (
                "You extract structured hospital-patient data from a case manager's "
                "free-text description, for a 30-day readmission risk model. Call "
                "record_patient_fields with ONLY the fields the text actually states "
                "or clearly implies (e.g. 'diabetic senior' implies diag group Diabetes "
                "and an older age bracket, but does not imply lab counts). Never invent "
                "specific numbers (lab counts, medication counts) that weren't stated."
            ),
        },
        *history,
        {"role": "user", "content": message},
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_patient_fields"}},
            max_tokens=500,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        extracted = json.loads(tool_call.function.arguments)
        clean = {}
        for k, v in extracted.items():
            if k not in DEFAULT_PATIENT:
                continue
            if k in VALID_VALUES and v not in VALID_VALUES[k]:
                continue
            clean[k] = v
        return clean
    except Exception as e:
        # CHANGED: log the real error instead of silently discarding it.
        # This is the exact bug you're hitting — before this fix, ANY Groq
        # failure (bad key, network, rate limit, deprecated model) silently
        # produced {} here, which made every patient collapse to the same
        # DEFAULT_PATIENT and therefore the same prediction every time.
        logger.error("extract_fields_from_text: Groq call failed: %r", e)
        return {}

def fill_defaults(partial: dict) -> tuple[dict, list[str]]:
    """
    Merges a partial patient dict with DEFAULT_PATIENT. Returns (full_dict, filled_keys)
    so callers/UI can show the user exactly what was assumed vs. what they stated.
    """
    full = dict(DEFAULT_PATIENT)
    filled_keys = []
    for key in DEFAULT_PATIENT:
        if key in partial and partial[key] is not None:
            full[key] = partial[key]
        else:
            filled_keys.append(key)

    # Derived fields — always computed, never taken from extraction or defaults dict.
    full["total_prior_visits"] = (
        full["number_outpatient"] + full["number_emergency"] + full["number_inpatient"]
    )
    full["diabetes_related"] = 1 if "Diabetes" in (
        full["diag_1_group"], full["diag_2_group"], full["diag_3_group"]
    ) else 0

    return full, filled_keys


def generate_full_report(prob: float, patient_dict: dict, filled_keys: list[str] | None = None) -> str:
    """
    Longer-form AI report (as opposed to main.py's 2-3 sentence risk_note).
    Sections: summary, key risk drivers, what was assumed vs. stated, suggested
    next steps for a case manager. Grounded strictly in the given inputs.
    """
    filled_keys = filled_keys or []
    risk_level = "high" if prob >= 0.5 else "moderate" if prob >= 0.3 else "low"

    assumed_note = (
        f"The following fields were not provided and were filled with typical/default "
        f"values: {', '.join(filled_keys)}." if filled_keys
        else "All fields were provided by the user; nothing was defaulted."
    )

    prompt = f"""You are assisting a hospital case manager using a 30-day readmission
risk model. The model predicted a {risk_level} risk ({prob:.1%} probability) for a
patient with this full profile (after defaults were applied for anything unstated):

{json.dumps(patient_dict, indent=2)}

{assumed_note}

Write a structured plain-language report with these sections, using short paragraphs
or bullet points, and clear section headers:

## Risk Summary
One or two sentences on the overall risk level and probability.

## Key Contributing Factors
2-4 bullet points naming the specific fields above that most plausibly drive this
risk level (e.g. prior visit counts, time in hospital, diagnosis group). Only reference
fields present in the data above.

## What Was Assumed vs. Stated
Briefly note which parts of this assessment rest on stated patient information vs.
default assumptions, so the case manager knows how much to trust the specifics.

## Suggested Next Steps
2-4 concrete, non-clinical case-management actions (e.g. schedule a follow-up call,
verify medication adherence, coordinate a home health visit) — no medical/treatment
advice, only care-coordination suggestions.

Do not invent information not present in the data above. Do not give medical advice."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=650,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return (
            f"AI report unavailable ({e}). Fallback summary: this patient has a "
            f"{risk_level} estimated 30-day readmission risk ({prob:.1%})."
        )