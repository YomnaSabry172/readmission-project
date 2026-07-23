import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Readmission Risk Predictor", layout="wide")

API_URL = "http://127.0.0.1:8000/predict"

st.title("🏥 Hospital Readmission Risk Predictor")
st.caption("Cost-aware ML model predicting 30-day readmission risk — UCI Diabetes 130-US Hospitals dataset")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Data Explorer", "📈 Model Insights", "ℹ️ About"])

# TAB 1: PREDICT
with tab1:
    st.header("Patient Risk Prediction")
    st.write("Fill in patient details below to get a readmission risk prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)",
                                          "[40-50)", "[50-60)", "[60-70)", "[70-80)",
                                          "[80-90)", "[90-100)"])
        weight = st.selectbox("Weight", ["Missing"])  # ~97% missing in real data — default is realistic

        st.subheader("Admission Details")
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 3)
        payer_code = st.selectbox("Payer code", ["Missing", "MC", "HM", "SP", "BC", "Other"])
        admission_type_desc = st.selectbox("Admission type", ["Emergency", "Urgent", "Elective", "Newborn", "Not Available"])
        discharge_disposition_desc = st.selectbox("Discharge disposition", ["Discharged to home", "Other"])
        admission_source_desc = st.selectbox("Admission source", ["Emergency Room", "Physician Referral", "Other"])

    with col2:
        st.subheader("Clinical Counts")
        num_lab_procedures = st.slider("Number of lab procedures", 0, 130, 40)
        num_procedures = st.slider("Number of procedures", 0, 10, 1)
        num_medications = st.slider("Number of medications", 0, 80, 15)
        number_outpatient = st.slider("Outpatient visits (past year)", 0, 40, 0)
        number_emergency = st.slider("Emergency visits (past year)", 0, 40, 0)
        number_inpatient = st.slider("Inpatient visits (past year)", 0, 20, 0)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
        total_prior_visits = st.slider(
            "Total prior visits (outpatient + emergency + inpatient)", 0, 100,
            number_outpatient + number_emergency + number_inpatient
        )

        st.subheader("Test Results")
        max_glu_serum = st.selectbox("Max glucose serum result", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C result", ["None", "Norm", ">7", ">8"])

    with col3:
        st.subheader("Medications")
        med_options = ["No", "Steady", "Up", "Down"]
        metformin = st.selectbox("Metformin", med_options)
        repaglinide = st.selectbox("Repaglinide", med_options)
        glimepiride = st.selectbox("Glimepiride", med_options)
        glipizide = st.selectbox("Glipizide", med_options)
        glyburide = st.selectbox("Glyburide", med_options)
        pioglitazone = st.selectbox("Pioglitazone", med_options)
        rosiglitazone = st.selectbox("Rosiglitazone", med_options)
        insulin = st.selectbox("Insulin", med_options)
        change = st.selectbox("Change in medications", ["No", "Ch"])
        diabetesMed = st.selectbox("Diabetes medication prescribed", ["Yes", "No"])
        num_meds_changed = st.slider("Number of medications changed", 0, 10, 0)

        st.subheader("Diagnosis Groups")
        diag_groups = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                        "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"]
        diag_1_group = st.selectbox("Primary diagnosis group", diag_groups)
        diag_2_group = st.selectbox("Secondary diagnosis group", diag_groups)
        diag_3_group = st.selectbox("Additional diagnosis group", diag_groups)
        diabetes_related = st.selectbox("Diabetes appears anywhere in diagnoses?", ["Yes", "No"])

    st.divider()

    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        payload = {
            "race": race, "gender": gender, "age": age, "weight": weight,
            "time_in_hospital": time_in_hospital, "payer_code": payer_code,
            "admission_type_desc": admission_type_desc,
            "discharge_disposition_desc": discharge_disposition_desc,
            "admission_source_desc": admission_source_desc,
            "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
            "num_medications": num_medications, "number_outpatient": number_outpatient,
            "number_emergency": number_emergency, "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses, "total_prior_visits": total_prior_visits,
            "max_glu_serum": max_glu_serum, "A1Cresult": A1Cresult,
            "metformin": metformin, "repaglinide": repaglinide, "glimepiride": glimepiride,
            "glipizide": glipizide, "glyburide": glyburide, "pioglitazone": pioglitazone,
            "rosiglitazone": rosiglitazone, "insulin": insulin, "change": change,
            "diabetesMed": diabetesMed, "num_meds_changed": num_meds_changed,
            "diag_1_group": diag_1_group, "diag_2_group": diag_2_group,
            "diag_3_group": diag_3_group,
            "diabetes_related": 1 if diabetes_related == "Yes" else 0,
        }

        with st.spinner("Running prediction..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()

                prob = result["readmission_probability"]
                note = result["risk_note"]

                if prob >= 0.5:
                    badge, color = "🔴 High Risk", "red"
                elif prob >= 0.3:
                    badge, color = "🟡 Moderate Risk", "orange"
                else:
                    badge, color = "🟢 Low Risk", "green"

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Readmission Risk", f"{prob:.1%}")
                    st.markdown(f"### :{color}[{badge}]")
                with r2:
                    st.info(note)

            except requests.exceptions.ConnectionError:
                st.error("Could not reach the prediction API. Make sure `main.py` (uvicorn) is running.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# TAB 2: DATA EXPLORER
with tab2:
    st.header("Dataset Explorer")
    st.write("A look at the engineered dataset behind this model.")

    if st.button("Show sample rows"):
        try:
            df_sample = pd.read_csv("../data/diabetic_data_engineered.csv", nrows=10)
            st.dataframe(df_sample)
        except FileNotFoundError:
            st.warning("Dataset file not found at ../data/diabetic_data_engineered.csv")

    st.subheader("Key EDA Findings")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Target imbalance:** only 11.39% of encounters resulted in a <30-day readmission.")
        st.markdown("**Strongest predictor:** `number_inpatient` — readmission rate rises from ~8.6% (0 prior inpatient visits) to ~46% (8 prior visits).")
    with ec2:
        st.markdown("**Dataset size:** 99,343 encounters / 69,990 unique patients after cleaning.")
        st.markdown("**Common diagnoses:** Circulatory conditions are the most frequent primary diagnosis, more common than diabetes itself.")

# TAB 3: MODEL INSIGHTS
with tab3:
    st.header("Model Performance & Cost-Aware Analysis")
    st.write("Six models were trained and compared — not just on standard metrics, but on real-world cost.")

    st.subheader("Evaluation Summary")
    st.caption("Paste your `results_df` output from Block 7 here, or load it from a saved CSV.")
    # Example placeholder — replace with your real results_df, saved via results_df.to_csv()
    # results_df = pd.read_csv("model_results.csv")
    # st.dataframe(results_df)

    st.subheader("Cost-Aware Ranking")
    st.caption("Paste your `cost_df` output from Block 10 here.")
    # cost_df = pd.read_csv("cost_results.csv")
    # st.dataframe(cost_df)

    st.subheader("Threshold Simulation")
    st.caption("This is where your Cost vs. Threshold chart (Block 11) goes — save it as a PNG and display with st.image(), or rebuild it live with matplotlib.")
    # st.image("threshold_curve.png")

# TAB 4: ABOUT
with tab4:
    st.header("About This Project")
    st.markdown("""
    **Dataset:** UCI Diabetes 130-US Hospitals (1999–2008), ~100,000 inpatient diabetes encounters.

    **Goal:** Predict 30-day hospital readmission risk, then go beyond standard ML metrics with a
    cost-aware business layer — since a missed readmission (false negative) is more costly in
    practice than an unnecessary follow-up (false positive).

    **Models compared:** Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest,
    XGBoost, CatBoost — evaluated on precision, recall, F1, ROC-AUC, and total expected cost.

    **Known limitation:** Gaussian Naive Bayes has no built-in class-imbalance handling mechanism,
    unlike the other five models — this is documented, not silently ignored.

    **Risk notes:** Generated live via an LLM (Groq/Llama 3.3), grounded in the model's actual
    prediction and the patient's key features, with a rule-based fallback if the API is unavailable.
    """)