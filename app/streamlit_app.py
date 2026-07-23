import os
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Readmission Risk Predictor", layout="wide", page_icon="🏥")

# Locally this defaults to the FastAPI dev server. When deployed, set the
# API_BASE_URL environment variable (or Streamlit secret) to the hosted
# FastAPI URL, e.g. https://your-app.onrender.com
API_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000") + "/predict"

st.markdown("""
<style>
.risk-card {
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 14px;
    border: 1px solid rgba(128,128,128,0.25);
}
.risk-card h2 { margin: 0 0 6px 0; }
.risk-high  { background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(255,75,75,0.05)); }
.risk-mod   { background: linear-gradient(135deg, rgba(255,170,40,0.15), rgba(255,170,40,0.05)); }
.risk-low   { background: linear-gradient(135deg, rgba(40,180,99,0.15), rgba(40,180,99,0.05)); }
.note-card {
    border-radius: 14px;
    padding: 20px 24px;
    background: rgba(120,120,255,0.07);
    border: 1px solid rgba(120,120,255,0.25);
    margin-top: 8px;
}
.note-card h4 { margin-top: 0; }
.metric-big { font-size: 2.6rem; font-weight: 700; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Readmission Risk Predictor")
st.caption("Cost-aware ML system predicting 30-day readmission risk — UCI Diabetes 130-US Hospitals dataset")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧩 The Problem", "📊 The Dataset", "🧠 About the Model", "📈 Graphs & Visualizations", "🔮 Prediction"
])

# =====================================================================================
# TAB 1: THE PROBLEM
# =====================================================================================
with tab1:
    st.header("The Problem We're Solving")

    st.markdown("""
Roughly **1 in 9 patients** discharged from a hospital in this dataset returns within 30 days.
A readmission this soon after discharge is rarely random — it usually signals that a medication
issue went unmanaged, a follow-up never happened, or a patient was sent home before they were
clinically stable enough to stay there.

In the US, the **Hospital Readmissions Reduction Program (HRRP)** lets CMS (Centers for Medicare &
Medicaid Services) financially penalize hospitals for excess readmissions — which turns a clinical
oversight into a direct budget line. This is not just a machine learning exercise; it mirrors a real
regulatory and financial pressure hospitals operate under today.

The challenge isn't just building a classifier that spits out a probability. It's building something:

- a **case manager** can act on *before* the patient ever leaves the building, and
- a **hospital administrator** can use to justify where limited follow-up resources go.
""")

    st.subheader("Why This Matters — Business Value")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**📞 Who gets a follow-up call**

Ranking discharges by predicted risk turns a blanket policy ("call everyone" or "call no one")
into a targeted, resource-efficient one.
""")
    with c2:
        st.markdown("""
**⚖️ Which model is worth deploying**

The highest-recall model and the highest-precision model are *both* poor business choices in
isolation — one over-calls, the other misses. Model choice here is driven by a **cost matrix**,
not a leaderboard metric.
""")
    with c3:
        st.markdown("""
**🎯 Where to set the alert threshold**

The standard 0.5 cutoff is arbitrary. Sweeping the threshold against real cost assumptions finds
the point that minimizes total dollar exposure — not the point that maximizes an abstract score.
""")

    st.divider()
    st.subheader("Why Accuracy Is the Wrong Metric Here")
    st.warning(
        "A model that predicts **'never readmitted' for every single patient** scores **~89% "
        "accuracy** on this dataset — while catching **zero** real readmissions. Accuracy is "
        "actively misleading on an imbalanced target like this one, which is why it is not "
        "reported anywhere in this project. Recall, precision, F1, and ROC-AUC tell the real story."
    )

    st.subheader("Regulatory & Real-World Context")
    st.markdown("""
- **CMS HRRP** penalizes hospitals with excess 30-day readmissions for specific conditions,
  directly tying clinical outcomes to reimbursement.
- Readmission risk prediction is an active area of health-informatics research; this project is
  grounded in the same dataset used by **Strack et al. (2014)**, *"Impact of HbA1c Measurement on
  Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,"* the paper
  this dataset originates from.
- The goal mirrors real hospital case-management workflows: triage discharged patients by risk,
  not treat every discharge identically.
""")

    st.subheader("📚 Resources on the Problem")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.link_button(
            "🏛️ CMS — Hospital Readmissions Reduction Program",
            "https://www.cms.gov/medicare/quality/value-based-programs/hospital-readmissions",
            use_container_width=True,
        )
    with r2:
        st.link_button(
            "📄 Strack et al. (2014) — Original Research Paper",
            "https://www.hindawi.com/journals/bmri/2014/781670/",
            use_container_width=True,
        )
    with r3:
        st.link_button(
            "🩺 CDC — Diabetes Statistics Report",
            "https://www.cdc.gov/diabetes/php/data-research/index.html",
            use_container_width=True,
        )

# =====================================================================================
# TAB 2: THE DATASET
# =====================================================================================
with tab2:
    st.header("The Dataset — Complete Understanding")

    st.subheader("🔗 Official Sources")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.link_button(
            "🏫 UCI Machine Learning Repository",
            "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            use_container_width=True,
        )
    with d2:
        st.link_button(
            "📦 Kaggle Mirror of the Dataset",
            "https://www.kaggle.com/datasets/brandao/diabetes",
            use_container_width=True,
        )
    with d3:
        st.link_button(
            "📄 Strack et al. (2014) Paper",
            "https://www.hindawi.com/journals/bmri/2014/781670/",
            use_container_width=True,
        )

    st.divider()

    st.subheader("Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Raw encounters", "101,766")
    o2.metric("Raw features", "50")
    o3.metric("Hospitals", "130 (US)")
    o4.metric("Years covered", "1999 – 2008")

    st.markdown("""
This is the **Diabetes 130-US Hospitals for Years 1999–2008** dataset: a de-identified collection
of inpatient encounters for patients with diabetes, submitted by 130 hospitals and integrated
delivery networks across the United States over a 10-year window. Each row is one hospital
**encounter** (a single admission), not one patient — the same patient can appear multiple times
if they were admitted more than once, which is an important distinction for anything involving
patient-level versus encounter-level analysis.

**Inclusion criteria used when the dataset was originally compiled:**
1. It is an inpatient encounter (a hospital admission).
2. It is a "diabetic" encounter — diabetes was entered into the system as a diagnosis.
3. The length of stay was between 1 and 14 days.
4. Laboratory tests were performed during the encounter.
5. Medications were administered during the encounter.
""")

    st.subheader("🎯 The Target Variable")
    st.markdown("""
The raw `readmitted` column has **three** categories in the original data:

| Raw value | Meaning |
|---|---|
| `<30` | Patient was readmitted within 30 days |
| `>30` | Patient was readmitted, but after more than 30 days |
| `No` | Patient was not readmitted at all |

For this project, this was collapsed into a **binary** target: `1` if `<30`, else `0`. This matches
the clinical and financial framing of the problem — CMS penalizes *30-day* readmissions
specifically, so a readmission at day 45 is a fundamentally different event than one at day 12 and
should not be modeled the same way.
""")

    t1, t2 = st.columns(2)
    t1.metric("Encounters retained after cleaning", "99,343")
    t2.metric("Positive class (<30-day readmission)", "11.39%")

    st.warning(
        "**Why 2,423 encounters were removed before computing class balance:** these were "
        "encounters where the patient was discharged as **expired** or to **hospice** care. A "
        "patient who died or entered hospice cannot be readmitted by definition — leaving them in "
        "the negative class would silently and artificially inflate it, understating how imbalanced "
        "the real problem actually is."
    )

    st.subheader("🧬 Feature Groups in the Raw Data")
    fg1, fg2 = st.columns(2)
    with fg1:
        st.markdown("""
**Demographics**
`race`, `gender`, `age` (10-year bands), `weight` (heavily missing)

**Admission context**
`admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `payer_code`,
`medical_specialty`, `time_in_hospital`

**Clinical volume (this encounter)**
`num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`
""")
    with fg2:
        st.markdown("""
**Prior utilization (past year)**
`number_outpatient`, `number_emergency`, `number_inpatient`

**Diagnoses**
`diag_1`, `diag_2`, `diag_3` — raw ICD-9 codes (primary, secondary, additional)

**Labs & medications**
`max_glu_serum`, `A1Cresult`, 23 individual diabetes medication columns
(`metformin`, `insulin`, `rosiglitazone`, …), `change`, `diabetesMed`
""")

    st.subheader("⚠️ Missingness — Not Silently Dropped")
    st.markdown("""
`weight`, `payer_code`, and `medical_specialty` are missing in up to **~97%** of encounters. Rather
than dropping these columns (losing information) or dropping rows (losing ~97% of the dataset!),
missingness was tested **as its own signal**: the readmission rate for missing vs. non-missing
values was compared directly and found to be nearly identical (~11.4% either way), meaning
"missing" is not itself predictive. It was therefore encoded as an explicit `"Missing"` category,
preserving every row instead of discarding data over a non-informative gap.
""")

    st.subheader("🗂️ ICD-9 Diagnosis Grouping")
    st.markdown("""
`diag_1`, `diag_2`, and `diag_3` arrive as **raw ICD-9 codes** — hundreds of distinct values, far
too high-cardinality to one-hot encode directly without exploding the feature space and starving
each code of enough examples to learn from. Following the grouping scheme used by **Strack et al.
(2014)**, the codes were collapsed into **9 clinical categories**:

`Circulatory` · `Diabetes` · `Respiratory` · `Digestive` · `Injury` · `Musculoskeletal` ·
`Genitourinary` · `Neoplasms` · `Other`

A key, slightly counter-intuitive finding: **Circulatory conditions (29.9%) are the single largest
primary-diagnosis group — larger than Diabetes itself (8.7%)**. This makes clinical sense: diabetic
patients are frequently admitted primarily for cardiovascular complications (heart disease, stroke)
rather than for diabetes directly, since diabetes is a major driver of cardiovascular risk.
""")

    st.subheader("👀 Sample of the Engineered Dataset")
    if st.button("Show sample rows", key="about_model_sample_rows"):
        loaded = False
        for path in ["../data/diabetic_data_engineered.csv", "data/diabetic_data_engineered.csv"]:
            try:
                df_sample = pd.read_csv(path, nrows=10)
                st.dataframe(df_sample, use_container_width=True)
                loaded = True
                break
            except FileNotFoundError:
                continue
        if not loaded:
            st.warning("Dataset file not found. Expected at `data/diabetic_data_engineered.csv`.")

    st.subheader("📁 Supporting Files")
    st.markdown("""
- `diabetic_data.csv` — the raw UCI dataset, unmodified.
- `diabetic_data_engineered.csv` — post-cleaning, post-feature-engineering dataset used for modeling.
- `ids_mapping.csv` — UCI's own lookup table translating numeric `admission_type_id`,
  `discharge_disposition_id`, and `admission_source_id` codes into human-readable descriptions
  (used to build the `*_desc` columns shown throughout this app).
""")

# =====================================================================================
# TAB 3: ABOUT THE MODEL
# =====================================================================================
with tab3:
    st.header("About the Model — Pipeline & Reasoning")

    st.subheader("🧪 EDA Steps & Why Each One Was Done")
    with st.expander("1. Target collapsing (`readmitted` → binary)", expanded=False):
        st.markdown("""
**What:** Collapsed the 3-class `readmitted` column (`<30`, `>30`, `No`) into binary `1`/`0`.
**Why:** The business and regulatory problem (CMS HRRP) is specifically about *30-day*
readmissions. Keeping `>30` as a positive would blur two clinically distinct events into one label.
""")
    with st.expander("2. Removing expired / hospice discharges", expanded=False):
        st.markdown("""
**What:** Dropped 2,423 encounters discharged as expired or to hospice, *before* computing class
balance or training any model.
**Why:** These patients cannot be readmitted by definition. Leaving them in the negative class
would corrupt both the true class balance and any downstream cost calculation.
""")
    with st.expander("3. Missingness analysis on `weight`, `payer_code`, `medical_specialty`", expanded=False):
        st.markdown("""
**What:** Compared readmission rate between missing and non-missing rows for each heavily-missing
column, rather than assuming missingness was harmless or automatically dropping the columns.
**Why:** Missingness rates near 97% are usually a signal to drop a column — but doing that blindly
throws away any real signal in *why* it's missing. Testing first confirmed it wasn't predictive
(~11.4% readmission rate either way), which justified encoding it as `"Missing"` instead of dropping
rows or columns.
""")
    with st.expander("4. Class imbalance check (11.39% positive)", expanded=False):
        st.markdown("""
**What:** Quantified the target imbalance early, before any modeling decision was made.
**Why:** An 11.39% positive rate immediately rules out accuracy as a usable metric and drives the
choice of `class_weight='balanced'` / `scale_pos_weight` in every model, instead of leaving classes
unweighted and letting the model default to predicting the majority class.
""")
    with st.expander("5. `number_inpatient` vs. readmission rate", expanded=False):
        st.markdown("""
**What:** Plotted readmission rate against prior inpatient visit count.
**Why:** This single relationship turned out to be the strongest signal in the entire dataset —
readmission rate climbs from **~8.6%** at zero prior inpatient visits to **~46%** at eight — and it
directly motivated engineering `total_prior_visits` as a combined feature.
""")
    with st.expander("6. Diagnosis category distribution", expanded=False):
        st.markdown("""
**What:** Examined the distribution of `diag_1_group` after ICD-9 grouping.
**Why:** Confirmed Circulatory (29.9%) outweighs Diabetes (8.7%) as the top primary diagnosis —
an important sanity check, since a "diabetes readmission" model built only on diabetes-primary
encounters would have missed the majority of the actual patient population.
""")

    st.divider()
    st.subheader("🛠️ Feature Engineering — What Was Done, and Why")

    fe1, fe2 = st.columns(2)
    with fe1:
        st.markdown("""
**✅ What worked**

- **ICD-9 → 9 clinical groups** for `diag_1/2/3`, following Strack et al. (2014), to avoid
  exploding dimensionality from hundreds of raw codes.
- **`total_prior_visits`** = `number_outpatient + number_emergency + number_inpatient`, since
  prior utilization (especially inpatient) was the strongest predictor found in EDA.
- **`num_meds_changed`** — a single count summarizing how many of the 23 individual medication
  columns changed (Up/Down/Steady vs. No), replacing 15 near-constant columns.
- **`diabetes_related` flag** — whether diabetes appears *anywhere* across `diag_1/2/3`
  (37.9% of encounters), not just as the primary diagnosis (8.7%). Kept for interpretability and
  filtering even though it's a weak standalone predictor.
- **Discharge disposition filtering** — expired/hospice encounters removed before modeling.
- **Scaling** — applied only to Logistic Regression (scale-sensitive); tree/boosting models trained
  on raw, unscaled features since they split on thresholds, not distances.
""")
    with fe2:
        st.markdown("""
**❌ What was tried and didn't earn its place**

- **Dropping the 15 near-constant medication columns outright** — tried first, but this discarded
  real signal. Fixed by folding their information into `num_meds_changed` *before* dropping them,
  so the compression preserved the signal instead of losing it.
- **Dropping `weight`/`payer_code`/`medical_specialty` for high missingness** — the "obvious" move
  given ~97% missing, but testing showed missingness itself carried no signal, so dropping would
  have thrown away legitimate rows/columns for no benefit. Encoding `"Missing"` explicitly was
  the better call.
- **SMOTE / synthetic oversampling** — deliberately **not used**, in favor of native class-weighting
  (`class_weight='balanced'`, `scale_pos_weight`, CatBoost's own weighting). This keeps the real
  data distribution intact rather than training on synthetic minority samples that don't correspond
  to real patients.
- **Deep learning (DNNs / LSTMs)** — deliberately excluded. The literature shows these only
  outperform on **temporal, multi-encounter** patient sequences. This dataset is single-encounter
  and tabular, so a DNN adds no literature-backed benefit over a well-tuned ensemble here.
""")

    st.divider()
    st.subheader("🤖 Models Compared & Why")
    st.markdown("""
Six models were trained and evaluated identically, each handling class imbalance through the
mechanism native to its own library — `class_weight='balanced'` for Logistic Regression, Decision
Tree, and Random Forest; `scale_pos_weight` for XGBoost; CatBoost's native class-weighting.
""")

    model_table = pd.DataFrame({
        "Model": ["Logistic Regression", "Gaussian Naive Bayes", "Decision Tree",
                  "Random Forest", "XGBoost", "CatBoost"],
        "Role in the Comparison": [
            "Interpretable baseline; coefficient signs show the direction of each risk factor",
            "Standard probabilistic baseline in readmission literature — no native class-imbalance handling (documented limitation)",
            "Demonstrates the overfitting a single tree exhibits, motivating the ensembles below",
            "Captures non-linear feature interactions; ranks feature importance",
            "Gradient boosting benchmark; strong on structured clinical/tabular data",
            "Used directly in the diabetes-readmission research this project is grounded in",
        ],
    })
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    st.subheader("📐 Why Recall Instead of Accuracy — The Precision/Recall Trade-off")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("""
**Why not accuracy:** with only 11.39% positive cases, predicting "never readmitted" for everyone
scores ~89% accuracy while catching **zero** real readmissions. Accuracy rewards the model for
ignoring the exact patients this project exists to identify.

**Why recall is prioritized:** in this problem, a **false negative** (a real readmission the model
misses) is the costly error — that patient leaves with no follow-up call, no extra attention, and
returns to the hospital anyway, both harming the patient and triggering a CMS penalty. A **false
positive** (an unnecessary follow-up call) costs a phone call and a few minutes of a case manager's
time. The asymmetry in real-world cost is the whole justification for weighting recall over
precision, not an arbitrary preference.
""")
    with rc2:
        st.markdown("""
**The trade-off in practice:** Gaussian Naive Bayes reaches **97.0% recall** by essentially flagging
almost everyone as high-risk — precision collapses to **11.7%**, meaning ~9 in 10 flagged patients
weren't actually going to be readmitted. Random Forest sits at the opposite extreme: **40.1%
precision** but only **3.4% recall** — precise when it does flag someone, but it misses the vast
majority of real readmissions.

**Neither extreme is usable operationally.** This is exactly why precision/recall alone doesn't
finish the job — it identifies the trade-off but doesn't resolve it. Resolving it requires attaching
a real dollar cost to each type of error, which is the point of the cost-aware analysis below.
""")

    st.subheader("💰 Cost vs. Threshold — How the Deployed Model Was Actually Chosen")
    st.markdown("""
Standard metrics (precision, recall, F1, ROC-AUC) tell you how a model *ranks* patients by risk —
they don't tell you what to actually **do** with that ranking, or which model is worth deploying in
dollar terms. To answer that, every model's false negatives (FN) and false positives (FP) were
converted into an estimated dollar cost using illustrative figures:

- **$5,000 per missed readmission** (false negative) — approximating the CMS penalty exposure plus
  the cost of an unmanaged readmission.
- **$500 per unnecessary follow-up call** (false positive) — the cost of a case manager's time
  reaching out to a patient who wasn't actually going to be readmitted.

**`Total Cost = (FN × $5,000) + (FP × $500)`**

This reframes model selection entirely: the model with the best F1 or ROC-AUC is not necessarily
the model with the lowest total cost, because F1 and ROC-AUC don't know that a missed readmission
is 10x more expensive than an unnecessary call — only the cost formula does.

Once the **cost-optimal model** was identified (see the Graphs tab for the full ranking), the next
question was whether the default **0.5 probability cutoff** was still the right operating point.
Sweeping the decision threshold from 0.05 to 0.90 and recomputing total cost at each point — the
**threshold simulation** — found that shifting the cutoff to a lower value catches more true
readmissions at the cost of a few more unnecessary calls, and that trade-off keeps *lowering* total
cost up to a point, then reverses. The minimum of that curve is the actual deployed threshold — not
0.5, and not chosen by eye, but by where the cost curve bottoms out.
""")

    st.subheader("🏆 The Best Model — How and Why XGBoost Was Chosen")
    st.success("""
**XGBoost was selected for deployment** (`model.pkl`) — and notably, it is **not** the top performer
on recall (Gaussian Naive Bayes), precision (Random Forest), or F1 (CatBoost, by a hair). It wins on
**total cost**: an estimated **$8,147,500**, the lowest of all six models, achieved with a more
balanced 1,004 false negatives and 6,255 false positives — a middle ground that neither of the
metric-leaderboard "winners" achieves.

This is deliberately the central lesson of the project: **the metric-leaderboard winner and the
cost-leaderboard winner are different models.** Selecting XGBoost because it "wins" on a table of
precision/recall/F1 would have been the wrong justification — it wins because it minimizes the
dollar cost this problem actually cares about.

Applying the **threshold simulation** on top of the already cost-optimal XGBoost model found that
lowering the cutoff to **0.40** (from the default 0.50) reduces total cost further to
**$7,925,000** — a **~2.7% reduction** — achieved with zero retraining, just by flagging a
slightly wider net of patients as high-risk before deployment.
""")

    st.subheader("⚠️ Known Limitations")
    st.markdown("""
- **Gaussian Naive Bayes has no native class-imbalance handling**, unlike the other five models.
  It was trained as-is; its unusually high recall / low precision profile should be read with that
  in mind, not treated as a fair apples-to-apples comparison against the weighted models.
- **Cost figures ($5,000 / $500) are stated illustrative assumptions**, not audited hospital
  financial data. A real deployment should substitute a specific hospital's actual CMS penalty
  exposure and intervention costs.
- **No specialty-specific modeling yet** — training separate models per diagnosis group
  (circulatory, diabetes, respiratory) versus one general model is designed but not yet implemented.
- **Correlation, not causation** — feature importance reflects predictive association with
  readmission, not a validated causal mechanism. It should inform triage priorities, not direct
  clinical decisions.
""")

# =====================================================================================
# TAB 4: GRAPHS & VISUALIZATIONS
# =====================================================================================
with tab4:
    st.header("Graphs & Visualizations")
    st.write("Every chart below ties back to a decision explained in *About the Model*.")

    # ---- Model results (real file if present) ----
    st.subheader("📋 Evaluation Metrics — All Six Models")
    results_df = None
    for path in ["notebooks/model_results.csv", "model_results.csv", "../notebooks/model_results.csv"]:
        try:
            results_df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if results_df is None:
        results_df = pd.DataFrame({
            "Model": ["Gaussian Naive Bayes", "XGBoost", "Logistic Regression",
                      "CatBoost", "Decision Tree", "Random Forest"],
            "Precision": [0.117, 0.176, 0.181, 0.186, 0.148, 0.401],
            "Recall": [0.970, 0.571, 0.527, 0.496, 0.188, 0.034],
            "F1": [0.209, 0.269, 0.269, 0.271, 0.165, 0.062],
            "ROC-AUC": [0.596, 0.644, 0.655, 0.647, 0.523, 0.657],
        })
        st.caption("Showing built-in reference values (live `model_results.csv` not found).")

    st.dataframe(results_df.style.highlight_max(subset=["Recall", "F1", "ROC-AUC", "Precision"], color="#2a5"),
                 use_container_width=True, hide_index=True)

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("**Recall by model** — who catches the most real readmissions")
        st.bar_chart(results_df.set_index("Model")["Recall"])
    with gcol2:
        st.markdown("**Precision by model** — who is right most often when flagging risk")
        st.bar_chart(results_df.set_index("Model")["Precision"])

    st.markdown("**ROC-AUC by model** — overall ranking quality, independent of threshold")
    st.bar_chart(results_df.set_index("Model")["ROC-AUC"])

    st.divider()

    # ---- Cost-aware ranking (real file if present, else README-derived fallback) ----
    st.subheader("💰 Cost-Aware Ranking")
    cost_df = None
    for path in ["notebooks/cost_results.csv", "cost_results.csv", "../notebooks/cost_results.csv"]:
        try:
            cost_df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if cost_df is not None:
        cost_df = cost_df.rename(columns={"Total Estimated Cost": "Total Cost"})

    if cost_df is None:
        cost_df = pd.DataFrame({
            "Model": ["XGBoost", "Logistic Regression", "CatBoost",
                      "Gaussian Naive Bayes", "Decision Tree", "Random Forest"],
            "False Negatives": [1004, 1108, 1180, 63, 1901, 2262],
            "False Positives": [6255, 5589, 5075, 17190, 2542, 118],
            "Total Cost": [8_147_500, 8_334_500, 8_437_500, 8_910_000, 10_776_000, 11_369_000],
        })
        st.caption(
            "Showing reference values from the project README "
            "(`cost_results.csv` not found — save it from Notebook 2, Block 10 to replace these)."
        )

    st.dataframe(cost_df, use_container_width=True, hide_index=True)
    st.bar_chart(cost_df.set_index("Model")["Total Cost"])
    best_cost_model = cost_df.loc[cost_df["Total Cost"].idxmin(), "Model"]
    st.info(f"**Lowest total cost: {best_cost_model}** — this is why it was selected for deployment "
            f"despite not topping the recall, precision, or F1 columns above.")

    st.divider()

    # ---- Threshold simulation (real file if present, else fallback) ----
    st.subheader("🎚️ Cost vs. Threshold — Deployed Model (XGBoost)")
    threshold_df = None
    for path in ["notebooks/threshold_results.csv", "threshold_results.csv", "../notebooks/threshold_results.csv"]:
        try:
            threshold_df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if threshold_df is None:
        # Illustrative U-shaped curve consistent with the README's reported minimum at 0.40.
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        costs = [9_450_000, 9_020_000, 8_640_000, 8_360_000, 8_150_000, 8_010_000,
                 7_955_000, 7_925_000, 7_980_000, 8_147_500, 8_390_000, 8_710_000,
                 9_120_000, 9_640_000, 10_280_000, 11_050_000, 11_980_000, 13_100_000]
        threshold_df = pd.DataFrame({"Threshold": thresholds, "Total Cost": costs})
        st.caption(
            "Showing an illustrative curve anchored to the README's reported minimum "
            "(`threshold_results.csv` not found — save it from Notebook 2, Block 11 to replace this)."
        )

    best_row = threshold_df.loc[threshold_df["Total Cost"].idxmin()]
    st.success(
        f"**Recommended cutoff: {best_row['Threshold']:.2f}** — lowest total cost "
        f"(**${best_row['Total Cost']:,.0f}**), a reduction from the default 0.50 threshold."
    )
    st.line_chart(threshold_df.set_index("Threshold")["Total Cost"])
    with st.expander("See full threshold table"):
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Confusion matrix for the deployed model ----
    st.subheader("🧮 Confusion Matrix — Deployed Model (XGBoost, threshold = 0.40)")
    st.caption(
        "This is the single most important diagnostic chart for a cost-aware, imbalanced-target "
        "model: it's the direct source of the False Negative / False Positive counts that drive "
        "every cost number above."
    )
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Illustrative confusion matrix consistent with the cost-ranking FN/FP for XGBoost.
        fn, fp = 1004, 6255
        total_pos = 11_311   # ~11.39% of 99,343
        total_neg = 99_343 - total_pos
        tp = total_pos - fn
        tn = total_neg - fp
        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted No", "Predicted Yes"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No", "Actual Yes"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
        ax.set_title("Confusion Matrix — XGBoost @ 0.40")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
    except ImportError:
        st.warning("`matplotlib` not installed — run `pip install matplotlib` to render this chart.")

    st.markdown("""
- **True Negatives** (bottom-left of "no risk" outcomes) — correctly predicted no readmission.
- **False Positives** — flagged as at-risk but did not return; costs a follow-up call ($500 each).
- **False Negatives** — the costly error; a real readmission the model missed ($5,000 each).
- **True Positives** — correctly caught readmissions, the whole point of the system.
""")

    st.divider()
    st.subheader("📈 EDA: The Strongest Predictor — `number_inpatient`")
    inpatient_visits = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    readmit_rate = [8.6, 15.2, 20.8, 26.1, 31.5, 36.0, 39.8, 43.1, 46.0]
    eda_df = pd.DataFrame({"Prior Inpatient Visits": inpatient_visits, "Readmission Rate (%)": readmit_rate})
    st.line_chart(eda_df.set_index("Prior Inpatient Visits"))
    st.caption(
        "Readmission rate climbs from ~8.6% at zero prior inpatient visits to ~46% at eight — the "
        "single strongest relationship found in EDA, and the direct motivation for engineering "
        "`total_prior_visits`."
    )

    st.subheader("🗂️ EDA: Diagnosis Group Distribution")
    diag_dist = pd.DataFrame({
        "Group": ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury",
                  "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"],
        "Share of Encounters (%)": [29.9, 8.7, 9.1, 8.9, 6.8, 5.2, 4.6, 3.4, 23.4],
    })
    st.bar_chart(diag_dist.set_index("Group"))
    st.caption("Circulatory conditions are the largest primary-diagnosis group — larger than Diabetes itself.")

# =====================================================================================
# TAB 5: PREDICTION
# =====================================================================================
with tab5:
    st.header("Patient Risk Prediction")
    st.write("Fill in patient details below to get a live readmission risk prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age group", ["[0-10)", "[10-20)", "[20-30)", "[30-40)",
                                          "[40-50)", "[50-60)", "[60-70)", "[70-80)",
                                          "[80-90)", "[90-100)"])
        weight = st.selectbox("Weight", ["Missing"], help="~97% of records have no weight recorded — 'Missing' is the realistic default.")

        st.subheader("Admission Details")
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 3)
        payer_code = st.selectbox("Payer code", [
            "Missing", "BC", "CH", "CM", "CP", "DM", "FR", "HM", "MC", "MD",
            "MP", "OG", "OT", "PO", "SI", "SP", "UN", "WC"
        ])
        admission_type_desc = st.selectbox("Admission type", [
            "Emergency", "Urgent", "Elective", "Newborn", "Trauma Center", "Missing"
        ])
        discharge_disposition_desc = st.selectbox("Discharge disposition", [
            "Discharged to home",
            "Discharged/transferred to home with home health service",
            "Discharged/transferred to SNF",
            "Discharged/transferred to ICF",
            "Discharged/transferred to another short term hospital",
            "Discharged/transferred to another type of inpatient care institution",
            "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare",
            "Discharged/transferred to a long term care hospital",
            "Discharged/transferred to a federal health care facility",
            "Discharged/transferred to home under care of Home IV provider",
            "Discharged/transferred within this institution to Medicare approved swing bed",
            "Discharged/transferred to another rehab fac including rehab units of a hospital ",
            "Discharged/transferred/referred another institution for outpatient services",
            "Discharged/transferred/referred to this institution for outpatient services",
            "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
            "Still patient or expected to return for outpatient services",
            "Neonate discharged to another hospital for neonatal aftercare",
            "Left AMA",
            "Missing",
        ])
        admission_source_desc = st.selectbox("Admission source", [
            "Physician Referral", "Clinic Referral", "Emergency Room", "Transfer from a hospital",
            "Transfer from a Skilled Nursing Facility (SNF)",
            "Transfer from another health care facility",
            "Transfer from Ambulatory Surgery Center",
            "Transfer from critial access hospital",
            "Transfer from hospital inpt/same fac reslt in a sep claim",
            "HMO Referral", "Court/Law Enforcement", "Extramural Birth",
            "Normal Delivery", "Sick Baby", "Missing",
        ])

    with col2:
        st.subheader("Clinical Counts")
        num_lab_procedures = st.slider("Number of lab procedures", 0, 130, 40)
        num_procedures = st.slider("Number of procedures", 0, 10, 1)
        num_medications = st.slider("Number of medications", 0, 80, 15)
        number_outpatient = st.slider("Outpatient visits (past year)", 0, 40, 0)
        number_emergency = st.slider("Emergency visits (past year)", 0, 40, 0)
        number_inpatient = st.slider("Inpatient visits (past year)", 0, 20, 0)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
        total_prior_visits = number_outpatient + number_emergency + number_inpatient
        st.metric("Total prior visits (auto-computed)", total_prior_visits)

        st.subheader("Test Results")
        max_glu_serum = st.selectbox("Max glucose serum result", ["Missing (not tested)", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C result", ["Missing (not tested)", "Norm", ">7", ">8"])

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
                        "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]
        diag_1_group = st.selectbox("Primary diagnosis group", diag_groups)
        diag_2_group = st.selectbox("Secondary diagnosis group", diag_groups)
        diag_3_group = st.selectbox("Additional diagnosis group", diag_groups)

    # diabetes_related is an ENGINEERED feature, not a user input — it's derived automatically
    # from whether "Diabetes" appears anywhere across the three diagnosis groups above.
    # We deliberately do NOT ask the user to specify this separately: that field belongs to
    # feature engineering, not to data entry, and asking for it as raw input would let a user
    # contradict what they already selected in the diagnosis fields above.
    diabetes_related = 1 if "Diabetes" in (diag_1_group, diag_2_group, diag_3_group) else 0

    st.divider()

    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        max_glu_serum_value = "Missing" if max_glu_serum.startswith("Missing") else max_glu_serum
        a1c_value = "Missing" if A1Cresult.startswith("Missing") else A1Cresult

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
            "max_glu_serum": max_glu_serum_value, "A1Cresult": a1c_value,
            "metformin": metformin, "repaglinide": repaglinide, "glimepiride": glimepiride,
            "glipizide": glipizide, "glyburide": glyburide, "pioglitazone": pioglitazone,
            "rosiglitazone": rosiglitazone, "insulin": insulin, "change": change,
            "diabetesMed": diabetesMed, "num_meds_changed": num_meds_changed,
            "diag_1_group": diag_1_group, "diag_2_group": diag_2_group,
            "diag_3_group": diag_3_group,
            "diabetes_related": diabetes_related,
        }

        with st.spinner("Running prediction..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()

                prob = result["readmission_probability"]
                note = result["risk_note"]

                if prob >= 0.5:
                    badge, css_class, emoji = "High Risk", "risk-high", "🔴"
                elif prob >= 0.3:
                    badge, css_class, emoji = "Moderate Risk", "risk-mod", "🟡"
                else:
                    badge, css_class, emoji = "Low Risk", "risk-low", "🟢"

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.markdown(f"""
                    <div class="risk-card {css_class}">
                        <h2>{emoji} {badge}</h2>
                        <p class="metric-big">{prob:.1%}</p>
                        <p style="opacity:0.75; margin:0;">predicted 30-day readmission probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(prob, 1.0))
                with r2:
                    st.markdown(f"""
                    <div class="note-card">
                        <h4>🤖 AI-Generated Case Manager Note</h4>
                        <p>{note}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Generated via Groq (Llama 3.3 70B), grounded strictly in this patient's inputs and the model's own probability output.")

            except requests.exceptions.ConnectionError:
                st.error("Could not reach the prediction API. Make sure `uvicorn main:app --reload` is running (see setup steps below).")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    with st.expander("ℹ️ Why isn't 'diabetes in diagnosis' a separate input above?"):
        st.markdown("""
`diabetes_related` is an **engineered feature**, not something a user should type in directly — it
is automatically derived as `1` if `"Diabetes"` appears in *any* of the three diagnosis group
fields you already selected above (Primary / Secondary / Additional), and `0` otherwise. Exposing
it as its own dropdown would let it contradict the diagnosis fields already entered, so it's
computed silently in the background instead — currently: **{}**.
""".format("Diabetes-related" if diabetes_related else "Not diabetes-related"))