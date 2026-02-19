import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime
import base64
import numpy as np
import base64
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="HeartWell",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>

/* -------- FIX ACTIVE TAB STYLE -------- */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #555 !important;
    border: none !important;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background: transparent !important;
    color: #e15757 !important;
    border-bottom: 3px solid #e15757 !important;
}

/* Remove blue focus */
button[data-baseweb="tab"]:focus {
    box-shadow: none !important;
}

/* -------- COLOR SYSTEM -------- */
:root {
    --accent: #d46a6a;
    --sidebar-bg: #fdeaea;
    --soft-border: #f3d4d4;
}

/* -------- REMOVE STREAMLIT TOP STRIP -------- */
header {visibility: hidden;}
.block-container {
    padding-top: 2rem;
}

/* -------- MAIN BACKGROUND -------- */
.stApp {
    background-color: #fff7f7;
    font-family: 'Segoe UI', sans-serif;
}

/* Force ONLY main AI heading */
div[data-testid="stMarkdownContainer"] h2 {
    color: #d46a6a !important;
}
         
/* -------- MAIN TEXT COLOR (ONLY NORMAL TEXT) -------- */
/*.stApp p,
.stApp span,
.stApp div {
    color: #000000;
}*/

/* -------- SIDEBAR -------- */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--soft-border);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #333 !important;
}

/* Sidebar title */
section[data-testid="stSidebar"] h2 {
    color: var(--accent) !important;
    font-weight: 700;
}

/* Sidebar slider color */
.stSlider [data-baseweb="slider"] > div > div:first-child {
    background-color: var(--accent) !important;
}

.stSlider [role="slider"] {
    background-color: var(--accent) !important;
    border: 2px solid white !important;
}
            
/* Remove sidebar dark overlay */
section[data-testid="stSidebar"] {
    background-color: #fdeaea !important;
    border-right: none !important;
    box-shadow: none !important;
}

/* Remove internal overlay layer */
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

/* -------- DROPDOWN -------- */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid var(--soft-border) !important;
    border-radius: 6px !important;
}

/* Remove unwanted sidebar highlight / shadow */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: none !important;
    box-shadow: none !important;
}

div[data-testid="stSidebarNav"] {
    border-right: none !important;
}

/* -------- BUTTONS -------- */
.stButton > button {
    background-color: var(--accent) !important;
    color: white !important;
    border-radius: 8px;
    border: none !important;
    padding: 10px 26px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: #c85757 !important;
    color: white !important;
    transform: translateY(-1px);
}

/* -------- DOWNLOAD BUTTON -------- */
div.stDownloadButton > button {
    background-color: var(--accent);
    color: white;
    border-radius: 6px;
    border: none;
}

/* -------- PROGRESS BAR -------- */
.stProgress > div > div > div > div {
    background-color: var(--accent);
}

/* -------- MATCH TEXT SELECTION -------- */
::selection {
    background: #fdeaea;
    color: #e15757;
}

::-moz-selection {
    background: #fdeaea;
    color: #e15757;
}

/* -------- HR -------- */
hr {
    border: none;
    height: 1px;
    background: var(--soft-border);
}

/* -------- REMOVE FOOTER -------- */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================
artifacts = joblib.load("heart_failure_xgboost_project.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["features"]

# =====================================================
# HEADER WITH LOGO
# =====================================================
def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

data_vidwan_logo_base64 = load_logo("data_vidwan_logo.png")


# =====================================================
# HEADER WITH HEARTWELL + DATA VIDWAN LOGOS
# =====================================================

col1, col2 = st.columns([7, 2])

with col1:
    st.markdown("""
<div style="margin-bottom:5px;">
    <span style="
        font-family: Georgia, serif;
        font-size: 48px;
        font-style: italic;
        color: #d46a6a;
        font-weight: 600;
    ">
        HeartWell
    </span>
</div>
""", unsafe_allow_html=True)


    st.markdown(
        f"""
<div style="display:flex; flex-direction:column; align-items:flex-start;">

<h2 style="color: #d46a6a; margin:0;">
❤️ AI Based Heart Failure Risk Prediction System
</h2>

<p style="margin:5px 0 0 0; color:#000000;">
    Smarter Insights for a Healthier Heart
</p>

<p style="margin:0; color:#000000;">
    Predict • Understand • Prevent
</p>




</div>
""",
        unsafe_allow_html=True
    )

with col2:
    st.image("data_vidwan_logo.png", width=220)

st.markdown(
    '<hr style="margin-top:10px; margin-bottom:10px;">',
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("🩺 Patient Details")

age = st.sidebar.slider("Age", 18, 90, 65)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

bmi = st.sidebar.slider("BMI", 15.0, 40.0, 27.0)
resting_bp = st.sidebar.slider("Resting BP", 90, 180, 130)
cholesterol = st.sidebar.slider("Cholesterol", 120, 350, 240)
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 4.0, 1.5)

fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar", ["yes", "no", "Unknown"])
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["yes", "no", "Unknown"])
diabetes = st.sidebar.selectbox("Diabetes", ["yes", "no", "Unknown"])

num_major_vessels = st.sidebar.selectbox("Major Vessels", [0, 1, 2, 3, "Unknown"])

chest_pain_type = st.sidebar.selectbox(
    "Chest Pain Type", ["typical", "atypical", "non-anginal", "asymptomatic", "Unknown"]
)
rest_ecg = st.sidebar.selectbox(
    "Rest ECG", ["normal", "ST-T abnormality", "LV hypertrophy", "Unknown"]
)
slope = st.sidebar.selectbox(
    "Slope", ["upsloping", "flat", "downsloping", "Unknown"]
)
thalassemia = st.sidebar.selectbox(
    "Thalassemia", ["normal", "fixed defect", "reversible defect", "Unknown"]
)
smoking_status = st.sidebar.selectbox(
    "Smoking Status", ["never", "former", "current"]
)

# =====================================================
# RAW INPUT SUMMARY
# =====================================================
raw_input_summary = {
    "Age": age,
    "Sex": sex,
    "BMI": bmi,
    "Resting BP": resting_bp,
    "Cholesterol": cholesterol,
    "Oldpeak": oldpeak,
    "Fasting Blood Sugar": fasting_blood_sugar,
    "Exercise Angina": exercise_angina,
    "Diabetes": diabetes,
    "Major Vessels": num_major_vessels,
    "Chest Pain Type": chest_pain_type,
    "Rest ECG": rest_ecg,
    "Slope": slope,
    "Thalassemia": thalassemia,
    "Smoking Status": smoking_status
}

# =====================================================
# HELPERS
# =====================================================
def binary_or_nan(val):
    if val == "yes":
        return 1
    if val == "no":
        return 0
    return np.nan

def cat_or_nan(val):
    return val.lower() if val != "Unknown" else np.nan

# =====================================================
# INPUT PREPROCESSING
# =====================================================
input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "Male" else 0],
    "resting_bp": [resting_bp],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [binary_or_nan(fasting_blood_sugar)],
    "exercise_angina": [binary_or_nan(exercise_angina)],
    "oldpeak": [oldpeak],
    "num_major_vessels": [float(num_major_vessels) if num_major_vessels != "Unknown" else np.nan],
    "bmi": [bmi],
    "diabetes": [binary_or_nan(diabetes)],
    "chest_pain_type": [cat_or_nan(chest_pain_type)],
    "rest_ecg": [cat_or_nan(rest_ecg)],
    "slope": [cat_or_nan(slope)],
    "thalassemia": [cat_or_nan(thalassemia)],
    "smoking_status": [smoking_status]
})

input_data = pd.get_dummies(
    input_data,
    columns=["chest_pain_type", "rest_ecg", "slope", "thalassemia", "smoking_status"],
    drop_first=False
)

input_data = input_data.reindex(columns=feature_columns, fill_value=0)  
input_data[scaler.feature_names_in_] = scaler.transform(
    input_data[scaler.feature_names_in_]
)



# =====================================================
# EXPLANATION LOGIC
# =====================================================
reasons, tips = [], []

if age > 60:
    reasons.append("Advanced age increases cardiac workload.")
    tips.append("Schedule regular cardiac evaluations.")

if bmi > 24.9:
    reasons.append("High BMI indicates obesity, increasing heart strain.")
    tips.append("Adopt gradual weight reduction through diet and exercise.")

if cholesterol > 200:
    reasons.append("Elevated cholesterol can lead to arterial blockage.")
    tips.append("Reduce saturated fats and monitor lipid levels.")

if resting_bp > 120:
    reasons.append("High blood pressure stresses heart muscles.")
    tips.append("Limit salt intake and monitor BP regularly.")

if exercise_angina == "yes":
    reasons.append("Exercise-induced chest pain indicates reduced blood supply to the heart.")
    tips.append("Avoid strenuous activity and seek cardiac evaluation.")

if fasting_blood_sugar == "yes":
    reasons.append("Elevated fasting blood sugar increases cardiovascular risk.")
    tips.append("Maintain healthy diet and monitor blood sugar levels regularly.")

if oldpeak > 1.0:
    reasons.append("High ST depression during exercise indicates abnormal cardiac stress response.")
    tips.append("Limit physical exertion and consult a cardiologist.")

if num_major_vessels == 1:
    reasons.append("Blockage in one major coronary vessel indicates early coronary artery disease.")
    tips.append("Adopt heart-healthy habits and schedule regular cardiac checkups.")

if num_major_vessels == 2:
    reasons.append("Blockage in two major coronary vessels suggests moderate coronary artery disease.")
    tips.append("Strict lifestyle modification and regular cardiology follow-up are recommended.")

if num_major_vessels == 3:
    reasons.append("Blockage in three major coronary vessels indicates severe coronary artery disease.")
    tips.append("Immediate cardiology consultation and advanced medical management are advised.")

if diabetes == "yes":
    reasons.append("Diabetes accelerates cardiovascular damage.")
    tips.append("Maintain strict blood sugar control.")

if chest_pain_type == "typical":
    reasons.append("Typical chest pain is commonly associated with heart-related ischemia.")
    tips.append("Immediate cardiac evaluation is recommended.")

if chest_pain_type == "atypical":
    reasons.append("Atypical chest pain may still indicate underlying cardiac issues.")
    tips.append("Further diagnostic tests are advised to rule out heart disease.")

if chest_pain_type == "non-anginal":
    reasons.append("Non-anginal chest pain is usually not related to heart disease.")
    tips.append("Continue monitoring symptoms and maintain a healthy lifestyle.")

if chest_pain_type == "asymptomatic":
    reasons.append("Absence of chest pain can sometimes mask underlying cardiac conditions.")
    tips.append("Regular cardiac screening is recommended.")

if rest_ecg == "ST-T abnormality":
    reasons.append("ST-T wave abnormalities suggest possible myocardial ischemia.")
    tips.append("Further ECG monitoring and cardiac evaluation are recommended.")

if rest_ecg == "LV hypertrophy":
    reasons.append("Left ventricular hypertrophy indicates increased cardiac workload.")
    tips.append("Blood pressure control and cardiac follow-up are advised.")

if slope == "flat":
    reasons.append("Flat ST segment during exercise indicates moderate cardiac risk.")
    tips.append("Regular cardiovascular monitoring is recommended.")

if slope == "downsloping":
    reasons.append("Downsloping ST segment strongly suggests myocardial ischemia.")
    tips.append("Immediate cardiology consultation is advised.")

if thalassemia == "fixed defect":
    reasons.append("Fixed defect indicates permanent impairment in blood flow.")
    tips.append("Long-term cardiac monitoring is recommended.")

if thalassemia == "reversible defect":
    reasons.append("Reversible defect suggests temporary blood flow reduction during stress.")
    tips.append("Stress management and further cardiac testing are advised.")

if smoking_status == "former":
    reasons.append("Past smoking history contributes to residual cardiovascular risk.")
    tips.append("Continue avoiding smoking and maintain heart-healthy habits.")

if smoking_status == "current":
    reasons.append("Active smoking damages blood vessels and increases heart failure risk.")
    tips.append("Immediate smoking cessation is strongly recommended.")

# =====================================================
# PDF GENERATION 
# =====================================================

def generate_dynamic_explanations(raw_inputs):
    explanations = []
    recommendations = []

    age = raw_inputs.get("Age")
    if isinstance(age, (int, float)) and age > 65:
        explanations.append(
            f"Advanced age ({age} years) is associated with increased cardiovascular risk "
            "due to reduced arterial elasticity and cumulative exposure to risk factors."
        )
        recommendations.append(
            "Regular cardiac evaluations and age-appropriate health monitoring are advised."
        )

    bmi = raw_inputs.get("BMI")
    if isinstance(bmi, (int, float)):
        if bmi > 24.9:
            explanations.append(
                f"Body Mass Index (BMI) of {bmi} indicates obesity, increasing cardiac workload."
            )
            recommendations.append(
                "Gradual weight reduction through diet control and regular physical activity is recommended."
            )

    bp = raw_inputs.get("Resting BP")
    if isinstance(bp, (int, float)) and bp > 120:
        explanations.append(
            f"Resting blood pressure of {bp} mmHg is above the normal range, increasing strain on the heart."
        )
        recommendations.append(
            "Blood pressure should be controlled through medication, reduced salt intake, and stress management."
        )

    chol = raw_inputs.get("Cholesterol")
    if isinstance(chol, (int, float)) and chol > 200:
        explanations.append(
            f"Cholesterol level of {chol} mg/dL exceeds the recommended limit, increasing plaque formation risk."
        )
        recommendations.append(
            "Dietary fat reduction and lipid profile monitoring are advised."
        )

    if raw_inputs.get("Exercise Angina") == "yes":
        explanations.append(
            "Chest pain during exercise indicates reduced oxygen supply to the heart."
        )
        recommendations.append(
            "Avoid strenuous activity and seek medical evaluation."
        )

    if raw_inputs.get("Fasting Blood Sugar") == "yes":
        explanations.append(
            "Elevated fasting blood sugar increases long-term cardiovascular risk."
        )
        recommendations.append(
            "Dietary control and regular glucose monitoring are advised."
        )

    oldpeak_val = raw_inputs.get("Oldpeak")
    if isinstance(oldpeak_val, (int, float)) and oldpeak_val > 1.0:
        explanations.append(
            "Significant ST depression during exercise suggests abnormal cardiac stress response."
        )
        recommendations.append(
            "Limit physical exertion and consult a cardiologist."
        )

    vessels = raw_inputs.get("Major Vessels")

    if raw_inputs.get("num_major_vessels") == 1:
        explanations.append(
            "Blockage detected in one major coronary vessel, indicating early-stage coronary artery disease."
        )
        recommendations.append(
            "Lifestyle modification and periodic cardiology evaluation are recommended."
        )

    if raw_inputs.get("num_major_vessels") == 2:
        explanations.append(
            "Blockage detected in two major coronary vessels, suggesting moderate coronary artery disease."
        )
        recommendations.append(
            "Close cardiac monitoring and strict risk factor control are advised."
        )

    if raw_inputs.get("num_major_vessels") == 3:
        explanations.append(
            "Blockage detected in three major coronary vessels, indicating severe coronary artery disease."
        )
        recommendations.append(
            "Immediate cardiology consultation and advanced medical intervention are recommended."
        )

    if raw_inputs.get("Diabetes") == "yes":
        explanations.append(
            "Presence of diabetes accelerates cardiovascular damage and increases heart failure risk."
        )
        recommendations.append(
            "Strict blood glucose control and periodic diabetic evaluation are essential."
        )

    if raw_inputs.get("Chest Pain Type") == "typical":
        explanations.append(
            "Typical chest pain is strongly associated with reduced blood flow to the heart."
        )
        recommendations.append(
            "Immediate cardiac evaluation and diagnostic testing are advised."
        )

    if raw_inputs.get("Chest Pain Type") == "atypical":
        explanations.append(
            "Atypical chest pain may still indicate underlying cardiac abnormalities."
        )
        recommendations.append(
            "Further clinical assessment is recommended to rule out heart disease."
        )

    if raw_inputs.get("Chest Pain Type") == "asymptomatic":
        explanations.append(
            "Absence of chest pain does not always indicate absence of heart disease."
        )
        recommendations.append(
            "Routine cardiac screening is recommended."
        )

    if raw_inputs.get("Rest ECG") == "ST-T abnormality":
        explanations.append(
            "ST-T wave abnormalities suggest possible myocardial ischemia."
        )
        recommendations.append(
            "Further ECG evaluation and cardiac monitoring are advised."
        )

    if raw_inputs.get("Rest ECG") == "LV hypertrophy":
        explanations.append(
            "Left ventricular hypertrophy indicates chronic pressure overload on the heart."
        )
        recommendations.append(
            "Blood pressure management and cardiac follow-up are recommended."
        )

    if raw_inputs.get("Slope") == "flat":
        explanations.append(
            "Flat ST segment during exercise indicates moderate ischemic risk."
        )
        recommendations.append(
            "Regular cardiovascular monitoring is advised."
        )

    if raw_inputs.get("Slope") == "downsloping":
        explanations.append(
            "Downsloping ST segment strongly suggests myocardial ischemia."
        )
        recommendations.append(
            "Immediate cardiology consultation is recommended."
        )

    

    if raw_inputs.get("Thalassemia") == "fixed defect":
        explanations.append(
            "Fixed defect indicates permanent reduction in blood flow."
        )
        recommendations.append(
            "Long-term cardiac monitoring is advised."
        )

    if raw_inputs.get("Thalassemia") == "reversible defect":
        explanations.append(
            "Reversible defect suggests temporary ischemia during physical stress."
        )
        recommendations.append(
            "Stress management and further cardiac evaluation are recommended."
        )

    if raw_inputs.get("Smoking Status") == "former":
        explanations.append(
        "Previous smoking history contributes to residual cardiovascular risk."
        )
        recommendations.append(
            "Continued abstinence from smoking is strongly advised."
        )

    if raw_inputs.get("Smoking Status") == "current":
        explanations.append(
            "Active smoking causes vascular damage and increases heart failure risk."
        )
        recommendations.append(
            "Complete smoking cessation is strongly recommended."
        )


    if not explanations:
        explanations.append(
            "All evaluated clinical parameters are within acceptable ranges, indicating lower cardiovascular risk."
        )
        recommendations.append(
            "Continue maintaining a healthy lifestyle with routine medical checkups."
        )

    return explanations, recommendations


class PDFWithPageNumbers(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "–": "-", "—": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "≤": "<=", "≥": ">=",
        "°": " deg", "μ": "u"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_pdf(prob, risk_label, raw_inputs, explanations, recommendations):
    pdf = PDFWithPageNumbers()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ================= HEADER =================
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "REPORT", ln=True, align="C")

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "AI Based Heart Failure Risk Assessment", ln=True, align="C")

    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ================= PATIENT & REPORT SUMMARY =================
    pdf.set_font("Arial", size=11)

    label_w, value_w = 35, 40
    age_val = clean_text(raw_inputs.get("Age", "-"))
    sex_val = clean_text(raw_inputs.get("Sex", "-"))
    report_date = datetime.now().strftime('%d-%m-%Y')
    report_time = datetime.now().strftime('%H:%M:%S')

    pdf.cell(label_w, 7, "Age")
    pdf.cell(value_w, 7, f": {age_val} Years")
    pdf.cell(label_w, 7, "Sex")
    pdf.cell(0, 7, f": {sex_val}", ln=True)

    pdf.cell(label_w, 7, "Report Date")
    pdf.cell(value_w, 7, f": {report_date}")
    pdf.cell(label_w, 7, "Risk Level")
    pdf.cell(0, 7, f": {risk_label}", ln=True)

    pdf.cell(label_w, 7, "Report Time")
    pdf.cell(value_w, 7, f": {report_time}")
    pdf.cell(label_w, 7, "Risk Probability")
    pdf.cell(0, 7, f": {prob:.2f}%", ln=True)

    pdf.ln(6)

    # ================= RESULTS TABLE =================
    pdf.set_font("Arial", "B", 11)
    pdf.cell(70, 8, "TEST / PARAMETER")
    pdf.cell(50, 8, "RESULT", align="C")
    pdf.cell(0, 8, "REFERENCE RANGE", align="C", ln=True)

    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    normal_ranges = {
        "BMI": "18.5 - 24.9",
        "Resting BP": "90 - 120 mmHg",
        "Cholesterol": "< 200 mg/dL",
        "Oldpeak": "0.0 - 1.0",
        "Major Vessels": "0"
    }

    numeric_tests = normal_ranges.keys()

    pdf.set_font("Arial", size=10)

    for key in numeric_tests:
        value = raw_inputs.get(key, "-")
        underline = False
        color = (0, 0, 0)

        try:
            val = float(value)
            if (
                (key == "BMI" and val > 24.9) or
                (key == "Cholesterol" and val > 200) or
                (key == "Resting BP" and val > 120) or
                (key == "Oldpeak" and val > 1) or
                (key == "Major Vessels" and val > 0)
            ):
                color = (220, 0, 0)
                underline = True
        except:
            pass

        pdf.set_text_color(0, 0, 0)
        pdf.cell(70, 8, key)

        pdf.set_text_color(*color)
        pdf.set_font("Arial", "U" if underline else "")
        pdf.cell(50, 8, clean_text(value), align="C")

        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, normal_ranges[key], align="C", ln=True)

    # ================= CLINICAL OBSERVATIONS =================
    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Clinical Observations", ln=True)

    pdf.set_font("Arial", size=10)
    for key, value in raw_inputs.items():
        if key not in numeric_tests and key not in ["Age", "Sex"]:
            pdf.cell(60, 7, key)
            pdf.cell(0, 7, f": {clean_text(value)}", ln=True)

    # ================= DYNAMIC INTERPRETATION =================
     # ================= PAGE 2 =================
    pdf.add_page()
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Clinical Interpretation", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, " ".join(explanations))

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Lifestyle Recommendations", ln=True)

    pdf.set_font("Arial", size=10)
    for rec in recommendations:
        pdf.multi_cell(0, 6, f"- {rec}")
     
  
    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Follow-up Advice", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(
        0, 6,
        "Consult a qualified cardiologist for further evaluation. "
        "Regular monitoring and adherence to medical advice are essential."
    )

    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0, 5,
        "This report is generated using an AI-based decision support system "
        "and is intended for informational purposes only. It does not replace "
        "professional medical diagnosis."
    )

    pdf.cell(0, 6, "Generated using Streamlit & XGBoost | Data Vidwan", align="C")

    file_name = "Heart_Failure_Lab_Report.pdf"
    pdf.output(file_name)
    return file_name


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediction",
    "📌 Reason & Prevention",
    "📋 Input Summary",
    "📘 Feature Explanation & Ranges"
])

# =====================================================
# TAB 1 — PREDICTION
# =====================================================
with tab1:

    if st.button("🚀 Predict Heart Failure Risk"):

        # ---------------- Prediction ----------------
        prob = model.predict_proba(
            input_data,
            validate_features=False
        )[0][1] * 100

        pred = model.predict(
            input_data,
            validate_features=False
        )[0]

        # ---------------- Header ----------------
        st.subheader("🎯 Risk Assessment")
        st.progress(int(prob))

                # ---------------- Gauge ----------------
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob,
                number={
                    "suffix": "%",
                    "font": {
                        "color": "#2c2c2c",
                        "size": 75
                    }
                },
                domain={
                    "x": [0.25, 0.75],   # THIS controls width
                    "y": [0, 1]
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#2c2c2c",
                        "tickwidth": 1
                    },
                    "bar": {
                        "color": "#e15757",
                        "thickness": 0.3
                    },
                    "steps": [
                        {"range": [0, 30], "color": "#2ecc71"},
                        {"range": [30, 60], "color": "#f1c40f"},
                        {"range": [60, 100], "color": "#e74c3c"},
                    ],
                }
            )
        )

        gauge.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=0),
            paper_bgcolor="#fff7f7",
            plot_bgcolor="#fff7f7",
            font={"color": "#2c2c2c"}
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


        # ---------------- Risk Label ----------------
        if pred == 0:
            st.success("✅ LOW RISK")
            risk_label = "Low Risk"
        else:
            st.error("⚠️ HIGH RISK")
            risk_label = "High Risk"

        # ---------------- Dynamic Explanation ----------------
        dynamic_explanations, dynamic_recommendations = generate_dynamic_explanations(
            raw_input_summary
        )

        st.session_state["dynamic_explanations"] = dynamic_explanations
        st.session_state["dynamic_recommendations"] = dynamic_recommendations

        # ---------------- PDF Generation ----------------
        pdf_file = generate_pdf(
            prob,
            risk_label,
            raw_input_summary,
            dynamic_explanations,
            dynamic_recommendations
        )

        # ---------------- Download Button ----------------
        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download Medical Report (PDF)",
                f,
                file_name=pdf_file,
                mime="application/pdf"
            )

# =====================================================
# TAB 2 — EXPLANATION
# =====================================================
with tab2:
    st.subheader("📌 Reason")

    if "dynamic_explanations" in st.session_state:
        for r in st.session_state["dynamic_explanations"]:
            st.write("🔴", r)
    else:
        st.info("Run prediction to see clinical explanations.")

    st.subheader("✅ Personalized Recommendations")

    if "dynamic_recommendations" in st.session_state:
        for t in st.session_state["dynamic_recommendations"]:
            st.write("🟢", t)
    else:
        st.info("Run prediction to see recommendations.")


# =====================================================
# TAB 3 — INPUT SUMMARY (FINAL FIX)
# =====================================================
with tab3:
    st.subheader("📋 Patient Input Summary")

    html_rows = ""
    for key, value in raw_input_summary.items():
        html_rows += (
            "<tr>"
            f"<td style='padding:10px; border:1px solid #ddd; font-weight:600;'>{key}</td>"
            f"<td style='padding:10px; border:1px solid #ddd;'>{value}</td>"
            "</tr>"
        )

    html_table = (
        "<table style='border-collapse:collapse; width:100%; margin-top:10px;'>"
        "<thead>"
        "<tr style='background-color:#f5f5f5;'>"
        "<th style='padding:10px; border:1px solid #ddd; text-align:left;'>Feature</th>"
        "<th style='padding:10px; border:1px solid #ddd; text-align:left;'>Entered Value</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        f"{html_rows}"
        "</tbody>"
        "</table>"
    )

    st.markdown(html_table, unsafe_allow_html=True)




# =====================================================
# TAB 4 — FEATURE GUIDE
# =====================================================

with tab4:
    st.subheader("Feature Definitions, Medical Meaning & Normal Ranges")

    st.markdown("""
**Age:**  
Represents the patient's age in years. Cardiovascular risk increases with age due to reduced arterial elasticity and cumulative exposure to risk factors.  
*Normal:* 18 - 65 years  

**BMI (Body Mass Index):**  
Measures body fat based on height and weight. Higher BMI increases cardiac workload and is associated with hypertension and diabetes.  
*Normal:* 18.5 - 24.9  

**Resting Blood Pressure:**  
Pressure exerted by blood on arterial walls at rest. Persistent elevation damages heart muscles and blood vessels.  

*Normal:* < 120 mmHg  

*High blood pressure:* ≥ 140 mm Hg

**Cholesterol:**  
Total serum cholesterol level. High cholesterol leads to plaque formation and arterial blockage.  

*Normal:* < 200 mg/dL 

*Borderline high:* 200 - 239 mg/dL

*High:* ≥ 240 mg/dL 

**Oldpeak:**  
ST depression induced by exercise. Higher values indicate abnormal heart response under stress.  
*Normal:* 0 - 1  

**Major Vessels:**  
Number of major coronary vessels showing blockage. Higher values indicate severe coronary disease.  
*Normal:* 0  

**Diabetes / Fasting Blood Sugar:**  
Indicates abnormal glucose metabolism. Diabetes accelerates cardiovascular damage.  
*Normal:* No  

**Exercise Angina:**  
Chest pain during physical exertion, indicating insufficient blood supply to the heart.  
*Normal:* No  

**Chest Pain Type:**  
Categorizes the nature of chest pain, reflecting different cardiac conditions. 
                 
* Typical:
Classic chest pain associated with heart-related causes.

* Atypical:
Chest pain with non-classical symptoms.

* Non-anginal:
Chest pain not related to heart disease.

* Asymptomatic:
No chest pain experienced.
                                
*Normal:* Non-anginal  

**Rest ECG:**  
Electrical activity of the heart at rest. Abnormalities indicate structural or rhythm issues.  

* Normal:
Normal electrical heart activity.

* ST-T abnormality:
Abnormal ST-T wave patterns, indicating possible ischemia.

* LV hypertrophy:
Thickening of the left ventricle, often due to long-term high blood pressure.
                
*Normal:* Normal ECG  

**Slope:**  
Slope of ST segment during exercise. Downsloping indicates higher ischemic risk.  

* Upsloping:
Gradual upward ST segment, generally considered normal.

* Flat:
No significant slope change, indicating moderate concern.

* Downsloping:
Downward slope, strongly associated with cardiac ischemia.
                
*Normal:* Upsloping  

**Thalassemia:**  
Blood disorder affecting oxygen transport and cardiac workload.  

* Normal:
No thalassemia detected.

* Fixed defect:
Permanent blood flow abnormality.

* Reversible defect:
Temporary blood flow abnormality during stress.
                
*Normal:* Normal  

**Smoking Status:**  
Smoking damages blood vessels and increases cardiovascular risk.  
*Normal:* Never
""")

st.markdown("---")

st.markdown(
    '<p style="color:#000000; text-align:center; font-size:14px;">'
    'Built with ❤️ using Streamlit & XGBoost | Data Vidwan'
    '</p>',
    unsafe_allow_html=True
)
