import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import math
import sklearn

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_base64_image("aub_msfea_logo.png")

# Set page config for better layout
st.set_page_config(page_title="Patient Mortality Prediction", page_icon="💡", layout="wide")

# Add custom styling for the button
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 24px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    
    /* Hover effect - Darker red with white text */
    div.stButton > button:hover {
        background-color: #d43f3f;  /* Slightly darker red */
        color: white !important;   /* Ensures text remains white */
        transform: scale(1.05);
    }
            
    div.stButton > button:active {
            background-color: #b53232 !important;
            color: white !important;
    }      
    div.stButton > button:focus {
            background-color: #ff4b4b !important; /* Keep original color */
            color: white !important; /* Keep text white */
        }        
    </style>
""", unsafe_allow_html=True)


# Use columns to arrange the title and the logo side by side
col1, col2 = st.columns([3, 1])  # 3:1 ratio for title and logo


# Title in the first column
col1.title("🏥 Hip Fracture Pre-operative Mortality Prediction")
col1.write("Enter patient details below:")



#col2.image("aub_msfea_logo.png", width=380)
col2.markdown(
    f"""
    <img src="data:image/png;base64,{image_base64}" width="380" style="pointer-events: none; user-select: none;" />
    """,
    unsafe_allow_html=True
)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)



with col1:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1, value=np.nan)
    age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, step=1.0, value=np.nan)
    height = st.number_input("Height (meters)", min_value=1.2, max_value=2.5, step=0.01, value=np.nan)
    prsepis = st.selectbox(
        "Pre-Op Sepsis",
        options=["Select an option", 0, 1, 2, 3],
        format_func=lambda x: {0: "None", 1: "SIRS", 2: "Sepsis", 3: "Septic Shock"}.get(x, "Select an option"),
        index=0,
        help="""Preoperative Sepsis Classification: 
                SIRS: ≥2 criteria (e.g., fever, tachycardia) without infection |
                Sepsis: SIRS + documented infection |
                Septic Shock: Sepsis + hypotension requiring vasopressors"""

    )
    
    fnstatus2 = st.selectbox(
        "Functional Status",
        options=["Select an option", 0, 1, 2],
        format_func=lambda x: {0: "Independent", 1: "Partially Dependent", 2: "Totally Dependent"}.get(x, "Select an option"),
        index=0,
        help="""NSQIP Functional Status:
        Independent: Performs ADLs without assistance |
        Partially Dependent: Requires some help with ADLs |
        Totally Dependent: Completely dependent for care"""
        )
    

with col2:
    prplate = st.number_input("Pre-Op Platelet Count (10³/μL)", min_value=1.0, max_value=1000.0, step=10.0, value=np.nan)
    prhct = st.number_input("Pre-Op Hematocrit (%)", min_value=19.0, max_value=60.0, step=0.1, value=np.nan)
    prsodm = st.number_input("Pre-Op Sodium (mg/dL)", min_value=110.0, max_value=170.0, step=1.0, value=np.nan)
    asaclas = st.selectbox("ASA Classification", options=["Select an option", 1, 2, 3, 4, 5], index=0, 
                           help="""American Society of Anesthesiologists (ASA) Physical Status:
                                1 - Healthy |
                                2 - Mild systemic disease |
                                3 - Severe systemic disease |
                                4 - Severe systemic disease that is a constant threat to life |
                                5 - Moribund, not expected to survive without surgery"""
                            )
    prinr = st.number_input("Pre-Op INR", min_value=0.1, max_value=10.0, step=0.1, value=np.nan, help="Measures blood clotting")

with col3:
    female = st.selectbox(
        "Gender",
        options=["Select an option", 0, 1],
        format_func=lambda x: {0: "Male", 1: "Female"}.get(x, "Select an option"),
        index=0
    )
    prbun = st.number_input("Pre-Op Blood Urea Nitrogen (mg/dL)", min_value=1.0, max_value=200.0, step=1.0, value=np.nan)
    prwbc = st.number_input("Pre-Op White Blood Count (10³/μL)", min_value=0.1, max_value=170.0, step=0.1, value=np.nan)
    cptnew = st.selectbox(
        "CPT Code",
        options=["Select an option", "27236", "27244", "27245" , "27130"],
        format_func=lambda x: f"CPT {x}" if x != "Select an option" else x,
        index=0,
        help="""Common Procedural Terminology (CPT) Codes:
                27236 : Open reduction and internal fixation (ORIF) of a femoral neck fracture |
                27244 : ORIF of inter/peri/subtrochanteric femoral fracture |
                27245 : Intramedullary fixation of inter/peri/sub trochanteric femoral fracture |
                27130 : Total hip arthroplasty"""
                    )
    hxchf = st.selectbox(
        "History of Severe COPD",
        options=["Select an option", 0, 1],
        format_func=lambda x: {0: "No", 1: "Yes"}.get(x, "Select an option"),
        index=0,
        help="Has the patient ever had severe chronic obstructive pulmonary disease (COPD)?"
    )



# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
saved_feature_names = joblib.load("feature_names.pkl")

if st.button("Predict"):
    # Check if any essential fields are missing
    if any(
        (math.isnan(value) if isinstance(value, float) else value == "Select an option")
        for value in [
            weight, age, height, prwbc, fnstatus2, prplate, prhct, prsodm,
            asaclas, prinr, female, prbun, prsepis, cptnew, hxchf
        ]
    ):
        st.error("Please fill in all required fields.")
    else:    
        # Convert user input into a DataFrame
        input_data = pd.DataFrame([[weight, age, height, prwbc, fnstatus2, prplate, prhct, prsodm, asaclas, prinr, female, prbun, prsepis, cptnew, hxchf]],
                                columns=["Weight_kg", "Age_m", "Height_meter", "PRWBC_m", "FNSTATUS2_m",
                                        "PRPLATE_m", "PRHCT_m", "PRSODM_m", "ASACLAS_m", "PRINR_m",
                                        "Female", "PRBUN_m", "PRSEPIS_m", "CPTnew", "HXCHF_m"])

        # Convert categorical variables
        input_data["CPTnew_27236"] = (input_data["CPTnew"] == "27236").astype(int)
        input_data["CPTnew_27245"] = (input_data["CPTnew"] == "27245").astype(int)
        input_data["CPTnew_27244"] = (input_data["CPTnew"] == "27244").astype(int)
        input_data["CPTnew_27130"] = (input_data["CPTnew"] == "27130").astype(int)
        input_data.drop(columns=["CPTnew"], inplace=True)

        # Ensure the input data columns match the order during training
        input_data = input_data.reindex(columns=saved_feature_names, fill_value=0)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Get the prediction probability
        prediction_proba = model.predict_proba(input_scaled)[:, 1][0] * 100

        # Create a styled circular progress bar with a title
        st.markdown(f"""
            <style>
                .title {{
                    text-align: center;
                    font-size: 28px;
                    font-weight: bold;
                    color: #ff4b4b;  /* Vibrant Red */
                    margin-bottom: 10px;
                }}

                .circle-container {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                    flex-direction: column;
                }}

                .progress-circle {{
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    background: conic-gradient(#ff4b4b {prediction_proba}%, #ddd {prediction_proba}% 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 24px;
                    font-weight: bold;
                    color: black;
                    box-shadow: 0px 0px 15px rgba(255, 75, 75, 0.5);
                }}
            </style>
            
            <div class="circle-container">
                <div class="title">Predicted Risk Score</div>
                <div class="progress-circle">{prediction_proba:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)



st.markdown(
    """
    <hr style="margin-top: 40px; margin-bottom: 20px;">

    <div style="font-size: 15px; color: #555; text-align: left; line-height: 1.6;">
        <strong>Disclaimer:</strong><br>
        This tool is intended for educational and research purposes only.<br>
        It should not be used as a substitute for professional medical judgment.
        <br><br>
        <strong>Citation:</strong><br>
        If you use this tool or its outputs in your work, please cite:
        <br>
        <em>
            Fouad Trad, Bassel Isber, Ryan Yammine, Khaled Hatoum, Dana Obeid, Mohamad Chahine, Rachid Haidar, Ghada El-Hajj Fuleihan, Ali Chehab (2025).<br>
            <strong>Parsimonious and Explainable Machine Learning for Predicting Mortality in Patients Post Hip Fracture Surgery.</strong><br>
            <i>Scientific Reports</i>, DOI: <a href="https://doi.org/10.1038/s41598-025-98713-6" target="_blank">10.1038/s41598-025-98713-6</a>
        </em>
    </div>
    """,
    unsafe_allow_html=True
)



    

# To run the app, use the command: streamlit run app.py