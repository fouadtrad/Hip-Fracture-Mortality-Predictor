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
col1, col2 = st.columns([4, 1])  # 3:1 ratio for title and logo

# Title in the first column
col1.title("🏥 Hip Fracture Pre-operative Mortality Prediction")
col1.write("Enter patient details below:")

# Logo in the second column
#col2.image("AUB.jpg", width=300)


# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1, value=np.nan)
    age = st.number_input("Age", min_value=18.0, max_value=100.0, step=1.0, value=np.nan)
    height = st.number_input("Height (meters)", min_value=1.2, max_value=2.5, step=0.01, value=np.nan)
    prwbc = st.number_input("Pre-Op White Blood Count (WBC)", min_value=1.0, max_value=20.0, step=0.1, value=np.nan)
    fnstatus2 = st.selectbox(
        "Functional Status",
        options=["Select an option", 0, 1, 2],
        format_func=lambda x: {0: "Independent", 1: "Partially Dependent", 2: "Totally Dependent"}.get(x, "Select an option"),
        index=0
    )

with col2:
    prplate = st.number_input("Pre-Op Platelet Count", min_value=10.0, max_value=1000.0, step=10.0, value=np.nan)
    prhct = st.number_input("Pre-Op Hematocrit", min_value=10.0, max_value=50.0, step=0.1, value=np.nan)
    prsodm = st.number_input("Pre-Op Sodium", min_value=120.0, max_value=160.0, step=1.0, value=np.nan)
    asaclas = st.selectbox("ASA Classification", options=["Select an option", 1, 2, 3, 4, 5], index=0)
    prinr = st.number_input("Pre-Op INR", min_value=0.5, max_value=5.0, step=0.1, value=np.nan)

with col3:
    female = st.selectbox(
        "Gender",
        options=["Select an option", 0, 1],
        format_func=lambda x: {0: "Male", 1: "Female"}.get(x, "Select an option"),
        index=0
    )
    prbun = st.number_input("Pre-Op BUN", min_value=5.0, max_value=100.0, step=1.0, value=np.nan)
    prsepis = st.selectbox(
        "Pre-Op Sepsis",
        options=["Select an option", 0, 1, 2, 3],
        format_func=lambda x: {0: "None", 1: "SIRS", 2: "Sepsis", 3: "Septic Shock"}.get(x, "Select an option"),
        index=0
    )
    cptnew = st.selectbox(
        "CPT Code",
        options=["Select an option", "27236", "27245", "27244", "27130"],
        format_func=lambda x: f"CPT {x}" if x != "Select an option" else x,
        index=0
    )
    hxchf = st.selectbox(
        "History of CHF",
        options=["Select an option", 0, 1],
        format_func=lambda x: {0: "No", 1: "Yes"}.get(x, "Select an option"),
        index=0
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


    

# To run the app, use the command: streamlit run app.py