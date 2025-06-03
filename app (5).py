
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load your trained model (full pipeline)
model = joblib.load('final_model.pkl')

st.title("Processing Time Prediction")

st.write("Please input the following values:")

permit_type = st.selectbox("PERMIT_TYPE", [
    "PERMIT - NEW CONSTRUCTION", 
    "PERMIT - RENOVATION/ALTERATION", 
    "PERMIT - WRECKING/DEMOLITION"
])

review_type = st.selectbox("REVIEW_TYPE", [
    "DIRECT DEVELOPER SERVICES", 
    "EASY PERMIT", 
    "STANDARD PLAN REVIEW", 
    "DEMOLITION PERMIT"
])

reported_cost = st.number_input("REPORTED_COST", min_value=10000, value=2000000)

community_area = st.number_input("Community Areas (number only)", min_value=1, value=77)
community_area_str = f"Community {community_area}"

wd_code = st.number_input("WD_ONEHOTENCODED (number only)", min_value=1, value=12)
wd_code_str = f"WD{wd_code}"

# Date inputs (raw dates, no sin/cos applied)
application_start_date = st.date_input("APPLICATION_START_DATE", value=datetime.today())
issue_date = st.date_input("ISSUE_DATE", value=datetime.today())

# Build input dataframe matching model input
input_data = pd.DataFrame({
    'PERMIT_TYPE': [permit_type],
    'REVIEW_TYPE': [review_type],
    'REPORTED_COST': [reported_cost],
    'Community Areas': [community_area_str],
    'WD_ONEHOTENCODED': [wd_code_str],
    'APPLICATION_START_DATE': [pd.to_datetime(application_start_date)],
    'ISSUE_DATE': [pd.to_datetime(issue_date)]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Processing Time: {prediction[0]:.2f}")
