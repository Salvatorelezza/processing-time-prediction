
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load trained pipeline
model = joblib.load('final_model.pkl')

st.title("Processing Time Prediction")

st.write("Please input the following values:")

# Collect all required inputs

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

# WORK_DESCRIPTION (RE-ADDED — required by model)
work_description = st.selectbox("WORK_DESCRIPTION", [
    "NEW BUILDING", 
    "INTERIOR RENOVATION", 
    "DEMOLITION", 
    "ADDITION"
])

reported_cost = st.number_input("REPORTED_COST", min_value=10000, value=2000000)

community_area = st.number_input("Community Areas (number only)", min_value=1, value=77)
community_area_str = f"Community {community_area}"

wd_code = st.number_input("WD_ONEHOTENCODED (number only)", min_value=1, value=12)
wd_code_str = f"WD{wd_code}"

# Date inputs
application_start_date = st.date_input("APPLICATION_START_DATE", value=datetime.today())
issue_date = st.date_input("ISSUE_DATE", value=datetime.today())

# Feature engineering (you must do it manually as done during training)

app_month = application_start_date.month
app_weekday = application_start_date.weekday()
issue_month = issue_date.month
issue_weekday = issue_date.weekday()

app_month_sin = np.sin(2 * np.pi * app_month / 12)
app_month_cos = np.cos(2 * np.pi * app_month / 12)
app_weekday_sin = np.sin(2 * np.pi * app_weekday / 7)
app_weekday_cos = np.cos(2 * np.pi * app_weekday / 7)

issue_month_sin = np.sin(2 * np.pi * issue_month / 12)
issue_month_cos = np.cos(2 * np.pi * issue_month / 12)
issue_weekday_sin = np.sin(2 * np.pi * issue_weekday / 7)
issue_weekday_cos = np.cos(2 * np.pi * issue_weekday / 7)

# Build input dataframe exactly as model expects
input_data = pd.DataFrame({
    'PERMIT_TYPE': [permit_type],
    'REVIEW_TYPE': [review_type],
    'WORK_DESCRIPTION': [work_description],
    'REPORTED_COST': [reported_cost],
    'Community Areas': [community_area_str],
    'WD_ONEHOTENCODED': [wd_code_str],
    'APPLICATION_START_DATE_month_sin': [app_month_sin],
    'APPLICATION_START_DATE_month_cos': [app_month_cos],
    'ISSUE_DATE_month_sin': [issue_month_sin],
    'ISSUE_DATE_month_cos': [issue_month_cos],
    'APPLICATION_START_DATE_weekday_sin': [app_weekday_sin],
    'APPLICATION_START_DATE_weekday_cos': [app_weekday_cos],
    'ISSUE_DATE_weekday_sin': [issue_weekday_sin],
    'ISSUE_DATE_weekday_cos': [issue_weekday_cos]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Processing Time: {prediction[0]:.2f}")
