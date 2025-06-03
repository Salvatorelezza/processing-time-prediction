
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load your trained model (full pipeline)
model = joblib.load('final_model.pkl')

st.title("Processing Time Prediction")

st.write("Please input the following values:")

# Input fields matching your features
permit_type = st.selectbox("PERMIT_TYPE", ["Permit A", "Permit B", "Permit C"])
review_type = st.selectbox("REVIEW_TYPE", ["Review A", "Review B", "Review C"])
work_description = st.selectbox("WORK_DESCRIPTION", ["Work A", "Work B", "Work C"])
reported_cost = st.number_input("REPORTED_COST", min_value=0.0, value=1000.0)
community_area = st.selectbox("Community Areas", ["Community 1", "Community 2", "Community 3"])
wd_code = st.selectbox("WD_ONEHOTENCODED", ["WD1", "WD2", "WD3"])

# Date inputs
application_start_date = st.date_input("APPLICATION_START_DATE", value=datetime.today())
issue_date = st.date_input("ISSUE_DATE", value=datetime.today())

# Feature engineering: same as your training pipeline

# Extract month and weekday from dates
app_month = application_start_date.month
app_weekday = application_start_date.weekday()
issue_month = issue_date.month
issue_weekday = issue_date.weekday()

# Apply sin/cos transformations
app_month_sin = np.sin(2 * np.pi * app_month / 12)
app_month_cos = np.cos(2 * np.pi * app_month / 12)
app_weekday_sin = np.sin(2 * np.pi * app_weekday / 7)
app_weekday_cos = np.cos(2 * np.pi * app_weekday / 7)

issue_month_sin = np.sin(2 * np.pi * issue_month / 12)
issue_month_cos = np.cos(2 * np.pi * issue_month / 12)
issue_weekday_sin = np.sin(2 * np.pi * issue_weekday / 7)
issue_weekday_cos = np.cos(2 * np.pi * issue_weekday / 7)

# Build input dataframe matching model input
input_data = pd.DataFrame({
    'PERMIT_TYPE': [permit_type],
    'REVIEW_TYPE': [review_type],
    'WORK_DESCRIPTION': [work_description],
    'REPORTED_COST': [reported_cost],
    'Community Areas': [community_area],
    'WD_ONEHOTENCODED': [wd_code],
    'APPLICATION_START_DATE_month_sin': [app_month_sin],
    'APPLICATION_START_DATE_month_cos': [app_month_cos],
    'APPLICATION_START_DATE_weekday_sin': [app_weekday_sin],
    'APPLICATION_START_DATE_weekday_cos': [app_weekday_cos],
    'ISSUE_DATE_month_sin': [issue_month_sin],
    'ISSUE_DATE_month_cos': [issue_month_cos],
    'ISSUE_DATE_weekday_sin': [issue_weekday_sin],
    'ISSUE_DATE_weekday_cos': [issue_weekday_cos]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Processing Time: {prediction[0]:.2f}")
