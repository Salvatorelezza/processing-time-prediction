
import streamlit as st

st.write('Hello, *World!* :sunglasses:')
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('final_model.pkl')

st.title("Processing Time Prediction")

st.write("Please input the following values:")

# --- Input fields ---
# Replace these fields with your real features:
# For now, I'm making up some common ones you likely had

reported_cost = st.number_input("REPORTED_COST", min_value=0.0, value=1000.0)

application_start_month_sin = st.slider("APPLICATION_START_DATE month (sin)", -1.0, 1.0, 0.0)
application_start_month_cos = st.slider("APPLICATION_START_DATE month (cos)", -1.0, 1.0, 0.0)
application_start_weekday_sin = st.slider("APPLICATION_START_DATE weekday (sin)", -1.0, 1.0, 0.0)
application_start_weekday_cos = st.slider("APPLICATION_START_DATE weekday (cos)", -1.0, 1.0, 0.0)

issue_month_sin = st.slider("ISSUE_DATE month (sin)", -1.0, 1.0, 0.0)
issue_month_cos = st.slider("ISSUE_DATE month (cos)", -1.0, 1.0, 0.0)
issue_weekday_sin = st.slider("ISSUE_DATE weekday (sin)", -1.0, 1.0, 0.0)
issue_weekday_cos = st.slider("ISSUE_DATE weekday (cos)", -1.0, 1.0, 0.0)

# Example categorical variable (modify as needed):
community_area = st.selectbox("Community Areas", ["Community 1", "Community 2", "Community 3"])
wd_code = st.selectbox("WD_ONEHOTENCODED", ["WD1", "WD2", "WD3"])

# --- Prepare input dataframe ---

input_dict = {
    'REPORTED_COST': [reported_cost],
    'APPLICATION_START_DATE_month_sin': [application_start_month_sin],
    'APPLICATION_START_DATE_month_cos': [application_start_month_cos],
    'APPLICATION_START_DATE_weekday_sin': [application_start_weekday_sin],
    'APPLICATION_START_DATE_weekday_cos': [application_start_weekday_cos],
    'ISSUE_DATE_month_sin': [issue_month_sin],
    'ISSUE_DATE_month_cos': [issue_month_cos],
    'ISSUE_DATE_weekday_sin': [issue_weekday_sin],
    'ISSUE_DATE_weekday_cos': [issue_weekday_cos],
    'Community Areas': [community_area],
    'WD_ONEHOTENCODED': [wd_code]
}

input_df = pd.DataFrame(input_dict)

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Processing Time: {prediction[0]:.2f}")
