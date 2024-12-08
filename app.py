import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

def preprocess_and_predict(input_data, model, scaler, encoder):
    numeric_features = [
        "Curricular_units_2nd_sem_approved", 
        "Curricular_units_2nd_sem_grade", 
        "Curricular_units_1st_sem_approved", 
        "Curricular_units_1st_sem_grade", 
        "Age_at_enrollment"
    ]
    categorical_features = [
        "Tuition_fees_up_to_date", 
        "Scholarship_holder", 
        "Debtor", 
        "Gender"
    ]
    
    input_df = pd.DataFrame([input_data], columns=numeric_features + categorical_features)
    scaled_numeric = scaler.transform(input_df[numeric_features])
    encoded_categorical = encoder.transform(input_df[categorical_features])
    transformed_data = np.hstack([scaled_numeric, encoded_categorical])
    
    try:
        prediction = model.predict(transformed_data)
        prediction_proba = model.predict_proba(transformed_data)
        status_dict = {0: "Graduate", 1: "Dropout"}
        predicted_status = status_dict[prediction[0]]
        graduate_prob = prediction_proba[0][0]
        dropout_prob = prediction_proba[0][1]
        return predicted_status, graduate_prob, dropout_prob
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

loaded_model = joblib.load('best_model.joblib')
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.title('Student Dropout Prediction')

curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Semester Approved', min_value=0, max_value=30, value=5)
curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Semester Grade', min_value=0, max_value=20, value=5)
curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Semester Approved', min_value=0, max_value=30, value=6)
curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Semester Grade', min_value=0, max_value=20, value=9)
age_at_enrollment = st.number_input('Age at Enrollment', min_value=15, max_value=70, value=20)
tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
scholarship_holder = st.selectbox('Scholarship Holder', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
debtor = st.selectbox('Debtor', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')

input_data = [
    curricular_units_2nd_sem_approved,
    curricular_units_2nd_sem_grade,
    curricular_units_1st_sem_approved,
    curricular_units_1st_sem_grade,
    age_at_enrollment,
    tuition_fees_up_to_date,
    scholarship_holder,
    debtor,
    gender
]

if st.button('Predict'):
    predicted_status, graduate_prob, dropout_prob = preprocess_and_predict(
        input_data, loaded_model, scaler, encoder
    )
    if predicted_status:
        st.write(f"Prediction: **{predicted_status}**")
        st.write(f"Probability of Graduate: **{graduate_prob:.2f}**")
        st.write(f"Probability of Dropout: **{dropout_prob:.2f}**")
