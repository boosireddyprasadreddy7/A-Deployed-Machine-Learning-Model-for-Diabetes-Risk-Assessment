import streamlit as st
import pickle
import numpy as np
import os

# Load the scaler and model
try:
    with open(os.path.join('Models', 'scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(os.path.join('Models', 'best_model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model or scaler not found. Please make sure 'scaler.pkl' and 'best_model.pkl' are in the 'Folders' directory.")
    st.stop()

# Title of the web app
st.title('Diabetes Prediction App')

st.write("Please enter the following details to predict diabetes:")

# Create input fields for the features
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=72)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=29)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47)
age = st.number_input('Age', min_value=0, max_value=120, value=33)

# One-hot encoded features
st.write("Select BMI Category:")
bmi_obesity_1 = st.checkbox('Obesity 1')
bmi_obesity_2 = st.checkbox('Obesity 2')
bmi_obesity_3 = st.checkbox('Obesity 3')
bmi_overweight = st.checkbox('Overweight')
bmi_underweight = st.checkbox('Underweight')


st.write("Select Glucose Category:")
glucose_normal = st.checkbox('Normal Glucose')
glucose_overweight = st.checkbox('Overweight Glucose')
glucose_secret = st.checkbox('Secret Glucose')


# Prediction button
if st.button('Predict'):
    # Create the input array for the model
    # The order of features should be the same as in the training data
    input_data = np.array([[
        pregnancies, blood_pressure, skin_thickness, insulin, dpf, age,
        float(bmi_obesity_1), float(bmi_obesity_2), float(bmi_obesity_3),
        float(bmi_overweight), float(bmi_underweight),
        float(glucose_normal), float(glucose_overweight), float(glucose_secret)
    ]])

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(scaled_data)

    # Display the result
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.write('The model predicts that the person has **Diabetes**.')
    else:
        st.write('The model predicts that the person does **not** have Diabetes.')

