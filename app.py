import streamlit as st
import joblib
import numpy as np

st.title("SalaryBot")
st.divider()

st.write("Well, with this app you can get estimations for the salaries of the company employees.")

# Input fields for years and job rate
years = st.number_input("Years of experience", value=1, step=1, min_value=0)
jobrate = st.number_input("Job rate", value=3.5, step=0.5, min_value=0.0)

# Prepare input features for prediction
x = [years, jobrate]

# Load the pre-trained model
model = joblib.load("linearmodel.pkl")

st.divider()

# Button to trigger the prediction
predict = st.button("Press the button for salary prediction")

if predict:
    st.balloons()  # Display balloons for fun
    x1 = np.array([x])  # Convert input to numpy array
    prediction = model.predict(x1)  # Make prediction

    st.write(f"Salary prediction is: {prediction[0]}")  # Display the prediction result

else:
    st.write("Please press the button for the app to make the prediction.")
