import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Covid Estimation")
st.title("Covid Patient Prediction")

# Load the pre-trained model
with open("model1.pkl", "rb") as file:
    LR = pickle.load(file)

st.header("Input Features")

# Input fields for user
age = st.text_input("Age", placeholder="Enter age in years")
fever = st.text_input("Fever", placeholder="Enter fever in Celsius")
gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cough = st.radio("Cough",options=[0, 1], format_func=lambda x: "Mild" if x == 0 else "Strong")
city = st.radio("City", options=["Bangalore", "Delhi", "Kolkata", "Mumbai"])

city_mapping= {
    'Bangalore': 0,
    'Delhi': 1,
    'Kolkata': 2,
    'Mumbai':3
}

x= city_mapping[city]


# Prepare the input data
test_input = {
    "age": [age],
    "gender": [gender],
    "cough": [cough],
    "city": [x],
    "fever": [fever],
}
test_df = pd.DataFrame(test_input)

# Prediction button and display results
if st.button("Predict"):
    try:
        prediction = LR.predict(test_df)
        prediction_proba = LR.predict_proba(test_df)
        
        st.subheader("Prediction")
        if int(prediction[0]) == 0:
            st.markdown("Patient does not have Covid")
        else:
            st.markdown("Patient has Covid")

        st.subheader("Prediction Probability")
        st.write(f"Probability of having Covid: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of not having Covid: {prediction_proba[0][0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
