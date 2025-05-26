import streamlit as st
import numpy as np
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="PCOS Risk Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("pcos_rf_model.pkl")
    return model

model = load_model()

# Define possible blood groups
blood_group_mapping = {
    "A+": 0, "A-": 1, "B+": 2, "B-": 3, "AB+": 4, "AB-": 5, "O+": 6, "O-": 7
}

# Hair Loss mapping
hair_loss_mapping = {
    "No": 0,
    "Yes": 1
}

# App title and instructions
st.title("🩺 PCOS Risk Prediction")
st.markdown("Please fill in the following medical details to predict your PCOS risk.")

# Input form
with st.form("pcos_form"):
    age = st.number_input("Age (Yrs):", min_value=0, max_value=100, step=1)
    blood_group = st.selectbox("Blood Group:", list(blood_group_mapping.keys()))
    pulse_rate = st.number_input("Pulse Rate (bpm):", min_value=30.0, max_value=200.0, step=0.1)
    hair_loss = st.selectbox("Hair Loss:", list(hair_loss_mapping.keys()))
    follicle_left = st.number_input("Follicle No Left Ovary:", min_value=0.0, max_value=50.0, step=0.1)
    follicle_right = st.number_input("Follicle No Right Ovary:", min_value=0.0, max_value=50.0, step=0.1)
    endometrium = st.number_input("Endometrium Thickness (mm):", min_value=0.0, max_value=20.0, step=0.1)

    submit = st.form_submit_button("🔍 Predict PCOS Risk")

# Process prediction
if submit:
    try:
        # Convert inputs to model-ready format
        input_data = [
            float(age),
            blood_group_mapping[blood_group],
            float(pulse_rate),
            hair_loss_mapping[hair_loss],
            float(follicle_left),
            float(follicle_right),
            float(endometrium)
        ]

        input_array = np.array(input_data).reshape(1, -1)
        probability = model.predict_proba(input_array)[0][1] * 100
        probability = round(probability, 2)

        # Risk status
        if probability > 70:
            status = "⚠️ High Risk"
        elif probability > 40:
            status = "🟡 Moderate Risk"
        else:
            status = "🟢 Low Risk"

        st.success(f"✅ PCOS Risk Score: **{probability}%**")
        st.info(f"Prediction Status: **{status}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
