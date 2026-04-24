import streamlit as st
import pandas as pd
import joblib

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# -------------------- Load Model --------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -------------------- Header --------------------
st.markdown(
    "<h1 style='text-align: center; color: #e63946;'>❤️ Heart Disease Risk Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>Enter patient details to predict heart disease risk using Machine Learning</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------- Input Section --------------------
st.subheader("🧑‍⚕️ Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

with col2:
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st.divider()

# -------------------- Prediction Button --------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    # Create input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Match training columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # -------------------- Result UI --------------------
    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
        st.markdown("👉 Recommendation: Consult a cardiologist immediately.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.markdown("👉 Keep maintaining a healthy lifestyle.")

# -------------------- Footer --------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)