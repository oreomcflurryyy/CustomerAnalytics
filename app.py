import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

pipeline = joblib.load(
    "models/customer_churn_pipeline.pkl"
)

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    72,
    24
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    20.0,
    120.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    value=1500.0
)

customer = pd.DataFrame({

    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phone],
    "MultipleLines":[multiple],
    "InternetService":[internet],
    "OnlineSecurity":[security],
    "OnlineBackup":[backup],
    "DeviceProtection":[device],
    "TechSupport":[support],
    "StreamingTV":[tv],
    "StreamingMovies":[movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]

})

if st.button("Predict Churn"):

    prediction = pipeline.predict(customer)[0]

    probability = pipeline.predict_proba(customer)[0][1]

    st.subheader("Prediction")

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn.")

    else:

        st.success("✅ Customer is unlikely to churn.")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )