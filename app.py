import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
# Load model
model = joblib.load("model.pkl")
df = pd.read_csv("data/Telco_Customer_Churn.csv")
# Page Config
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Prediction System")

st.write("""
Predict whether a telecom customer will churn or not using Machine Learning.
""")

# Sidebar
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

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
    "Tenure (Months)",
    0, 72, 12
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    0.0, 200.0, 70.0
)

# Accuracy
st.sidebar.success("✅ Model Accuracy: 81.6%")

# Prediction Button
if st.button("Predict Churn"):

    gender_val = 1 if gender == "Male" else 0
    partner_val = 1 if partner == "Yes" else 0
    dependents_val = 1 if dependents == "Yes" else 0

    if contract == "Month-to-month":
        contract_val = 0
    elif contract == "One year":
        contract_val = 1
    else:
        contract_val = 2

    input_data = pd.DataFrame([[

        gender_val,
        senior,
        partner_val,
        dependents_val,
        tenure,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        contract_val,
        0,
        0,
        monthly_charges,
        0.500000

    ]], columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer will CHURN! Probability: {probability:.1%}"
        )
    else:
        st.success(
            f"✅ Customer will NOT churn. Probability: {probability:.1%}"
        )

    # Probability Meter
    st.subheader("Churn Probability")
    st.progress(float(probability))

    # Risk Level
    st.subheader("Risk Level")

    if probability > 0.7:
        st.error("🔴 High Risk Customer")

    elif probability > 0.4:
        st.warning("🟠 Medium Risk Customer")

    else:
        st.success("🟢 Low Risk Customer")

# Model Comparison Chart
st.subheader("📈 Model Accuracy Comparison")

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

scores = [
    81.6,
    79.5,
    79.4
]

fig, ax = plt.subplots()

ax.bar(models, scores)

ax.set_ylabel("Accuracy (%)")

st.pyplot(fig)
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis")

# Churn Distribution
fig1, ax1 = plt.subplots()

df['Churn'].value_counts().plot(
    kind='bar',
    ax=ax1
)

ax1.set_title("Churn Distribution")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# Contract Type vs Churn
fig2, ax2 = plt.subplots()

pd.crosstab(
    df['Contract'],
    df['Churn']
).plot(
    kind='bar',
    ax=ax2
)

ax2.set_title("Contract Type vs Churn")

st.pyplot(fig2)

# Monthly Charges Distribution
fig3, ax3 = plt.subplots()

ax3.hist(
    df['MonthlyCharges'],
    bins=20
)

ax3.set_title("Monthly Charges Distribution")

st.pyplot(fig3)
# Footer
st.markdown("---")
st.write("Built by Kavyasri Makani 🚀")