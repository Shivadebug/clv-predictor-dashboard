import streamlit as st
import numpy as np
import pickle
import lzma
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL
# ==============================
with lzma.open("clv_model_small.pkl.xz", "rb") as f:
    model = pickle.load(f)

scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="CLV Predictor", layout="wide")

# ==============================
# TITLE
# ==============================
st.title("💰 Customer Lifetime Value Dashboard")
st.markdown("Predict how valuable a customer is for your business.")

# ==============================
# INPUT SECTION
# ==============================
st.subheader("📥 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 80, 30)
    tenure = st.slider("Tenure (Years with Bank)", 0, 10, 5)
    balance = st.number_input("Account Balance", value=50000.0)
    products = st.selectbox("Number of Bank Products", [1,2,3,4])

with col2:
    salary = st.number_input("Estimated Salary", value=50000.0)

    active = st.selectbox(
        "Is Active Customer?",
        ["No", "Yes"]
    )

    gender = st.selectbox("Gender", ["Female", "Male"])

    geography = st.selectbox(
        "Country",
        ["France", "Germany", "Spain"]
    )

    has_card = st.selectbox(
        "Has Credit Card?",
        ["No", "Yes"]
    )

# ==============================
# ENCODING
# ==============================
gender = 1 if gender == "Male" else 0
active = 1 if active == "Yes" else 0
has_card = 1 if has_card == "Yes" else 0

geo_map = {"France":0, "Germany":1, "Spain":2}
geography = geo_map[geography]

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict CLV"):

    input_data = np.array([[credit_score, geography, gender, age,
                            tenure, balance, products, has_card,
                            active, salary]])

    input_scaled = scaler.transform(input_data)
    raw_pred = model.predict(input_scaled)[0]

    # Custom CLV logic
    prediction = (
        (balance * 0.3) +
        (salary * 0.5) +
        (tenure * 2000) +
        (products * 5000) +
        (active * 10000)
    )

    prediction += raw_pred * 50

    # ==============================
    # RESULTS
    # ==============================
    st.subheader("📊 Prediction Result")

    colA, colB, colC = st.columns(3)

    colA.metric("💰 CLV Value", f"₹ {round(prediction,2)}")

    if prediction > 100000:
        cust_type = "High Value 🌟"
        risk = "Low Risk"
    elif prediction > 50000:
        cust_type = "Medium Value ⚖️"
        risk = "Moderate Risk"
    else:
        cust_type = "Low Value ⚠️"
        risk = "High Risk"

    colB.metric("Customer Type", cust_type)
    colC.metric("Risk Level", risk)

    # ==============================
    # 📊 GRAPH 1: CLV Breakdown
    # ==============================
    st.subheader("📊 CLV Contribution Breakdown")

    data = {
        "Factor": ["Balance", "Salary", "Tenure", "Products", "Activity"],
        "Contribution": [
            balance * 0.3,
            salary * 0.5,
            tenure * 2000,
            products * 5000,
            active * 10000
        ]
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df["Factor"], df["Contribution"])
    ax.set_title("CLV Contribution by Factors")

    st.pyplot(fig)

    # ==============================
    # 📊 GRAPH 2: Customer Profile
    # ==============================
    st.subheader("📊 Customer Profile Overview")

    profile_data = {
        "Feature": ["Credit Score", "Age", "Tenure", "Products"],
        "Value": [credit_score, age, tenure, products]
    }

    df2 = pd.DataFrame(profile_data)

    fig2, ax2 = plt.subplots()
    ax2.plot(df2["Feature"], df2["Value"], marker='o')

    st.pyplot(fig2)

    # ==============================
    # SUMMARY
    # ==============================
    st.subheader("📝 Business Summary")

    st.write(f"""
    This customer is predicted to generate **₹ {round(prediction,2)}** over their lifetime.

    🔹 **Customer Category:** {cust_type}  
    🔹 **Risk Level:** {risk}  

    📌 **Insights:**
    - Higher salary & balance → higher value  
    - Long-term customers → more profitable  
    - Active users → better retention  

    📊 **Recommended Action:**
    - High Value → Offer premium benefits  
    - Medium Value → Upsell services  
    - Low Value → Improve engagement  
    """)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📌 How to Use")

st.sidebar.write("""
1. Enter customer details  
2. Click Predict  
3. View results & graphs  

This tool helps identify:
✔ Valuable customers  
✔ Risk levels  
✔ Retention strategies  
""")
