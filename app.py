import pandas as pd
import matplotlib.pyplot as plt

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
# 📈 GRAPH 1: CLV Breakdown
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
ax.set_xlabel("Factors")
ax.set_ylabel("Contribution Value")

st.pyplot(fig)

# ==============================
# 📈 GRAPH 2: Customer Profile
# ==============================
st.subheader("📊 Customer Profile Overview")

profile_data = {
    "Feature": ["Credit Score", "Age", "Tenure", "Products"],
    "Value": [credit_score, age, tenure, products]
}

df2 = pd.DataFrame(profile_data)

fig2, ax2 = plt.subplots()
ax2.plot(df2["Feature"], df2["Value"], marker='o')
ax2.set_title("Customer Profile Metrics")

st.pyplot(fig2)
