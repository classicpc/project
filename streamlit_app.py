# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ===============================
# Load and preprocess dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("PulseBat Dataset.csv")  # ensure the file is uploaded in same directory
    return df

df = load_data()

# Features: U1–U21 cells, Target: SOH
cell_cols = [col for col in df.columns if col.startswith("U")]
X = df[cell_cols]
y = df["SOH"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Battery SOH Chatbot", layout="wide")

st.title("🔋 Battery Pack SOH Prediction & Chatbot")
st.write("Predict battery State of Health (SOH) and chat about batteries.")

# Show dataset
with st.expander("📂 View Dataset"):
    st.dataframe(df.head())

# ===============================
# Model Evaluation
# ===============================
st.subheader("📊 Model Evaluation")
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.metric("R² Score", f"{r2:.3f}")
st.metric("MSE", f"{mse:.3f}")
st.metric("MAE", f"{mae:.3f}")

# Plot predicted vs actual
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
ax.set_xlabel("Actual SOH")
ax.set_ylabel("Predicted SOH")
ax.set_title("Predicted vs Actual SOH")
st.pyplot(fig)

# ===============================
# Prediction Section
# ===============================
st.subheader("⚡ Check Battery SOH")

# Threshold slider
threshold = st.slider("Set SOH Threshold", 0.0, 1.0, 0.6, 0.01)

# User input for each cell
with st.form("prediction_form"):
    st.write("Enter cell values (U1–U21):")
    inputs = []
    cols = st.columns(3)
    for i, col in enumerate(cell_cols):
        val = cols[i % 3].number_input(
            col, 
            min_value=0.0, 
            max_value=1.0, 
            value=float(df[col].mean()), 
            step=0.01
        )
        inputs.append(val)
    submitted = st.form_submit_button("Predict SOH")

if submitted:
    features = np.array(inputs).reshape(1, -1)
    soh_pred = model.predict(features)[0]
    st.success(f"Predicted SOH: {soh_pred:.3f}")
    if soh_pred < threshold:
        st.error("The battery has a problem ⚠️")
    else:
        st.success("The battery is healthy ✅")

# ===============================
# Chatbot Section
# ===============================
st.subheader("💬 Battery Chatbot")

# Load OpenAI API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant specialized in batteries."}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me about batteries..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ===============================
# Save Model Option
# ===============================
with st.expander("💾 Save Model"):
    if st.button("Export trained model"):
        joblib.dump(model, "linear_regression_model.pkl")
        st.success("Model saved as linear_regression_model.pkl")
