# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import openai

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Battery SOH Chatbot",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 Battery Pack SOH Chatbot")
st.markdown("Upload the **PulseBat Dataset** and chat with the battery assistant.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload PulseBat Dataset (.csv)", type=["csv"])

# -------------------------------
# Chat Memory
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# -------------------------------
# Train Linear Regression
# -------------------------------
def train_model(df):
    feature_cols = [col for col in df.columns if col.startswith("U")]  # U1–U21
    X = df[feature_cols]
    y = df["SOH"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    metrics = {
        "R²": r2_score(y, y_pred),
        "MSE": mean_squared_error(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred)
    }
    return model, metrics

# -------------------------------
# Battery Health Classification
# -------------------------------
def classify_soh(soh_value, threshold=0.6):
    if soh_value < threshold:
        return "⚠️ The battery has a problem."
    else:
        return "✅ The battery is healthy."

# -------------------------------
# ChatGPT Integration
# -------------------------------
def ask_chatgpt(prompt):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]  # Add your key in Streamlit secrets
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant for battery health questions."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ ChatGPT error: {str(e)}"

# -------------------------------
# Main App Logic
# -------------------------------
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Train Model
    model, metrics = train_model(df)

    st.write("### Model Performance")
    st.json(metrics)

    # ---------------------------
    # Chatbot Section
    # ---------------------------
    st.write("### 💬 Chat with the Battery Assistant")

    user_input = st.chat_input("Ask something like 'Check battery SOH' or 'How to extend battery life?'")

    if user_input:
        # Save user message
        st.session_state.chat_history.append(("user", user_input))

        # Process "check battery SOH"
        if "soh" in user_input.lower():
            random_sample = df.sample(1).drop(columns=["SOH"])  # pick a random row
            predicted_soh = model.predict(random_sample)[0]
            health_status = classify_soh(predicted_soh)

            bot_reply = f"🔮 Predicted SOH: **{predicted_soh:.2f}**\n\n{health_status}"

        else:
            # General Q&A via ChatGPT
            bot_reply = ask_chatgpt(user_input)

        # Save bot reply
        st.session_state.chat_history.append(("bot", bot_reply))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(message)
        else:
            st.chat_message("assistant").markdown(message)

else:
    st.info("📂 Please upload the PulseBat Dataset CSV file to start.")
