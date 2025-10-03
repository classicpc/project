import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import openai
import os

st.set_page_config(page_title="Battery SOH Chatbot", page_icon="🔋", layout="wide")
st.title("🔋 Battery SOH Prediction & Chatbot")

# ------------------------------
# 1. Upload Dataset
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload PulseBat CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Ensure dataset has expected columns
    # Example assumption: U1, U2, ..., U21, SOH
    feature_cols = [col for col in df.columns if col.startswith("U")]
    target_col = "SOH"

    if target_col in df.columns and len(feature_cols) == 21:
        X = df[feature_cols]
        y = df[target_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        st.success("✅ Model trained successfully using uploaded dataset!")

        # ------------------------------
        # 2. Prediction Function
        # ------------------------------
        def predict_soh(cells, threshold=0.6):
            soh = model.predict([cells])[0]
            status = "✅ Healthy" if soh >= threshold else "⚠️ Problem"
            return soh, status

        # ------------------------------
        # 3. ChatGPT Integration
        # ------------------------------
        def ask_chatgpt(prompt):
            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"⚠️ ChatGPT Error: {e}"

        # ------------------------------
        # 4. Tabs Layout
        # ------------------------------
        tab1, tab2 = st.tabs(["🔮 Predict Battery SOH", "💬 Chat with Battery Bot"])

        # ---- Tab 1: Prediction ----
        with tab1:
            st.subheader("Enter SOH values for cells U1–U21")

            cells = []
            cols = st.columns(7)
            for i, col in enumerate(feature_cols):
                with cols[i % 7]:
                    val = st.number_input(f"{col}", 0.0, 1.0, float(df[col].mean()), step=0.01)
                    cells.append(val)

            threshold = st.slider("SOH Threshold", 0.1, 1.0, 0.6, step=0.05)

            if st.button("🔍 Predict"):
                soh, status = predict_soh(cells, threshold)
                st.metric("Predicted Pack SOH", f"{soh:.2f}")
                st.success(f"Battery Status: {status}")

        # ---- Tab 2: Chatbot ----
        with tab2:
            st.subheader("Ask me about batteries 🔋")
            user_input = st.text_input("Your question:")

            if st.button("💡 Ask"):
                if user_input.strip():
                    reply = ask_chatgpt(user_input)
                    st.markdown(f"**Chatbot:** {reply}")
                else:
                    st.warning("Please enter a question.")
    else:
        st.error("⚠️ The dataset must have columns: U1–U21 and SOH")
else:
    st.info("👆 Upload your PulseBat dataset (.csv) to get started.")
