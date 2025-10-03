import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import openai

# ---------------------------
# STREAMLIT APP CONFIG
# ---------------------------
st.set_page_config(page_title="🔋 Battery SOH Chatbot", layout="wide")

st.title("🔋 Battery Pack SOH Prediction & Chatbot")
st.markdown("Upload your dataset, predict **SOH**, and ask battery-related questions!")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # ---------------------------
    # MODEL TRAINING
    # ---------------------------
    if "soh" not in df.columns.str.lower():
        st.error("❌ Dataset must contain an 'SOH' column as the target variable.")
    else:
        # Identify target column (case-insensitive)
        target_col = [c for c in df.columns if c.lower() == "soh"][0]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        st.write("### 📊 Model Evaluation")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.3f}")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")

        # ---------------------------
        # USER INPUTS
        # ---------------------------
        st.subheader("🔢 Enter Battery Cell Features for Prediction")

        user_inputs = {}
        for col in X.columns:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())

            user_inputs[col] = st.number_input(
                f"{col}",
                col_min,
                col_max,
                col_mean,
                step=(col_max - col_min) / 100 if col_max > col_min else 0.01
            )

        if st.button("⚡ Predict SOH"):
            input_df = pd.DataFrame([user_inputs])
            soh_pred = model.predict(input_df)[0]

            st.success(f"🔮 Predicted SOH: **{soh_pred:.3f}**")

            if soh_pred < 0.6:
                st.error("🚨 The battery has a problem.")
            else:
                st.success("✅ The battery is healthy.")

        # ---------------------------
        # CHATBOT SECTION
        # ---------------------------
        st.subheader("🤖 Battery Chatbot")

        openai.api_key = st.secrets.get("OPENAI_API_KEY", None)

        if not openai.api_key:
            st.warning("⚠️ No OpenAI API key found. Add it in `.streamlit/secrets.toml`.")
        else:
            user_question = st.text_input("Ask a battery-related question:")

            if user_question:
                with st.spinner("Thinking..."):
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for battery health and recycling."},
                            {"role": "user", "content": user_question}
                        ]
                    )
                    st.write("💡 Answer:", response["choices"][0]["message"]["content"])
