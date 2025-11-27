# 🔋 Battery Pack SOH Prediction Platform

Interactive Streamlit app for analyzing PulseBat battery-pack data, training a linear-regression SOH model, and chatting with an AI battery assistant. Built as the SOFE3370 Final Project (Group 18).


---

## ✨ Highlights

- **Upload-ready PulseBat workflow** – drop in a CSV with 670 pack measurements (U1–U21) and instantly explore health stats, quality metrics, and correlations.
- **Configurable linear regression** – experiment with cell sorting strategies, train/test splits, CV folds, and thresholds to see how they influence R² / RMSE.
- **Production-grade UI** – dual-tone layout, responsive stat cards, glassmorphic sidebar, and tabbed analytics for fast navigation.
- **AI battery assistant** – Retrieval-Augmented Generation (RAG) combines your dataset, model metrics, and curated knowledge base to answer domain questions.
- **Feedback tooling** – every assistant response exposes a rating popover so stakeholders can capture qualitative feedback.

---

## 🧱 Project Structure

```
project/
├── streamlit_app.py            # Main Streamlit entrypoint
├── requirements.txt            # Python dependencies
├── PulseBat Data Description.md# Optional local documentation surfaced in-app
├── README.md                   # You are here
└── LICENSE                     # MIT
```

---

## 🚀 Getting Started

### 1. Clone & install deps

```bash
git clone <repo>
cd project
pip install -r requirements.txt
```

> 💡 Tip: create a virtual environment (`python -m venv .venv && .venv\Scripts\activate`) before installing packages.

### 2. Provide an OpenAI API key

Pick one:

1. **Best practice** – export an environment variable
   ```powershell
   setx OPENAI_API_KEY "sk-your-key"
   ```
2. **Quick hack** – edit `streamlit_app.py` and replace the `OPENAI_API_KEY` constant (remember not to commit real keys!).

### 3. Run Streamlit

```bash
streamlit run streamlit_app.py
```

Open the provided localhost URL (default `http://localhost:8501`).

---

## 🛠️ Using the App

1. **Upload PulseBat CSV** via the sidebar uploader. Needs columns `U1`–`U21` plus `SOH`.
2. **Tune preprocessing** – choose ascending/descending sorting, SOH threshold, and advanced options (split %, CV folds, random seed).
3. **Review analytics** – the tabs expose dataset previews, distribution charts, correlation heatmaps, linear-regression metrics, and residual plots.
4. **Chat with the AI assistant** – pick a suggested pill or type your own question. The assistant blends:
   - The curated knowledge base (maintenance, chemistry, recycling, safety, etc.)
   - Real-time regression metrics & dataset snapshots
   - Your latest user prompt
5. **Provide feedback** – rate responses to capture qualitative QA notes for later analysis.

---

## ⚙️ Configuration Notes

- **RAG context trim** – the app caps assistant context at ~20k chars to avoid token-limit errors.
- **Dataset documentation** – place a `PulseBat Data Description.md` next to the app to surface it in chat responses.
- **File uploads to OpenAI** – for very large CSVs, consider the OpenAI Files API + Assistants workflow (upload once, reuse the `file_id` per chat turn).
- **Environment variables** – besides `OPENAI_API_KEY`, you can set `STREAMLIT_SERVER_PORT` etc. using Streamlit’s standard settings.

---

## 📊 Tech Stack

- **Frontend**: Streamlit (chat elements, tabs, custom CSS)
- **Visualization**: Plotly Express / Graph Objects, seaborn, matplotlib
- **ML**: pandas, NumPy, scikit-learn (LinearRegression, StandardScaler, CV utilities)
- **AI Assistant**: OpenAI GPT-4.1 (chat completions API + RAG helper)

---

## 🧪 Testing Checklist

- Upload sample PulseBat CSV (ensure columns U1–U21 + SOH).
- Toggle dataset preview (full vs. head) and verify metrics update.
- Change train/test split, CV folds, and sorting – confirm metrics & plots refresh.
- Ask the assistant:
  - “Analyze the current Linear Regression model performance.”
  - “Give me battery maintenance tips.”
- Trigger feedback popover and submit a sample rating.

---

## 🙌 Credits

Created by **Group 18** – Pranav Ashok Chaudhari, Tarun Modekurty, Leela Alagala, Hannah Albi, Vandan Patel – for the SOFE3370 Final Project, 2025.

Licensed under the MIT License. PRs and feature requests welcome!
