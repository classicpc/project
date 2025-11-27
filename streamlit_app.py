# 🔋 Battery Pack SOH Prediction & AI Assistant Platform
# SOFE3370 Final Project - Group 18
# Pranav Ashok Chaudhari, Tarun Modekurty, Leela Alagala, Hannah Albi

import io
import json
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from openai import OpenAI
import os
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cap the serialized context we send to the LLM so token limits are respected on every turn.
MAX_ASSISTANT_CONTEXT_CHARS = 20000

def _json_default_serializer(obj):
    """Convert numpy/pandas objects to JSON-safe primitives."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    return str(obj)

# -------------------------------
# Internal API Configuration
# -------------------------------
# NOTE: The API key is hardcoded for convenience per user request.
# Do NOT commit real secrets to public repos.
OPENAI_API_KEY = "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"
DEFAULT_CHAT_MODEL = "gpt-4.1"
CHAT_COMPATIBLE_MODELS = {
    "gpt-5.1",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "o4-mini",
}

MODEL_CHOICES = [
    ("🔥 GPT-5.1 Codex (best)", "gpt-5.1-codex"),
    ("⚡ GPT-5.1", "gpt-5.1"),
    ("🧠 GPT-5", "gpt-5"),
    ("🛠️ GPT-5 Codex", "gpt-5-codex"),
    ("💬 GPT-5 Chat Latest", "gpt-5-chat-latest"),
    ("🚀 GPT-4.1", "gpt-4.1"),
    ("🌈 GPT-4o", "gpt-4o"),
    ("🧪 o1", "o1"),
    ("🛰️ o3", "o3"),
    ("⚙️ GPT-5.1 Codex Mini", "gpt-5.1-codex-mini"),
    ("⚡ GPT-5 Mini", "gpt-5-mini"),
    ("🔋 GPT-5 Nano", "gpt-5-nano"),
    ("📦 GPT-4.1 Mini", "gpt-4.1-mini"),
    ("📱 GPT-4.1 Nano", "gpt-4.1-nano"),
    ("🎯 GPT-4o Mini", "gpt-4o-mini"),
    ("🧩 o1 Mini", "o1-mini"),
    ("🧭 o3 Mini", "o3-mini"),
    ("💡 o4 Mini", "o4-mini"),
    ("⚡ Codex Mini Latest", "codex-mini-latest"),
]


def get_effective_api_key(override: str | None = None) -> str:
    """Resolve the API key, preferring explicit override, then env, then bundled fallback."""
    if override:
        return override.strip()

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    return OPENAI_API_KEY


def get_selected_model() -> str:
    selected = st.session_state.get("model_choice", DEFAULT_CHAT_MODEL)
    if selected not in CHAT_COMPATIBLE_MODELS:
        st.session_state["model_fallback_warning"] = selected
        return DEFAULT_CHAT_MODEL

    st.session_state["model_fallback_warning"] = None
    return selected

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🔋 Battery Pack SOH Prediction Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for two-tone professional UI. Keeping it inline makes deployment easier than
# shipping a separate stylesheet, which is handy for student projects.
st.markdown(r"""
<style>
    /* Import Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles - Two Tone Theme: Deep Navy (#0A1929) + Electric Cyan (#00D4FF) */
    * {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container - Clean White Background */
    .main {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFB 100%);
        padding: 0 !important;
    }
    
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px;
    }
    
    /* Premium Header - Two Tone Design */
    .main-header {
        background: linear-gradient(135deg, #0A1929 0%, #1A2F4A 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        border: 2px solid rgba(0, 212, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0, 212, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #FFFFFF 0%, #00D4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header h3 {
        font-size: 1.1rem;
        font-weight: 400;
        color: #00D4FF;
        margin-bottom: 0.3rem;
    }
    
    .main-header p {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
    }
    
    /* Sidebar - Deep Navy Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1929 0%, #05101C 100%);
        padding: 2rem 1rem;
        border-right: 2px solid rgba(0, 212, 255, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #E3F2FD !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #00D4FF !important;
        font-weight: 600;
    }

    .sidebar-section-heading {
        font-size: 1rem;
        font-weight: 600;
        color: #9EE8FF;
        letter-spacing: 0.04em;
        margin-bottom: 0.75rem;
    }

    .sidebar-stat-grid {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .sidebar-stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.35);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .sidebar-stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(0, 212, 255, 0.25);
    }

    .sidebar-stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: rgba(255, 255, 255, 0.72);
        margin-bottom: 0.35rem;
    }

    .sidebar-stat-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #5FE7FF;
        margin: 0;
    }

    .sidebar-stat-detail {
        font-size: 0.82rem;
        color: rgba(255, 255, 255, 0.6);
        margin: 0.15rem 0 0;
    }

    @media (max-width: 900px) {
        .sidebar-stat-card {
            padding: 0.8rem 0.9rem;
        }

        .sidebar-stat-value {
            font-size: 1.4rem;
        }
    }
    
    /* Chat Interface - Two Tone Design */
    
    /* User Message - Cyan Tone */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: linear-gradient(135deg, #00D4FF 0%, #00A8CC 100%) !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 1.2rem 1.5rem !important;
        margin: 1rem 0 !important;
        margin-left: auto !important;
        max-width: 75% !important;
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.3) !important;
        border: none !important;
        animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    [data-testid="stChatMessage"][data-testid*="user"] p {
        color: #0A1929 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    
    /* Assistant Message - Navy Tone */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: linear-gradient(135deg, #0A1929 0%, #1A2F4A 100%) !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 1.8rem !important;
        margin: 1rem 0 !important;
        max-width: 80% !important;
        box-shadow: 0 6px 24px rgba(10, 25, 41, 0.35) !important;
        border: 2px solid rgba(0, 212, 255, 0.3) !important;
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] h2 {
        color: #00D4FF !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] h3 {
        color: #00D4FF !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stChatMessage"][data-testid*="assistant"] strong {
        color: #00D4FF !important;
    }
    
    /* Chat Input - Modern Design */
    [data-testid="stChatInput"] {
        background: #FFFFFF;
        border-radius: 30px;
        border: 2px solid #E0E0E0;
        padding: 0.7rem 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #00D4FF;
        box-shadow: 0 6px 30px rgba(0, 212, 255, 0.25);
        transform: translateY(-2px);
    }
    
    [data-testid="stChatInput"] input {
        color: #0A1929 !important;
        font-size: 0.95rem !important;
    }
    
    /* Pills (Suggestions) - Cyan Theme */
    [data-testid="stHorizontalBlock"] button {
        background: linear-gradient(135deg, #00D4FF 0%, #00A8CC 100%);
        color: #0A1929;
        border: none;
        border-radius: 30px;
        padding: 0.7rem 1.6rem;
        font-weight: 600;
        font-size: 0.88rem;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    [data-testid="stHorizontalBlock"] button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, #00E5FF 0%, #00B8D4 100%);
    }
    
    [data-testid="stHorizontalBlock"] button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Tabs - Two Tone Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF;
        border-radius: 12px 12px 0 0;
        padding: 0.9rem 2rem;
        font-weight: 600;
        color: #666666;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #F5F5F5;
        color: #00D4FF;
        border-color: rgba(0, 212, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0A1929 0%, #1A2F4A 100%);
        color: #00D4FF !important;
        border: 2px solid #00D4FF;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.25);
    }
    
    /* Buttons - Cyan Theme */
    .stButton > button {
        background: linear-gradient(135deg, #00D4FF 0%, #00A8CC 100%);
        color: #0A1929;
        border: none;
        border-radius: 14px;
        padding: 0.9rem 2.8rem;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 28px rgba(0, 212, 255, 0.45);
        background: linear-gradient(135deg, #00E5FF 0%, #00B8D4 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Info Boxes - Cyan Accent */
    .info-box {
        background: linear-gradient(135deg, #E3F9FF 0%, #B8EEFF 100%);
        border-left: 4px solid #00D4FF;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.15);
    }
    
    /* Metrics - Clean Design */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFB 100%);
        padding: 1.8rem;
        border-radius: 14px;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.08);
        border: 2px solid #F0F0F0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.4);
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FBFF 100%);
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #666666;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 700;
        background: linear-gradient(135deg, #0A1929 0%, #00D4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
    }
    
    /* Health Status Cards */
    .health-good {
        background: linear-gradient(135deg, #00C853 0%, #00A843 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 200, 83, 0.3);
        font-weight: 600;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .health-bad {
        background: linear-gradient(135deg, #FF3D00 0%, #DD2C00 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(255, 61, 0, 0.3);
        font-weight: 600;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F5F5F5 0%, #E8E8E8 100%);
        border-radius: 12px;
        padding: 1.2rem;
        font-weight: 600;
        border: 2px solid #E0E0E0;
        color: #0A1929;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #E8F8FF 0%, #D4F1FF 100%);
        border-color: #00D4FF;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.15);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
        border-left: 4px solid #00C853;
        border-radius: 12px;
        padding: 1rem;
        color: #1B5E20;
    }
    
    .stError {
        background: linear-gradient(135deg, #FFCCBC 0%, #FFAB91 100%);
        border-left: 4px solid #FF3D00;
        border-radius: 12px;
        padding: 1rem;
        color: #BF360C;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #B3E5FC 0%, #81D4FA 100%);
        border-left: 4px solid #00D4FF;
        border-radius: 12px;
        padding: 1rem;
        color: #01579B;
    }
    
    /* Scrollbar - Cyan Theme */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F5F5F5;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00D4FF 0%, #00A8CC 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00A8CC 0%, #008FA6 100%);
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border: 2px solid #F0F0F0;
    }
    
    /* Spinner - Cyan */
    .stSpinner > div {
        border-top-color: #00D4FF !important;
    }

    /* Section & Stat Cards */
    .section-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(10, 25, 41, 0.05);
        box-shadow: 0 25px 60px rgba(10, 25, 41, 0.08);
        margin-bottom: 1.5rem;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .stat-card {
        background: linear-gradient(180deg, rgba(10, 25, 41, 0.9) 0%, rgba(10, 25, 41, 0.95) 100%);
        border-radius: 18px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 35px rgba(0, 212, 255, 0.25);
    }

    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.15rem;
        color: rgba(255, 255, 255, 0.65);
        margin-bottom: 0.75rem;
    }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #00D4FF;
    }

    .stat-detail {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.5rem;
    }

    .section-title {
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.5rem;
        letter-spacing: 0.03rem;
    }

    .section-subtitle {
        color: #000000;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🔋 Battery Pack SOH Prediction Platform</h1>
    <h3>AI-Powered Battery Health Assessment</h3>
    <p>State-of-the-art analytics, predictions, and an expert assistant — all in one app.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
with st.sidebar:
    st.markdown("## 🎛️ Configuration Panel")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "📁 Upload PulseBat Dataset", 
        type=["csv"],
        help="Upload your PulseBat dataset CSV file"
    )
    
    st.markdown("---")
    
    # Model Settings
    st.markdown("### 🤖 Model Configuration")
    
    # Preprocessing Options
    sort_method = st.selectbox(
        "📊 Cell Sorting Method:",
        ["None", "Ascending", "Descending"],
        help="Sort U1–U21 cell values for pattern analysis"
    )
    
    # SOH Threshold
    threshold = st.slider(
        "⚡ SOH Health Threshold:",
        min_value=0.0, max_value=1.0, value=0.6, step=0.01,
        help="Threshold below which battery is considered unhealthy"
    )
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        test_size = st.slider("Train/Test Split", 0.1, 0.5, 0.2)
        cv_folds = st.number_input("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random State", 1, 100, 42)
    
    st.markdown("---")

    st.markdown("### 🤖 AI Assistant Configuration")
    st.caption("The embedded project API key is used automatically for chatbot responses.")

    model_values = [choice[1] for choice in MODEL_CHOICES]
    model_label_lookup = {value: label for label, value in MODEL_CHOICES}
    current_model = st.session_state.get("model_choice", model_values[0])
    try:
        current_index = model_values.index(current_model)
    except ValueError:
        current_index = 0

    selected_model = st.selectbox(
        "Preferred GPT Model",
        options=model_values,
        index=current_index,
        format_func=lambda value: model_label_lookup.get(value, value),
        help=(
            "Premium tier: up to 250k tokens/day on GPT-5.1, GPT-5.1-Codex, GPT-5, GPT-5-Codex, GPT-5-Chat-Latest,"
            " GPT-4.1, GPT-4o, o1, o3."
            " Efficiency tier: up to 2.5M tokens/day on GPT-5.1-Codex-Mini and other *mini/nano* variants."
        ),
    )
    st.session_state.model_choice = selected_model
    st.caption(f"Model selected: {model_label_lookup.get(selected_model, selected_model)}")
    if selected_model not in CHAT_COMPATIBLE_MODELS:
        st.warning(
            f"{selected_model} uses the completions endpoint. Falling back to {DEFAULT_CHAT_MODEL} for chat responses.",
            icon="⚠️",
        )


# -------------------------------
# Chat Memory and RAG System
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "model_choice" not in st.session_state:
    st.session_state.model_choice = DEFAULT_CHAT_MODEL

# RAG Knowledge Base
BATTERY_KNOWLEDGE_BASE = {
    "soh_prediction": {
        "keywords": ["soh", "state of health", "battery health", "predict", "analysis"],
        "context": "State of Health (SOH) prediction using Linear Regression model trained on U1-U21 cell voltage data. The model achieves R² > 0.65 with RMSE < 0.04. SOH values below 0.6 indicate battery problems, while values above 0.6 indicate healthy batteries."
    },
    "battery_maintenance": {
        "keywords": ["maintenance", "care", "tips", "longevity", "extend", "lifespan"],
        "context": "Battery maintenance best practices: 1) Avoid deep discharges (keep above 20%), 2) Store at 40-60% charge, 3) Avoid extreme temperatures, 4) Use proper charging cycles, 5) Monitor cell voltage balance, 6) Regular health checks."
    },
    "battery_chemistry": {
        "keywords": ["chemistry", "lithium", "nmc", "lifepo4", "cobalt", "degradation"],
        "context": "Battery chemistry types: NMC (Nickel Manganese Cobalt) - high energy density, LiFePO4 - long cycle life, LiCoO2 - high voltage. Degradation mechanisms include SEI formation, lithium plating, and active material loss."
    },
    "recycling": {
        "keywords": ["recycle", "recycling", "sustainability", "environment", "disposal"],
        "context": "Battery recycling is crucial for sustainability. Retired batteries can be recycled to recover valuable materials like lithium, cobalt, and nickel. Proper recycling prevents environmental contamination and enables material recovery for new batteries."
    },
    "safety": {
        "keywords": ["safety", "danger", "fire", "explosion", "thermal", "protection"],
        "context": "Battery safety considerations: 1) Prevent thermal runaway, 2) Monitor temperature, 3) Use proper protection circuits, 4) Avoid physical damage, 5) Handle with care, 6) Follow manufacturer guidelines."
    },
    "model_performance": {
        "keywords": ["model", "performance", "accuracy", "metrics", "r2", "rmse"],
        "context": "Linear Regression model performance: R² score indicates model fit quality, RMSE shows prediction error, MAE measures average error. Cross-validation ensures model reliability. Feature importance shows which cells most influence SOH prediction."
    }
}

@st.cache_data(show_spinner=False)
def load_pulsebat_description():
    """Read the local PulseBat dataset description file if present."""
    description_path = Path(__file__).parent / "PulseBat Data Description.md"
    if not description_path.exists():
        return None
    try:
        return description_path.read_text(encoding="utf-8")
    except Exception:
        return None

def retrieve_relevant_context(user_query):
    """RAG: Retrieve relevant context based on user query"""
    query_lower = user_query.lower()
    relevant_contexts = []
    
    for topic, data in BATTERY_KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            if keyword in query_lower:
                relevant_contexts.append(data["context"])
                break
    
    return " ".join(relevant_contexts) if relevant_contexts else ""


def render_stat_cards(cards):
    """Render responsive stat cards for hero metrics"""
    card_html = '<div class="section-card"><div class="stat-grid">'
    for card in cards:
        card_html += textwrap.dedent(
            f"""
            <div class='stat-card'>
                <p class='stat-label'>{card['label']}</p>
                <p class='stat-value'>{card['value']}</p>
                <p class='stat-detail'>{card['detail']}</p>
            </div>
            """
        )
    card_html += "</div></div>"
    st.markdown(card_html, unsafe_allow_html=True)


def render_sidebar_stats(stats):
    """Render custom sidebar stat cards for improved readability."""
    if not stats:
        return

    cards_html = ["<div class='sidebar-stat-grid'>"]
    for stat in stats:
        detail = stat.get("detail")
        detail_html = f"<p class='sidebar-stat-detail'>{detail}</p>" if detail else ""
        cards_html.append(
            textwrap.dedent(
                f"""
                <div class='sidebar-stat-card'>
                    <p class='sidebar-stat-label'>{stat['label']}</p>
                    <p class='sidebar-stat-value'>{stat['value']}</p>
                    {detail_html}
                </div>
                """
            ).strip()
        )
    cards_html.append("</div>")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)


def build_linear_regression_context(
    model_results,
    df,
    u_cols,
    threshold,
    sort_method,
    figures=None,
    dataset_description=None,
):
    """Compile a comprehensive package of model, dataset, and visualization details for the AI assistant."""
    train_metrics = model_results["train_metrics"]
    test_metrics = model_results["test_metrics"]
    cv_scores = model_results["cv_scores"]
    cv_mean = float(np.mean(cv_scores)) if len(cv_scores) else float("nan")
    cv_std = float(np.std(cv_scores)) if len(cv_scores) else float("nan")
    train_metrics_serialized = {metric: float(value) for metric, value in train_metrics.items()}
    test_metrics_serialized = {metric: float(value) for metric, value in test_metrics.items()}

    feature_importance = (
        pd.DataFrame({
            "feature": model_results["feature_names"],
            "coefficient": model_results["model"].coef_,
        })
        .assign(abs_coeff=lambda df_: df_["coefficient"].abs())
        .sort_values("abs_coeff", ascending=False)
        .drop(columns="abs_coeff")
        .head(10)
    )
    coeff_lines = "\n".join(
        f"    - {row.feature}: {row.coefficient:.5f}"
        for row in feature_importance.itertuples()
    ) or "    - (coefficients unavailable)"

    # Provide compact context snippets to stay within API limits.
    snapshot_cols = [col for col in u_cols[:5]] + ["Pack_SOH"]
    snapshot_df = (
        df[snapshot_cols]
        .head(2)
        .round(4)
        .replace({np.nan: None})
    )
    snapshot_json = json.dumps(snapshot_df.to_dict(orient="records"), indent=2)

    dataset_metadata = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "u_columns": list(u_cols),
        "target_column": "Pack_SOH",
    }

    # These quick aggregates are enough for the assistant to talk about the dataset
    # without flooding the prompt with every row.
    dataset_summary = {
        "mean_soh": float(df["Pack_SOH"].mean()),
        "median_soh": float(df["Pack_SOH"].median()),
        "std_soh": float(df["Pack_SOH"].std()),
        "healthy_ratio": float(((df["Pack_SOH"] >= threshold).sum() / len(df)) if len(df) else 0.0),
    }

    coefficients = {
        feature: float(coef)
        for feature, coef in zip(model_results["feature_names"], model_results["model"].coef_)
    }

    scaler = model_results.get("scaler")
    scaler_payload = None
    if scaler is not None and hasattr(scaler, "mean_"):
        scaler_payload = {
            "mean": [float(x) for x in np.atleast_1d(scaler.mean_)],
            "scale": [float(x) for x in np.atleast_1d(getattr(scaler, "scale_", []))],
            "var": [float(x) for x in np.atleast_1d(getattr(scaler, "var_", []))] if hasattr(scaler, "var_") else None,
        }

    visualization_payload = []
    if figures:
        for label, fig in figures.items():
            if fig is None:
                continue
            title = None
            try:
                title = getattr(fig.layout.title, "text", None)
            except Exception:
                title = None
            visualization_payload.append(
                {
                    "label": label,
                    "title": title,
                    "has_data": True,
                }
            )

    summary_text = textwrap.dedent(
        f"""
        Linear Regression Context
        - Sorting Method: {sort_method}
        - Health Threshold: {threshold}
        - Train Metrics: R²={train_metrics['R²']:.3f}, RMSE={train_metrics['RMSE']:.4f}, MAE={train_metrics['MAE']:.4f}
        - Test Metrics: R²={test_metrics['R²']:.3f}, RMSE={test_metrics['RMSE']:.4f}, MAE={test_metrics['MAE']:.4f}
        - Cross-Validation R²: mean={cv_mean:.3f}, std={cv_std:.3f}

        Top Feature Coefficients:
{coeff_lines}

        Sample Pack Records (first 3 rows, subset of columns):
{snapshot_json}
        """
    ).strip()

    context_payload = {
        "summary": summary_text,
        "train_metrics": train_metrics_serialized,
        "test_metrics": test_metrics_serialized,
        "cv_scores": [float(x) for x in np.atleast_1d(cv_scores)],
        "threshold": threshold,
        "sort_method": sort_method,
        "model_artifact": {
            "type": "LinearRegression",
            "intercept": float(np.squeeze(model_results["model"].intercept_)),
            "coefficients": coefficients,
            "feature_names": list(model_results["feature_names"]),
            "scaler": scaler_payload,
        },
        "dataset": {
            "metadata": dataset_metadata,
            "summary": dataset_summary,
            "sample_rows": json.loads(snapshot_json),
        },
        "visualizations": visualization_payload,
    }

    if dataset_description:
        # Truncate the markdown so the downstream prompt stays under the LLM rate limits.
        trimmed_doc = dataset_description.strip()
        if len(trimmed_doc) > 1500:
            trimmed_doc = trimmed_doc[:1500] + "\n... (truncated)"
        context_payload["dataset"]["documentation_markdown"] = trimmed_doc

    context_json = json.dumps(context_payload, indent=2, default=_json_default_serializer)

    if len(context_json) > MAX_ASSISTANT_CONTEXT_CHARS:
        context_payload["dataset"].pop("sample_rows", None)
        context_json = json.dumps(context_payload, indent=2, default=_json_default_serializer)

    if len(context_json) > MAX_ASSISTANT_CONTEXT_CHARS:
        reduced_payload = {"summary": summary_text}
        context_json = json.dumps(reduced_payload, indent=2)

    return context_json

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_bytes):
    """Load and preprocess the PulseBat dataset"""
    # Streamlit gives us bytes, so we wrap them in BytesIO before handing to pandas.
    file_buffer = io.BytesIO(file_bytes)
    df = pd.read_csv(file_buffer)
    
    # Identify U1-U21 cell columns
    u_cols = [col for col in df.columns if col.startswith("U") and col[1:].isdigit()]
    u_cols = sorted(u_cols, key=lambda x: int(x[1:]))  # Sort U1, U2, ..., U21
    
    # Check if we have the expected U1-U21 columns
    if len(u_cols) != 21:
        st.warning(f"⚠️ Expected 21 cell columns (U1-U21), found {len(u_cols)}: {u_cols}")
    
    # Create pack SOH by aggregating individual cell SOH values.
    # Using mean as the primary aggregation method keeps the logic transparent.
    df["Pack_SOH_Mean"] = df[u_cols].mean(axis=1)
    df["Pack_SOH_Median"] = df[u_cols].median(axis=1)
    df["Pack_SOH_Min"] = df[u_cols].min(axis=1)
    df["Pack_SOH_Max"] = df[u_cols].max(axis=1)
    df["Pack_SOH_Std"] = df[u_cols].std(axis=1)
    
    # Use the existing SOH column as target (it represents pack SOH)
    if 'SOH' in df.columns:
        df["Pack_SOH"] = df["SOH"]
    else:
        df["Pack_SOH"] = df["Pack_SOH_Mean"]
    
    # Add data quality metrics
    df["Missing_Cells"] = df[u_cols].isnull().sum(axis=1)
    df["Cell_Range"] = df["Pack_SOH_Max"] - df["Pack_SOH_Min"]
    df["Cell_Variance"] = df[u_cols].var(axis=1)
    
    return df, u_cols

# -------------------------------
# Feature Preparation
# -------------------------------

def prepare_features(df, u_cols, sort_method):
    """Prepare features with different sorting techniques"""
    X = df[u_cols].copy()
    
    # Apply sorting if selected
    if sort_method == "Ascending":
        # Sorting each row lets us study voltage distribution independent of probe index.
        X = X.apply(lambda row: np.sort(row.values), axis=1, result_type="expand")
        X.columns = [f"U{i+1}_sorted_asc" for i in range(len(X.columns))]
    elif sort_method == "Descending":
        X = X.apply(lambda row: -np.sort(-row.values), axis=1, result_type="expand")
        X.columns = [f"U{i+1}_sorted_desc" for i in range(len(X.columns))]
    
    return X

# -------------------------------
# Linear Regression Model Training
# -------------------------------

@st.cache_data(show_spinner=False)
def train_linear_regression(df, u_cols, sort_method, test_size, cv_folds, random_state):
    """Train Linear Regression model with comprehensive evaluation"""
    
    # Prepare features
    X = prepare_features(df, u_cols, sort_method)
    y = df["Pack_SOH"]
    
    # Remove rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Feature scaling keeps the coefficients comparable across all U columns.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Cross-validation reports how consistent the model is across different splits.
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
    
    # Comprehensive metrics
    train_metrics = {
        "R²": r2_score(y_train, y_train_pred),
        "MSE": mean_squared_error(y_train, y_train_pred),
        "MAE": mean_absolute_error(y_train, y_train_pred),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred))
    }
    
    test_metrics = {
        "R²": r2_score(y_test, y_test_pred),
        "MSE": mean_squared_error(y_test, y_test_pred),
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred))
    }
    
    return {
        "model": model,
        "scaler": scaler,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_scores": cv_scores,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "feature_names": X.columns.tolist()
    }

# -------------------------------
# Battery Health Classification
# -------------------------------

def classify_battery_health(soh_value, threshold=0.6):
    """Classify battery health based on SOH threshold"""
    # The emoji copy keeps the UI friendly while we explain the threshold decision.
    if soh_value < threshold:
        return "⚠️ The battery has a problem.", "bad"
    else:
        return "✅ The battery is healthy.", "good"

# -------------------------------
# ChatGPT Integration
# -------------------------------

def ask_chatgpt_rag(prompt, context_data=None, api_key=None):
    """RAG-enhanced ChatGPT integration with battery expertise"""
    # RAG: Retrieve relevant context from knowledge base
    retrieved_context = retrieve_relevant_context(prompt)
    
    # Use internal API key if none provided
    api_key = get_effective_api_key(api_key)

    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        # Fallback response using RAG knowledge base
        if retrieved_context:
            return f"""## 🧠 RAG Knowledge Base Response

{retrieved_context}

---
*Note: The OpenAI API key is not configured. Please set it in the code to enable enhanced AI responses.*"""
        else:
            return """## ❌ OpenAI API key not configured

Please set `OPENAI_API_KEY` at the top of the app for AI-powered responses.

### Available without API key:
- Battery analysis and SOH predictions
- Model performance insights
- Quick action buttons

### With API key, you get:
- Advanced AI responses
- Detailed battery expertise
- Natural language conversations"""
    
    try:
        system_msg = """You are an expert AI assistant specializing in battery technology, state of health prediction, 
        and sustainable energy storage systems. You have deep knowledge of:
        - Battery chemistry and degradation mechanisms
        - Machine learning applications in battery management
        - Sustainable battery lifecycle management
        - Battery safety and maintenance best practices
        
        Provide detailed, technical, and actionable insights about battery technology. Use markdown formatting with headers, bullet points, and code blocks when appropriate."""
        
        # Combine retrieved context with current analysis context (model metadata, dataset stats, etc.).
        full_context = ""
        if retrieved_context:
            full_context += f"\n\nRetrieved Knowledge Base Context:\n{retrieved_context}"
        if context_data:
            full_context += f"\n\nCurrent Analysis Context:\n{context_data}"
        
        if full_context:
            system_msg += full_context
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=get_selected_model(),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        error_text = str(e)
        if "Incorrect API key" in error_text or "invalid_api_key" in error_text or "401" in error_text:
            return """## 🔐 OpenAI Authentication Error

The bundled API key was rejected by OpenAI. Update the `OPENAI_API_KEY` constant (or set the `OPENAI_API_KEY` environment variable) with a valid key that has access to your selected model, then rerun the app."""
        if "rate_limit" in error_text.lower() or "429" in error_text:
            return """## 🚦 OpenAI Rate Limit Reached

The last request exceeded the model's token-per-minute quota. Try again in a minute or reduce the dataset/model context (smaller CSV, lower sampling) so fewer tokens are sent."""
        return f"❌ AI Assistant Error: {error_text}"

def stream_chatgpt_rag(prompt, context_data=None, api_key=None):
    """Streaming RAG-enhanced ChatGPT integration"""
    # RAG: Retrieve relevant context from knowledge base
    retrieved_context = retrieve_relevant_context(prompt)
    
    # Use internal API key if none provided
    api_key = get_effective_api_key(api_key)

    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        # Fallback response using RAG knowledge base
        if retrieved_context:
            response = f"""## 🧠 RAG Knowledge Base Response

{retrieved_context}

---
*Note: The OpenAI API key is not configured. Please set it in the code to enable enhanced AI responses.*"""
        else:
            response = """## ❌ OpenAI API key not configured

Please set `OPENAI_API_KEY` at the top of the app for AI-powered responses.

### Available without API key:
- Battery analysis and SOH predictions
- Model performance insights
- Quick action buttons

### With API key, you get:
- Advanced AI responses
- Detailed battery expertise
- Natural language conversations"""
        
        # Simulate streaming for fallback
        words = response.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3])
            yield chunk + " "
            time.sleep(0.05)
        return
    
    try:
        system_msg = """You are an expert AI assistant specializing in battery technology, state of health prediction, 
        and sustainable energy storage systems. You have deep knowledge of:
        - Battery chemistry and degradation mechanisms
        - Machine learning applications in battery management
        - Sustainable battery lifecycle management
        - Battery safety and maintenance best practices
        
        Provide detailed, technical, and actionable insights about battery technology. Use markdown formatting with headers, bullet points, and code blocks when appropriate."""
        
        # Combine retrieved context with current analysis context to keep responses grounded.
        full_context = ""
        if retrieved_context:
            full_context += f"\n\nRetrieved Knowledge Base Context:\n{retrieved_context}"
        if context_data:
            full_context += f"\n\nCurrent Analysis Context:\n{context_data}"
        
        if full_context:
            system_msg += full_context
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=get_selected_model(),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=600,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        error_text = str(e)
        if "Incorrect API key" in error_text or "invalid_api_key" in error_text or "401" in error_text:
            yield """## 🔐 OpenAI Authentication Error

The embedded OpenAI key is invalid. Update `OPENAI_API_KEY` in the code (or export `OPENAI_API_KEY`) with a working key, then restart the app."""
        elif "rate_limit" in error_text.lower() or "429" in error_text:
            yield """## 🚦 OpenAI Rate Limit Reached

This request exceeded the model's token quota. Please retry shortly or reduce the context size (smaller dataset upload, fewer advanced settings) before asking again."""
        else:
            yield f"❌ AI Assistant Error: {error_text}"

def show_feedback_controls(message_index):
    """Shows the feedback control for assistant messages"""
    # We keep this lightweight so it can sit under every assistant response without clutter.
    st.write("")
    
    with st.popover("How did I do?"):
        form_key = f"feedback-{message_index}"
        with st.form(key=form_key, border=False):
            rating = st.slider("Rate this response", 1, 5, 5, key=f"rating-{message_index}")
            details = st.text_area("More information (optional)", key=f"details-{message_index}")
            include_history = st.checkbox(
                "Include chat history with my feedback", value=True, key=f"history-{message_index}"
            )

            if st.form_submit_button("Send feedback"):
                # In a full deployment, this is where we'd forward feedback to an analytics service.
                st.success("Thank you for your feedback! 🎉")

# -------------------------------
# Visualizations
# -------------------------------

def create_visualizations(model_results, df):
    """Create comprehensive visualizations"""
    # We separate plotting into its own helper so tabs can re-use the same figures.
    
    # 1. Predicted vs Actual SOH compares the regression fit on train vs test.
    fig1 = go.Figure()
    
    # Training data
    fig1.add_trace(go.Scatter(
        x=model_results["y_train"], 
        y=model_results["y_train_pred"],
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', opacity=0.6, size=6)
    ))
    
    # Test data
    fig1.add_trace(go.Scatter(
        x=model_results["y_test"], 
        y=model_results["y_test_pred"],
        mode='markers',
        name='Test Data',
        marker=dict(color='red', opacity=0.6, size=6)
    ))
    
    # Perfect prediction line
    min_val = min(model_results["y_train"].min(), model_results["y_test"].min())
    max_val = max(model_results["y_train"].max(), model_results["y_test"].max())
    fig1.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='green', dash='dash', width=2)
    ))
    
    fig1.update_layout(
        title="Linear Regression: Predicted vs Actual SOH",
        xaxis_title="Actual SOH",
        yaxis_title="Predicted SOH",
        template="plotly_white",
        width=800,
        height=600
    )
    
    # 2. Residual plot highlights any heteroscedasticity or bias.
    residuals_train = model_results["y_train"] - model_results["y_train_pred"]
    residuals_test = model_results["y_test"] - model_results["y_test_pred"]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=model_results["y_train_pred"], 
        y=residuals_train,
        mode='markers',
        name='Training Residuals',
        marker=dict(color='blue', opacity=0.6)
    ))
    fig2.add_trace(go.Scatter(
        x=model_results["y_test_pred"], 
        y=residuals_test,
        mode='markers',
        name='Test Residuals',
        marker=dict(color='red', opacity=0.6)
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="green")
    fig2.update_layout(
        title="Residuals Analysis",
        xaxis_title="Predicted SOH",
        yaxis_title="Residuals",
        template="plotly_white",
        width=600,
        height=400
    )
    
    # 3. Overall SOH distribution gives the user a quick feel for the dataset health spread.
    fig3 = px.histogram(
        df, x="Pack_SOH", 
        title="Battery Pack SOH Distribution",
        nbins=30,
        template="plotly_white",
        width=600,
        height=400
    )
    
    return fig1, fig2, fig3

# -------------------------------
# Main Application Logic
# -------------------------------

# Main application flow
if uploaded_file is not None:
    # Everything below this branch is data-driven, so we bail early if no CSV is present.
    try:
        file_bytes = uploaded_file.getvalue()
        with st.spinner("🔄 Loading and preprocessing dataset..."):
            df, u_cols = load_and_preprocess_data(file_bytes)
        st.success("✅ PulseBat dataset loaded successfully!")
    except Exception as exc:
        st.error(f"Unable to process the uploaded file: {exc}")
        st.stop()

    if not u_cols:
        st.error("No valid U1-U21 cell columns were found in the dataset. Please upload a valid PulseBat file.")
        st.stop()
    if df.empty:
        st.error("The uploaded dataset has no rows to analyze. Please provide a populated dataset.")
        st.stop()

    pack_soh_series = df["Pack_SOH"].dropna()
    healthy_count = (pack_soh_series >= threshold).sum()
    unhealthy_count = len(df) - healthy_count
    healthy_ratio = (healthy_count / len(df) * 100) if len(df) else 0
    avg_pack_soh = pack_soh_series.mean()
    soh_min = pack_soh_series.min()
    soh_max = pack_soh_series.max()
    median_soh = pack_soh_series.median()
    soh_std = pack_soh_series.std()
    avg_cell_spread = df["Cell_Range"].fillna(0).mean()
    missing_rows = int(df[u_cols].isnull().any(axis=1).sum())
    total_missing_entries = int(df[u_cols].isnull().sum().sum())
    rows_without_missing = len(df) - missing_rows
    duplicate_rows = int(df.duplicated().sum())
    zero_variance_cells = int((df[u_cols].std() == 0).sum())

    stat_cards = [
        {"label": "Total Samples", "value": f"{len(df):,}", "detail": "records analyzed"},
        {"label": "Average SOH", "value": f"{avg_pack_soh:.3f}", "detail": "mean pack health"},
        {"label": "Healthy Packs", "value": f"{healthy_ratio:.1f}%", "detail": f"{healthy_count:,} above threshold"},
        {"label": "Voltage Spread", "value": f"{avg_cell_spread:.3f}", "detail": "avg cell variance"}
    ]
    render_stat_cards(stat_cards)
    visualization_figures = {}
    dataset_description_markdown = load_pulsebat_description()
    
    # Prepare model insights once (cached)
    try:
        with st.spinner("🔄 Preparing linear regression insights..."):
            model_results = train_linear_regression(
                df, u_cols, sort_method, test_size, cv_folds, random_state
            )
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

    # Update sidebar stats
    with st.sidebar:
        st.markdown("<p class='sidebar-section-heading'>📊 Dataset Statistics</p>", unsafe_allow_html=True)
        sidebar_stats = [
            {"label": "Total Samples", "value": f"{len(df):,}", "detail": "records analyzed"},
            {"label": "Cell Feature Columns", "value": f"{len(u_cols)}", "detail": "U-series inputs"},
            {"label": "Median Pack SOH", "value": f"{median_soh:.3f}"},
            {"label": "SOH Std Dev", "value": f"{soh_std:.3f}"},
            {"label": "SOH Range", "value": f"{soh_min:.3f} – {soh_max:.3f}"},
            {"label": "Rows w/ Missing Cells", "value": f"{missing_rows:,}"},
            {"label": "Zero-Variance Cell Columns", "value": f"{zero_variance_cells}"},
        ]
        render_sidebar_stats(sidebar_stats)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🤖 Linear Regression", "📈 Visualizations", "💬 AI Assistant"])
    
    with tab1:
        st.markdown("## 📊 Comprehensive Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Overview")
            # Toggle allows laptops with limited VRAM to avoid rendering all 670 rows at once.
            show_full_dataset = st.toggle("Show entire dataset", value=False, key="show_full_dataset")
            if show_full_dataset:
                st.dataframe(df, use_container_width=True, height=420)
            else:
                max_preview = min(len(df), 200)
                preview_rows = st.slider(
                    "Rows to preview",
                    min_value=5,
                    max_value=max_preview,
                    value=min(10, max_preview),
                    step=5,
                    key="preview_row_slider",
                )
                st.dataframe(df.head(preview_rows), use_container_width=True, height=420)
            st.caption("Toggle to inspect the entire CSV or fine-tune the preview window.")
            
            st.markdown("### Data Quality Report")
            quality_metrics = {
                "Complete Rows": rows_without_missing,
                "Duplicate Rows": duplicate_rows,
                "Total Missing Cells": total_missing_entries,
                "Avg Cell Voltage": df[u_cols].mean().mean(),
                "Max Cell Range": df["Cell_Range"].max(),
                "Avg Cell Variance": df["Cell_Variance"].mean(),
            }
            for metric, value in quality_metrics.items():
                st.metric(metric, f"{value:.3f}" if isinstance(value, float) else value)
        
        with col2:
            st.markdown("### SOH Distribution Analysis")
            fig_hist = px.histogram(df, x="Pack_SOH", nbins=30, 
                                  title="Battery Pack SOH Distribution",
                                  template="plotly_white")
            visualization_figures["dataset_soh_distribution"] = fig_hist
            st.plotly_chart(fig_hist, use_container_width=True)
            
            health_df = pd.DataFrame({
                "status": ["Healthy", "Problematic"],
                "count": [healthy_count, unhealthy_count],
            })
            fig_pie = px.pie(
                health_df,
                values="count",
                names="status",
                title=f"Battery Health Classification (Threshold: {threshold})",
                color="status",
                color_discrete_map={"Healthy": "#4CAF50", "Problematic": "#f44336"}
            )
            visualization_figures["health_classification_breakdown"] = fig_pie
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Cell correlation analysis
        st.markdown("### 🔥 Cell Correlation Matrix")
        corr_matrix = df[u_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Battery Cell Correlation Heatmap",
            template="plotly_white",
            aspect="auto"
        )
        visualization_figures["cell_correlation_heatmap"] = fig_heatmap
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.markdown("## 🤖 Linear Regression Model Training")
        st.success("✅ Linear Regression model ready! Cached for instant reuse.")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📈 Training Performance")
            train_metrics = model_results["train_metrics"]
            for metric, value in train_metrics.items():
                st.metric(f"Train {metric}", f"{value:.4f}")
        
        with col2:
            st.markdown("### 🎯 Test Performance")
            test_metrics = model_results["test_metrics"]
            for metric, value in test_metrics.items():
                st.metric(f"Test {metric}", f"{value:.4f}")
        
        with col3:
            st.markdown("### 🔄 Cross-Validation")
            cv_mean = model_results["cv_scores"].mean()
            cv_std = model_results["cv_scores"].std()
            st.metric("CV R² Mean", f"{cv_mean:.4f}")
            st.metric("CV R² Std", f"{cv_std:.4f}")
        
        # Model interpretation
        st.markdown("### 🔍 Model Interpretation")
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'Feature': model_results["feature_names"],
            'Coefficient': model_results["model"].coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig_importance = px.bar(
            feature_importance.head(15), 
            x='Coefficient', y='Feature',
            title="Top 15 Feature Coefficients (Linear Regression)",
            template="plotly_white",
            orientation='h'
        )
        visualization_figures["feature_importance_bar"] = fig_importance
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model summary
        st.markdown("### 📋 Model Summary")
        st.markdown(f"""
        **Model Type:** Linear Regression  
        **Features:** {len(model_results["feature_names"])} cell voltages (U1-U21)  
        **Preprocessing:** {sort_method} sorting  
        **Train/Test Split:** {int((1-test_size)*100)}%/{int(test_size*100)}%  
        **Cross-Validation:** {cv_folds} folds  
        **Test R² Score:** {test_metrics['R²']:.4f}  
        **Test RMSE:** {test_metrics['RMSE']:.4f}  
        """)
    
    with tab3:
        st.markdown("## 📈 Model Performance Visualizations")
        
        # Create visualizations
        fig1, fig2, fig3 = create_visualizations(model_results, df)
        visualization_figures["predicted_vs_actual"] = fig1
        visualization_figures["residuals_scatter"] = fig2
        visualization_figures["soh_distribution_model_tab"] = fig3
        
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(fig3, use_container_width=True)
        
        # Additional analysis
        st.markdown("### 📊 Additional Analysis")
        
        # Residuals distribution
        residuals = model_results["y_test"] - model_results["y_test_pred"]
        fig_residuals = px.histogram(
            x=residuals,
            title="Distribution of Residuals",
            nbins=30,
            template="plotly_white"
        )
        visualization_figures["residuals_histogram"] = fig_residuals
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with tab4:
        # Battery AI Assistant - Modern Chat Interface
        st.markdown("## 🔋 Battery AI Assistant")
        
        # Battery-specific suggestions
        BATTERY_SUGGESTIONS = {
            "🔋 What is SOH prediction?": "What is State of Health (SOH) prediction and how does it work with battery cells?",
            "🛠️ Battery maintenance tips": "Give me comprehensive battery maintenance and care tips for longevity",
            "⚗️ Battery chemistry explained": "Explain different battery chemistries like NMC, LiFePO4 and their characteristics",
            "♻️ Battery recycling importance": "Why is battery recycling important for sustainability?",
            "🛡️ Battery safety guidelines": "What are the key safety considerations when working with batteries?",
            "📊 Model performance analysis": "Analyze the current Linear Regression model performance and provide insights"
        }
        
        # Initialize chat history
        if "battery_messages" not in st.session_state:
            st.session_state.battery_messages = []
        
        # Check if user has interacted
        user_just_asked_initial_question = (
            "initial_battery_question" in st.session_state and st.session_state.initial_battery_question
        )
        
        user_just_clicked_suggestion = (
            "selected_battery_suggestion" in st.session_state and st.session_state.selected_battery_suggestion
        )
        
        user_first_interaction = (
            user_just_asked_initial_question or user_just_clicked_suggestion
        )
        
        has_message_history = (
            "battery_messages" in st.session_state and len(st.session_state.battery_messages) > 0
        )
        
        # Show initial interface when no interaction
        if not user_first_interaction and not has_message_history:
            st.session_state.battery_messages = []
            
            # RAG Status
            st.markdown(
                """
                <div class="section-card">
                    <h3 class="section-title">🚀 Start a conversation</h3>
                    <p class="section-subtitle">Pick a curated topic or type your own question to tap into the battery expert assistant.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("🧠 **RAG System Active**: This chatbot uses Retrieval-Augmented Generation with a specialized battery knowledge base for enhanced responses.")
            
            with st.container():
                st.chat_input("Ask about batteries, SOH prediction, or maintenance...", key="initial_battery_question")
                
                selected_suggestion = st.pills(
                    label="Examples",
                    label_visibility="collapsed",
                    options=list(BATTERY_SUGGESTIONS.keys()),
                    key="selected_battery_suggestion",
                )
            
            # Knowledge Base Topics
            with st.expander("📚 Available Knowledge Base Topics"):
                st.markdown("""
                **The RAG system can provide expert insights on:**
                - 🔋 **SOH Prediction**: State of Health analysis and prediction methods
                - 🛠️ **Battery Maintenance**: Care tips and longevity best practices  
                - ⚗️ **Battery Chemistry**: NMC, LiFePO4, degradation mechanisms
                - ♻️ **Recycling & Sustainability**: Environmental impact and recycling processes
                - 🛡️ **Safety**: Thermal management and protection measures
                - 📊 **Model Performance**: Accuracy metrics and evaluation methods
                """)
            
            st.stop()
        
        # Show chat input at the bottom when conversation has started
        user_message = st.chat_input("Ask a follow-up...")
        
        if not user_message:
            if user_just_asked_initial_question:
                user_message = st.session_state.initial_battery_question
            if user_just_clicked_suggestion:
                user_message = BATTERY_SUGGESTIONS[st.session_state.selected_battery_suggestion]
        
        # Restart button
        col1, col2 = st.columns([1, 4])
        with col1:
            def clear_battery_conversation():
                st.session_state.battery_messages = []
                st.session_state.initial_battery_question = None
                st.session_state.selected_battery_suggestion = None
            
            st.button(
                "Restart",
                icon="🔄",
                on_click=clear_battery_conversation,
            )
        
        # Display chat messages from history
        for i, message in enumerate(st.session_state.battery_messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.container()  # Fix ghost message bug
                st.markdown(message["content"])
                
                # Add feedback controls for assistant messages
                if message["role"] == "assistant":
                    show_feedback_controls(i)
        
        if user_message:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_message)
            
            # Display assistant response
            with st.chat_message("assistant"):
                # Check if it's a prediction request
                if any(keyword in user_message.lower() for keyword in ["soh", "predict", "analyze", "health"]):
                    with st.spinner("Analyzing battery data..."):
                        # Select a random sample for prediction to simulate live pack analysis.
                        sample_idx = np.random.randint(0, len(df))
                        sample_data = df.iloc[sample_idx]
                        
                        X_sample = prepare_features(pd.DataFrame([sample_data]), u_cols, sort_method)
                        X_sample_scaled = model_results["scaler"].transform(X_sample)
                        prediction = model_results["model"].predict(X_sample_scaled)[0]
                        
                        health_status, health_class = classify_battery_health(prediction, threshold)
                        
                        response = f"""## 🔮 Battery SOH Prediction

**Predicted Pack SOH:** **{prediction:.3f}**  
**Health Status:** {health_status}  

### 📈 Model Details
- **Algorithm:** Linear Regression
- **Test R² Score:** {model_results['test_metrics']['R²']:.3f}
- **Test RMSE:** {model_results['test_metrics']['RMSE']:.3f}
- **Health Threshold:** {threshold}

### 📊 Dataset Context
- **Total Samples:** {len(df)} batteries
- **SOH Range:** {df['Pack_SOH'].min():.3f} - {df['Pack_SOH'].max():.3f}
- **Sorting Method:** {sort_method}

The model uses U1-U21 cell voltage data to predict overall battery pack health. Values above {threshold} indicate healthy batteries suitable for continued use."""
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add feedback controls for prediction response
                    show_feedback_controls(len(st.session_state.battery_messages))
                else:
                    # Use streaming RAG system for other questions
                    with st.spinner("Researching battery knowledge..."):
                        context_data = build_linear_regression_context(
                            model_results,
                            df,
                            u_cols,
                            threshold,
                            sort_method,
                            figures=visualization_figures,
                            dataset_description=dataset_description_markdown,
                        )
                    
                    # Stream the response
                    response = st.write_stream(stream_chatgpt_rag(user_message, context_data))
                    
                    # Add feedback controls for streaming response
                    show_feedback_controls(len(st.session_state.battery_messages))
                
                # Add messages to chat history
                st.session_state.battery_messages.append({"role": "user", "content": user_message})
                st.session_state.battery_messages.append({"role": "assistant", "content": response})

else:
    st.markdown("""
    ## 🚀 Welcome to Battery Pack SOH Prediction Platform
    
    ### Overview
    
    This platform delivers a comprehensive battery State of Health (SOH) prediction system using Linear Regression
    plus an AI-powered assistant for expert analysis.
    
    ### Key Features
    
    🔋 **Battery SOH Prediction**: Upload your PulseBat dataset to predict State of Health using U1-U21 cell data
    
    🤖 **Linear Regression Model**: Advanced machine learning model with comprehensive evaluation metrics
    
    📊 **Data Preprocessing**: Multiple sorting techniques (None, Ascending, Descending) for cell data analysis
    
    ⚡ **Health Classification**: Configurable threshold-based classification (default: 0.6)
    
    📈 **Rich Visualizations**: Interactive charts showing predicted vs actual SOH, residuals, and distributions
    
    💬 **AI Assistant**: Expert assistant for battery insights and maintenance tips
    
    ### Getting Started:
    1. 📁 Upload your PulseBat CSV file using the sidebar
    2. ⚙️ Configure your model settings (sorting method, threshold, etc.)
    3. 🎯 Explore the data analysis, model training, and visualizations
    4. 💭 Chat with our AI assistant for insights and battery tips
    
    ---
    
    **Need sample data for testing? Reach out to the project team for an anonymized PulseBat extract.**
    """)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔬 Linear Regression
        - U1-U21 cell feature extraction
        - Multiple sorting techniques
        - Cross-validation
        - Comprehensive metrics
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Rich Analytics  
        - Interactive visualizations
        - Correlation analysis
        - Distribution plots
        - Quality reports
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 AI Assistant
        - Battery expertise
        - Model insights  
        - Maintenance tips
        - Real-time analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Battery Pack SOH Prediction Platform</strong></p>
    <p>Linear Regression, rich analytics, and an AI assistant.</p>
</div>
""", unsafe_allow_html=True)