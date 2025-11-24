# Battery Pack SOH Prediction & AI Assistant Platform
# SOFE3370 Final Project - Group 18
# Pranav Ashok Chaudhari, Tarun Modekurty, Leela Alagala, Hannah Albi

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
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Internal API Configuration
# -------------------------------
# NOTE: The API key is hardcoded for convenience per user request.
# Do NOT commit real secrets to public repos.
OPENAI_API_KEY = "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"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Battery Pack SOH Prediction Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for exceptional UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
html, body, [class*="css"] { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Smooth animations */
* { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }

/* Main Header - Stunning gradient with animation */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 100%);
    background-size: 300% 300%;
    animation: gradient-shift 15s ease infinite;
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    color: #ffffff;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3), 0 0 0 1px rgba(255,255,255,0.1) inset;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-header h1 { 
    margin: 0; 
    font-weight: 800; 
    letter-spacing: -1px; 
    font-size: 2.5rem;
    text-shadow: 0 2px 20px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}

.main-header h3 { 
    margin: 0.5rem 0; 
    font-weight: 500; 
    color: rgba(255,255,255,0.95); 
    font-size: 1.3rem;
    position: relative;
    z-index: 1;
}

.main-header p { 
    margin: 0.5rem 0 0 0; 
    color: rgba(255,255,255,0.9);
    font-size: 1.05rem;
    position: relative;
    z-index: 1;
}

/* Glassmorphism Cards */
.uploader-card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.5rem;
}

.sidebar-section { 
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    padding: 1.25rem;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.06);
    margin-bottom: 1.5rem;
}

/* Enhanced Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin: 0.75rem 0;
    box-shadow: 0 10px 30px rgba(102,126,234,0.25);
    border: 1px solid rgba(255,255,255,0.1);
    transform: translateY(0);
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 40px rgba(102,126,234,0.35);
}

/* Health Status Cards */
.health-good { 
    background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    color: white;
    padding: 1.2rem;
    border-radius: 14px;
    text-align: center;
    margin: 0.75rem 0;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    border: 1px solid rgba(255,255,255,0.2);
    font-weight: 600;
}

.health-bad { 
    background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
    color: white;
    padding: 1.2rem;
    border-radius: 14px;
    text-align: center;
    margin: 0.75rem 0;
    box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
    border: 1px solid rgba(255,255,255,0.2);
    font-weight: 600;
}

/* Chat Messages */
.chat-message { 
    padding: 1.2rem;
    border-radius: 14px;
    margin: 0.75rem 0;
    background-color: #ffffff;
    color: #1e293b;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

.chat-message-user { 
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-left: 5px solid #3b82f6;
    color: #1e40af;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15);
}

.chat-message-assistant { 
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    border-left: 5px solid #a855f7;
    color: #6b21a8;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.15);
}

/* Info Box */
.info-box { 
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 5px solid #3b82f6;
    padding: 1.2rem;
    margin: 1.5rem 0;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

/* Enhanced Buttons */
.stButton>button { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    border: none;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    cursor: pointer;
    text-transform: none;
    letter-spacing: 0.3px;
}

.stButton>button:hover { 
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

.stButton>button:active {
    transform: translateY(0);
}

/* Tabs Enhancement */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f8fafc;
    padding: 0.5rem;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: transparent;
    border-radius: 10px;
    color: #64748b;
    font-weight: 600;
    padding: 0 1.5rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* Metrics Enhancement */
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
}

[data-testid="stMetricLabel"] {
    font-size: 0.95rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* DataFrame Enhancement */
.stDataFrame, .stAgGrid { 
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
}

/* Sidebar Enhancement */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}

/* Input Fields */
input, textarea, select {
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

/* Slider Enhancement */
.stSlider > div > div > div > div {
    background-color: #667eea !important;
}

/* Expander Enhancement */
.streamlit-expanderHeader {
    background-color: #f8fafc;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 600;
    border: 1px solid #e2e8f0;
}

.streamlit-expanderHeader:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    border: 2px dashed #cbd5e1;
}

[data-testid="stFileUploader"]:hover {
    border-color: #667eea;
    background-color: #f8fafc;
}

/* Pills (Suggestions) */
[data-testid="stHorizontalBlock"] button {
    background-color: #f1f5f9;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    padding: 0.5rem 1rem;
    font-weight: 500;
    color: #475569;
}

[data-testid="stHorizontalBlock"] button:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: transparent;
    transform: translateY(-2px);
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Loading Spinner */
.stSpinner > div {
    border-top-color: #667eea !important;
}

/* Success/Error/Warning Messages */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border-left-width: 5px;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header {
        padding: 2rem 1.5rem;
    }
    
    .main-header h1 {
        font-size: 1.8rem;
    }
    
    .main-header h3 {
        font-size: 1.1rem;
    }
}

/* Plotly Chart Enhancement */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
}

/* Form Elements */
.stForm {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
}

/* Popover Enhancement */
[data-testid="stPopover"] {
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(15, 23, 42, 0.15);
}

/* Chat Input Enhancement */
[data-testid="stChatInput"] {
    border-radius: 12px;
    border: 2px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
}

[data-testid="stChatInput"]:focus-within {
    border-color: #667eea;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
}

/* Select Box Enhancement */
.stSelectbox > div > div {
    border-radius: 10px;
    border: 2px solid #e2e8f0;
}

.stSelectbox > div > div:hover {
    border-color: #cbd5e1;
}

/* Number Input Enhancement */
.stNumberInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #e2e8f0;
}

/* Text Area Enhancement */
.stTextArea > div > div > textarea {
    border-radius: 10px;
    border: 2px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>⚡ Battery Pack SOH Prediction Platform</h1>
    <h3>AI-Powered Battery Health Assessment</h3>
    <p>State-of-the-art analytics, predictions, and an expert assistant — all in one app.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration (styled)
# -------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-section'><h3 style='margin:0 0 0.75rem 0; color:#1e293b; font-weight:700;'>⚙️ Configuration Panel</h3></div>", unsafe_allow_html=True)

    # File Upload (styled card)
    st.markdown("<div class='uploader-card'><h4 style='margin:0 0 0.75rem 0; color:#1e293b; font-weight:600;'>📁 Upload PulseBat Dataset</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", 
        type=["csv"],
        help="Upload your PulseBat dataset CSV file"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Model Settings (grouped)
    st.markdown("<div class='sidebar-section'><h4 style='margin:0 0 0.75rem 0; color:#1e293b; font-weight:600;'>🔧 Model Configuration</h4>", unsafe_allow_html=True)

    # Preprocessing Options
    sort_method = st.selectbox(
        "Cell Sorting Method:",
        ["None", "Ascending", "Descending"],
        help="Sort U1–U21 cell values for pattern analysis"
    )

    # SOH Threshold
    threshold = st.slider(
        "SOH Health Threshold:",
        min_value=0.0, max_value=1.0, value=0.6, step=0.01,
        help="Threshold below which battery is considered unhealthy"
    )

    # Advanced Settings
    with st.expander("Advanced Settings"):
        test_size = st.slider("Train/Test Split", 0.1, 0.5, 0.2)
        cv_folds = st.number_input("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random State", 1, 100, 42)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# -------------------------------
# Chat Memory and RAG System
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

@st.cache_data
def load_and_preprocess_data(file):
    """Load and preprocess the PulseBat dataset"""
    with st.spinner("Loading and preprocessing data..."):
        df = pd.read_csv(file)
        
        # Identify U1-U21 cell columns
        u_cols = [col for col in df.columns if col.startswith("U") and col[1:].isdigit()]
        u_cols = sorted(u_cols, key=lambda x: int(x[1:]))  # Sort U1, U2, ..., U21
        
        # Check if we have the expected U1-U21 columns
        if len(u_cols) != 21:
            st.warning(f"Expected 21 cell columns (U1-U21), found {len(u_cols)}: {u_cols}")
        
        # Create pack SOH by aggregating individual cell SOH values
        # Using mean as the primary aggregation method
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

@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate U1-U21 columns with realistic battery cell voltages
    data = {}
    for i in range(1, 22):
        # Simulate realistic battery cell voltages with some correlation
        base_voltage = np.random.normal(3.5, 0.1, n_samples)
        noise = np.random.normal(0, 0.02, n_samples)
        data[f'U{i}'] = np.clip(base_voltage + noise, 3.0, 4.0)
    
    # Generate SOH based on cell voltages with realistic patterns
    df = pd.DataFrame(data)
    u_cols = [f'U{i}' for i in range(1, 22)]
    
    # Create realistic SOH values
    cell_avg = df[u_cols].mean(axis=1)
    soh_base = (cell_avg - 3.0) / 1.0  # Normalize to 0-1 range
    soh_noise = np.random.normal(0, 0.05, n_samples)
    df["SOH"] = np.clip(soh_base + soh_noise, 0.1, 1.0)
    
    # Add metadata
    df["ID"] = range(1, n_samples + 1)
    df["Mat"] = np.random.choice(['NMC', 'LiCoO2', 'LiFePO4'], n_samples)
    
    return df, u_cols

# -------------------------------
# Feature Preparation
# -------------------------------

def prepare_features(df, u_cols, sort_method):
    """Prepare features with different sorting techniques"""
    X = df[u_cols].copy()
    
    # Apply sorting if selected
    if sort_method == "Ascending":
        X = X.apply(lambda row: np.sort(row.values), axis=1, result_type="expand")
        X.columns = [f"U{i+1}_sorted_asc" for i in range(len(X.columns))]
    elif sort_method == "Descending":
        X = X.apply(lambda row: -np.sort(-row.values), axis=1, result_type="expand")
        X.columns = [f"U{i+1}_sorted_desc" for i in range(len(X.columns))]
    
    return X

# -------------------------------
# Linear Regression Model Training
# -------------------------------

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
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Cross-validation
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
    if soh_value < threshold:
        return "The battery has a problem.", "bad"
    else:
        return "The battery is healthy.", "good"

# -------------------------------
# ChatGPT Integration
# -------------------------------

def ask_chatgpt_rag(prompt, context_data=None, api_key=None):
    """RAG-enhanced ChatGPT integration with battery expertise"""
    # RAG: Retrieve relevant context from knowledge base
    retrieved_context = retrieve_relevant_context(prompt)
    
    # Use internal API key if none provided
    if api_key is None:
        api_key = OPENAI_API_KEY

        if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
            # Fallback response using RAG knowledge base
            if retrieved_context:
                return f"""## RAG Knowledge Base Response

{retrieved_context}

---
*Note: The OpenAI API key is not configured. Please set it in the code to enable enhanced AI responses.*"""
            else:
                return """## OpenAI API key not configured

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
        
        # Combine retrieved context with current analysis context
        full_context = ""
        if retrieved_context:
            full_context += f"\n\nRetrieved Knowledge Base Context:\n{retrieved_context}"
        if context_data:
            full_context += f"\n\nCurrent Analysis Context:\n{context_data}"
        
        if full_context:
            system_msg += full_context
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Assistant Error: {str(e)}"

def stream_chatgpt_rag(prompt, context_data=None, api_key=None):
    """Streaming RAG-enhanced ChatGPT integration"""
    # RAG: Retrieve relevant context from knowledge base
    retrieved_context = retrieve_relevant_context(prompt)
    
    # Use internal API key if none provided
    if api_key is None:
        api_key = OPENAI_API_KEY

    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        # Fallback response using RAG knowledge base
        if retrieved_context:
            response = f"""## RAG Knowledge Base Response

{retrieved_context}

---
*Note: The OpenAI API key is not configured. Please set it in the code to enable enhanced AI responses.*"""
        else:
            response = """## OpenAI API key not configured

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
        
        # Combine retrieved context with current analysis context
        full_context = ""
        if retrieved_context:
            full_context += f"\n\nRetrieved Knowledge Base Context:\n{retrieved_context}"
        if context_data:
            full_context += f"\n\nCurrent Analysis Context:\n{context_data}"
        
        if full_context:
            system_msg += full_context
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"AI Assistant Error: {str(e)}"

def show_feedback_controls(message_index):
    """Shows the feedback control for assistant messages"""
    st.write("")
    
    with st.popover("How did I do?"):
        with st.form(key=f"feedback-{message_index}", border=False):
            with st.container(gap=None):
                st.markdown(":small[Rating]")
                rating = st.feedback(options="stars")

            details = st.text_area("More information (optional)")

            if st.checkbox("Include chat history with my feedback", True):
                relevant_history = st.session_state.battery_messages[:message_index]
            else:
                relevant_history = []

            ""  # Add some space

            if st.form_submit_button("Send feedback"):
                st.success("Thank you for your feedback!")
                # TODO: Submit feedback here!

# -------------------------------
# Visualizations
# -------------------------------

def create_visualizations(model_results, df):
    """Create comprehensive visualizations"""
    
    # 1. Predicted vs Actual SOH
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
    
    # 2. Residuals Plot
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
    
    # 3. SOH Distribution
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

# Note: Demo mode removed. Users must upload a dataset to proceed.
# To re-enable demo mode, re-add a button that sets `uploaded_file = "demo"`.

# Main application flow
if uploaded_file is not None:
    # Load data (demo or uploaded)
    if uploaded_file == "demo":
        df, u_cols = generate_sample_data()
        st.success("Demo dataset loaded (1000 synthetic battery samples)")
    else:
        df, u_cols = load_and_preprocess_data(uploaded_file)
        st.success("PulseBat dataset loaded successfully!")
    
    # Update sidebar stats
    with st.sidebar:
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0 0 1rem 0; color:#1e293b; font-weight:600;'>📊 Dataset Statistics</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📦 Total Samples", f"{len(df):,}")
            st.metric("🔋 Battery Cells", len(u_cols))
            st.metric("✅ Healthy", f"{(df['Pack_SOH'] >= threshold).sum():,}")
        with col2:
            st.metric("📈 Avg SOH", f"{df['Pack_SOH'].mean():.3f}")
            st.metric("📊 SOH Range", f"{df['Pack_SOH'].min():.2f}-{df['Pack_SOH'].max():.2f}")
            st.metric("⚠️ Problems", f"{(len(df) - (df['Pack_SOH'] >= threshold).sum()):,}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Linear Regression", "Visualizations", "AI Assistant"])
    
    with tab1:
        st.markdown("<h2 style='color:#1e293b; font-weight:700; margin-bottom:1.5rem;'>📊 Comprehensive Data Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("<h3 style='color:#1e293b; font-weight:600; margin-bottom:1rem;'>📋 Dataset Overview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True, height=400)
            
            st.markdown("<h3 style='color:#1e293b; font-weight:600; margin:2rem 0 1rem 0;'>🔍 Data Quality Report</h3>", unsafe_allow_html=True)
            quality_metrics = {
                "Missing Values": df[u_cols].isnull().sum().sum(),
                "Complete Records": len(df) - df[u_cols].isnull().any(axis=1).sum(),
                "Avg Cell Voltage": df[u_cols].mean().mean(),
                "Voltage Std Dev": df[u_cols].std().mean(),
                "Cell Range": df["Cell_Range"].mean()
            }
            for metric, value in quality_metrics.items():
                st.metric(metric, f"{value:.3f}" if isinstance(value, float) else value)
        
        with col2:
            st.markdown("<h3 style='color:#1e293b; font-weight:600; margin-bottom:1rem;'>📈 SOH Distribution Analysis</h3>", unsafe_allow_html=True)
            fig_hist = px.histogram(df, x="Pack_SOH", nbins=30, 
                                  title="Battery Pack SOH Distribution",
                                  template="plotly_white",
                                  color_discrete_sequence=['#667eea'])
            fig_hist.update_layout(
                title_font_size=16,
                title_font_color='#1e293b',
                title_font_family='Inter',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Health classification stats
            healthy_count = (df["Pack_SOH"] >= threshold).sum()
            unhealthy_count = len(df) - healthy_count
            
            fig_pie = px.pie(
                values=[healthy_count, unhealthy_count],
                names=['Healthy', 'Problematic'],
                title=f"Battery Health Classification (Threshold: {threshold})",
                color_discrete_sequence=['#10b981', '#ef4444']
            )
            fig_pie.update_layout(
                title_font_size=16,
                title_font_color='#1e293b',
                title_font_family='Inter'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Cell correlation analysis
        st.markdown("<h3 style='color:#1e293b; font-weight:600; margin:2rem 0 1rem 0;'>🔗 Cell Correlation Matrix</h3>", unsafe_allow_html=True)
        corr_matrix = df[u_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Battery Cell Correlation Heatmap",
            template="plotly_white",
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        fig_heatmap.update_layout(
            title_font_size=16,
            title_font_color='#1e293b',
            title_font_family='Inter'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.markdown("<h2 style='color:#1e293b; font-weight:700; margin-bottom:1.5rem;'>🤖 Linear Regression Model Training</h2>", unsafe_allow_html=True)
        
        with st.spinner("🔄 Training Linear Regression model..."):
            model_results = train_linear_regression(df, u_cols, sort_method, test_size, cv_folds, random_state)
        
        st.success("✅ Linear Regression model trained successfully!")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("<h3 style='color:#1e293b; font-weight:600; margin-bottom:1rem;'>📚 Training Performance</h3>", unsafe_allow_html=True)
            train_metrics = model_results["train_metrics"]
            for metric, value in train_metrics.items():
                delta = None
                if metric == "R²":
                    delta = f"{(value - 0.5) * 100:.1f}% vs baseline"
                st.metric(f"🎯 Train {metric}", f"{value:.4f}", delta=delta)
        
        with col2:
            st.markdown("<h3 style='color:#1e293b; font-weight:600; margin-bottom:1rem;'>🎯 Test Performance</h3>", unsafe_allow_html=True)
            test_metrics = model_results["test_metrics"]
            for metric, value in test_metrics.items():
                delta = None
                if metric == "R²":
                    delta = f"{(value - 0.5) * 100:.1f}% vs baseline"
                st.metric(f"📊 Test {metric}", f"{value:.4f}", delta=delta)
        
        with col3:
            st.markdown("<h3 style='color:#1e293b; font-weight:600; margin-bottom:1rem;'>✔️ Cross-Validation</h3>", unsafe_allow_html=True)
            cv_mean = model_results["cv_scores"].mean()
            cv_std = model_results["cv_scores"].std()
            st.metric("📈 CV R² Mean", f"{cv_mean:.4f}")
            st.metric("📉 CV R² Std", f"{cv_std:.4f}")
            st.metric("🔢 Folds", cv_folds)
        
        # Model interpretation
        st.markdown("<h3 style='color:#1e293b; font-weight:600; margin:2rem 0 1rem 0;'>🔬 Model Interpretation</h3>", unsafe_allow_html=True)
        
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
            orientation='h',
            color='Coefficient',
            color_continuous_scale='RdBu'
        )
        fig_importance.update_layout(
            title_font_size=16,
            title_font_color='#1e293b',
            title_font_family='Inter',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model summary
        st.markdown("<h3 style='color:#1e293b; font-weight:600; margin:2rem 0 1rem 0;'>📝 Model Summary</h3>", unsafe_allow_html=True)
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
        st.markdown("<h2 style='color:#1e293b; font-weight:700; margin-bottom:1.5rem;'>📊 Model Performance Visualizations</h2>", unsafe_allow_html=True)
        
        # Create visualizations
        fig1, fig2, fig3 = create_visualizations(model_results, df)
        
        # Enhanced figure layouts
        for fig in [fig1, fig2, fig3]:
            fig.update_layout(
                title_font_size=18,
                title_font_color='#1e293b',
                title_font_family='Inter',
                title_font_weight=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family='Inter',
                font_color='#475569'
            )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(fig3, use_container_width=True)
        
        # Additional analysis
        st.markdown("<h3 style='color:#1e293b; font-weight:600; margin:2rem 0 1rem 0;'>📈 Additional Analysis</h3>", unsafe_allow_html=True)
        
        # Residuals distribution
        residuals = model_results["y_test"] - model_results["y_test_pred"]
        fig_residuals = px.histogram(
            x=residuals,
            title="Distribution of Residuals",
            nbins=30,
            template="plotly_white",
            color_discrete_sequence=['#764ba2']
        )
        fig_residuals.update_layout(
            title_font_size=16,
            title_font_color='#1e293b',
            title_font_family='Inter'
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with tab4:
        # Battery AI Assistant - Modern Chat Interface
        st.markdown("<h2 style='color:#1e293b; font-weight:700; margin-bottom:1.5rem;'>🤖 Battery AI Assistant</h2>", unsafe_allow_html=True)
        
        # Battery-specific suggestions
        BATTERY_SUGGESTIONS = {
            "What is SOH prediction?": "What is State of Health (SOH) prediction and how does it work with battery cells?",
            "Battery maintenance tips": "Give me comprehensive battery maintenance and care tips for longevity",
            "Battery chemistry explained": "Explain different battery chemistries like NMC, LiFePO4 and their characteristics",
            "Battery recycling importance": "Why is battery recycling important for sustainability?",
            "Battery safety guidelines": "What are the key safety considerations when working with batteries?",
            "Model performance analysis": "Analyze the current Linear Regression model performance and provide insights"
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
            with st.expander("Available Knowledge Base Topics"):
                st.markdown("""
                **The RAG system can provide expert insights on:**
                - **SOH Prediction**: State of Health analysis and prediction methods
                - **Battery Maintenance**: Care tips and longevity best practices
                - **Battery Chemistry**: NMC, LiFePO4, degradation mechanisms
                - **Recycling & Sustainability**: Environmental impact and recycling processes
                - **Safety**: Thermal management and protection measures
                - **Model Performance**: Accuracy metrics and evaluation methods
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
                st.text(user_message)
            
            # Display assistant response
            with st.chat_message("assistant"):
                # Check if it's a prediction request
                if any(keyword in user_message.lower() for keyword in ["soh", "predict", "analyze", "health"]):
                    with st.spinner("Analyzing battery data..."):
                        # Select a random sample for prediction
                        sample_idx = np.random.randint(0, len(df))
                        sample_data = df.iloc[sample_idx]
                        
                        X_sample = prepare_features(pd.DataFrame([sample_data]), u_cols, sort_method)
                        X_sample_scaled = model_results["scaler"].transform(X_sample)
                        prediction = model_results["model"].predict(X_sample_scaled)[0]
                        
                        health_status, health_class = classify_battery_health(prediction, threshold)
                        
                        response = f"""## Battery SOH Prediction

**Predicted Pack SOH:** **{prediction:.3f}**  
**Health Status:** {health_status}  

### Model Details
- **Algorithm:** Linear Regression
- **Test R² Score:** {model_results['test_metrics']['R²']:.3f}
- **Test RMSE:** {model_results['test_metrics']['RMSE']:.3f}
- **Health Threshold:** {threshold}

### Dataset Context
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
                        context_data = f"""Current Analysis Context:
                        - Model: Linear Regression
                        - Performance: R² = {model_results['test_metrics']['R²']:.3f}, RMSE = {model_results['test_metrics']['RMSE']:.3f}
                        - Dataset: {len(df)} battery samples
                        - SOH Range: {df['Pack_SOH'].min():.3f} - {df['Pack_SOH'].max():.3f}
                        - Health Threshold: {threshold}
                        - Sorting Method: {sort_method}"""
                    
                    # Stream the response
                    response = st.write_stream(stream_chatgpt_rag(user_message, context_data))
                    
                    # Add feedback controls for streaming response
                    show_feedback_controls(len(st.session_state.battery_messages))
                
                # Add messages to chat history
                st.session_state.battery_messages.append({"role": "user", "content": user_message})
                st.session_state.battery_messages.append({"role": "assistant", "content": response})

else:
    # Welcome Screen with Enhanced Design
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color:#1e293b; font-weight:800; font-size:2.5rem; margin-bottom:1rem;'>Welcome to Battery SOH Platform</h1>
            <p style='color:#64748b; font-size:1.2rem; margin-bottom:2rem;'>Advanced AI-powered battery health prediction and analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("<h2 style='color:#1e293b; font-weight:700; text-align:center; margin:2rem 0;'>🚀 Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 16px; color: white; box-shadow: 0 10px 30px rgba(102,126,234,0.25); min-height: 300px;'>
            <h3 style='margin:0 0 1rem 0; font-size:1.5rem;'>🤖 Linear Regression</h3>
            <ul style='line-height: 1.8;'>
                <li>U1-U21 cell feature extraction</li>
                <li>Multiple sorting techniques</li>
                <li>Cross-validation analysis</li>
                <li>Comprehensive metrics</li>
                <li>Feature importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 16px; color: white; box-shadow: 0 10px 30px rgba(240,147,251,0.25); min-height: 300px;'>
            <h3 style='margin:0 0 1rem 0; font-size:1.5rem;'>📊 Rich Analytics</h3>
            <ul style='line-height: 1.8;'>
                <li>Interactive visualizations</li>
                <li>Correlation analysis</li>
                <li>Distribution plots</li>
                <li>Quality reports</li>
                <li>Real-time insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 16px; color: white; box-shadow: 0 10px 30px rgba(79,172,254,0.25); min-height: 300px;'>
            <h3 style='margin:0 0 1rem 0; font-size:1.5rem;'>🧠 AI Assistant</h3>
            <ul style='line-height: 1.8;'>
                <li>Battery expertise</li>
                <li>Model insights</li>
                <li>Maintenance tips</li>
                <li>Real-time analysis</li>
                <li>RAG-enhanced responses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("<h2 style='color:#1e293b; font-weight:700; text-align:center; margin:3rem 0 2rem 0;'>🎯 Getting Started</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.95); backdrop-filter: blur(20px); padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 8px 32px rgba(15,23,42,0.08);'>
            <h3 style='color:#1e293b; font-weight:600; margin-bottom:1.5rem;'>📝 Steps to Follow</h3>
            <ol style='line-height: 2; color:#475569; font-size:1.05rem;'>
                <li><strong>Upload Dataset:</strong> Use the sidebar to upload your PulseBat CSV file</li>
                <li><strong>Configure Settings:</strong> Adjust model parameters and thresholds</li>
                <li><strong>Explore Analysis:</strong> View comprehensive data insights</li>
                <li><strong>Train Model:</strong> Run Linear Regression prediction</li>
                <li><strong>Chat with AI:</strong> Get expert battery guidance</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.95); backdrop-filter: blur(20px); padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 8px 32px rgba(15,23,42,0.08);'>
            <h3 style='color:#1e293b; font-weight:600; margin-bottom:1.5rem;'>💡 What You'll Get</h3>
            <ul style='line-height: 2; color:#475569; font-size:1.05rem;'>
                <li><strong>Accurate Predictions:</strong> ML-powered SOH forecasting</li>
                <li><strong>Visual Insights:</strong> Beautiful interactive charts</li>
                <li><strong>Health Classification:</strong> Instant battery status</li>
                <li><strong>Expert Guidance:</strong> AI-powered recommendations</li>
                <li><strong>Export Ready:</strong> Downloadable reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; box-shadow: 0 10px 30px rgba(102,126,234,0.3);'>
            <h3 style='color: white; margin: 0 0 0.5rem 0; font-weight:700;'>👈 Ready to Begin?</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size:1.1rem;'>Upload your dataset using the sidebar</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-top: 3rem; box-shadow: 0 10px 40px rgba(102,126,234,0.2);">
    <h3 style="color: white; margin: 0 0 0.5rem 0; font-weight: 700; font-size: 1.5rem;">⚡ Battery Pack SOH Prediction Platform</h3>
    <p style="color: rgba(255,255,255,0.95); margin: 0; font-size: 1.1rem;">Powered by Linear Regression, AI, and Advanced Analytics</p>
    <p style="color: rgba(255,255,255,0.85); margin: 1rem 0 0 0; font-size: 0.95rem;">SOFE3370 Final Project - Group 18 | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)