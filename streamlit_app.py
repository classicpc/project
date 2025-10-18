# 🔋 Battery Pack SOH Prediction & AI Assistant Platform
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
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🔋 Battery Pack SOH Prediction Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .health-good {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .health-bad {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
        color: #333333;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
    }
    .chat-message-assistant {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: #4a148c;
    }
    .stTab > div > div > div > div {
        padding: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🔋 Battery Pack SOH Prediction Platform</h1>
    <h3>SOFE3370 Final Project - Group 18</h3>
    <p>Linear Regression & AI-Powered Battery Health Assessment</p>
    <p><strong>Team:</strong> Pranav Ashok Chaudhari, Tarun Modekurty, Leela Alagala, Hannah Albi</p>
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
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key for ChatGPT integration"
    )
    
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ API Key required for AI Assistant")

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
    with st.spinner("🔄 Loading and preprocessing data..."):
        df = pd.read_csv(file)
        
        # Identify U1-U21 cell columns
        u_cols = [col for col in df.columns if col.startswith("U") and col[1:].isdigit()]
        u_cols = sorted(u_cols, key=lambda x: int(x[1:]))  # Sort U1, U2, ..., U21
        
        # Check if we have the expected U1-U21 columns
        if len(u_cols) != 21:
            st.warning(f"⚠️ Expected 21 cell columns (U1-U21), found {len(u_cols)}: {u_cols}")
        
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
    
    if not api_key:
        # Fallback response using RAG knowledge base
        if retrieved_context:
            return f"""## 🧠 RAG Knowledge Base Response

{retrieved_context}

---
*Note: Add your OpenAI API key in the sidebar for enhanced AI responses.*"""
        else:
            return """## ❌ OpenAI API key not configured

Please add your API key in the sidebar for AI-powered responses.

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
        return f"❌ AI Assistant Error: {str(e)}"

def stream_chatgpt_rag(prompt, context_data=None, api_key=None):
    """Streaming RAG-enhanced ChatGPT integration"""
    # RAG: Retrieve relevant context from knowledge base
    retrieved_context = retrieve_relevant_context(prompt)
    
    if not api_key:
        # Fallback response using RAG knowledge base
        if retrieved_context:
            response = f"""## 🧠 RAG Knowledge Base Response

{retrieved_context}

---
*Note: Add your OpenAI API key in the sidebar for enhanced AI responses.*"""
        else:
            response = """## ❌ OpenAI API key not configured

Please add your API key in the sidebar for AI-powered responses.

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
        yield f"❌ AI Assistant Error: {str(e)}"

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
                st.success("Thank you for your feedback! 🎉")
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

# Demo mode option
if uploaded_file is None:
    st.markdown("### 🎯 Try Demo Mode")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Load Demo Data", type="primary"):
            uploaded_file = "demo"
    with col2:
        st.info("👆 Click to try the app with sample battery data")

# Main application flow
if uploaded_file is not None:
    # Load data (demo or uploaded)
    if uploaded_file == "demo":
        df, u_cols = generate_sample_data()
        st.success("✅ Demo dataset loaded! (1000 synthetic battery samples)")
    else:
        df, u_cols = load_and_preprocess_data(uploaded_file)
        st.success("✅ PulseBat dataset loaded successfully!")
    
    # Update sidebar stats
    with st.sidebar:
        st.markdown("### 📊 Dataset Statistics")
        st.metric("Total Samples", len(df))
        st.metric("Battery Cells", len(u_cols))
        st.metric("Avg Pack SOH", f"{df['Pack_SOH'].mean():.3f}")
        st.metric("SOH Range", f"{df['Pack_SOH'].min():.3f} - {df['Pack_SOH'].max():.3f}")
        
        # Health classification stats
        healthy_count = (df["Pack_SOH"] >= threshold).sum()
        unhealthy_count = len(df) - healthy_count
        st.metric("Healthy Batteries", healthy_count)
        st.metric("Problematic Batteries", unhealthy_count)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🤖 Linear Regression", "📈 Visualizations", "💬 AI Assistant"])
    
    with tab1:
        st.markdown("## 📊 Comprehensive Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Overview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### Data Quality Report")
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
            st.markdown("### SOH Distribution Analysis")
            fig_hist = px.histogram(df, x="Pack_SOH", nbins=30, 
                                  title="Battery Pack SOH Distribution",
                                  template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Health classification stats
            healthy_count = (df["Pack_SOH"] >= threshold).sum()
            unhealthy_count = len(df) - healthy_count
            
            fig_pie = px.pie(
                values=[healthy_count, unhealthy_count],
                names=['Healthy', 'Problematic'],
                title=f"Battery Health Classification (Threshold: {threshold})",
                color_discrete_sequence=['#4CAF50', '#f44336']
            )
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
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.markdown("## 🤖 Linear Regression Model Training")
        
        with st.spinner("🔄 Training Linear Regression model..."):
            model_results = train_linear_regression(df, u_cols, sort_method, test_size, cv_folds, random_state)
        
        st.success("✅ Linear Regression model trained successfully!")
        
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
                        context_data = f"""Current Analysis Context:
                        - Model: Linear Regression
                        - Performance: R² = {model_results['test_metrics']['R²']:.3f}, RMSE = {model_results['test_metrics']['RMSE']:.3f}
                        - Dataset: {len(df)} battery samples
                        - SOH Range: {df['Pack_SOH'].min():.3f} - {df['Pack_SOH'].max():.3f}
                        - Health Threshold: {threshold}
                        - Sorting Method: {sort_method}"""
                    
                    # Stream the response
                    response = st.write_stream(stream_chatgpt_rag(user_message, context_data, api_key))
                    
                    # Add feedback controls for streaming response
                    show_feedback_controls(len(st.session_state.battery_messages))
                
                # Add messages to chat history
                st.session_state.battery_messages.append({"role": "user", "content": user_message})
                st.session_state.battery_messages.append({"role": "assistant", "content": response})

else:
    st.markdown("""
    ## 🚀 Welcome to Battery Pack SOH Prediction Platform
    
    ### Project Overview
    
    This platform implements a comprehensive battery State of Health (SOH) prediction system using Linear Regression 
    and AI-powered analysis, developed for the SOFE3370 Final Project.
    
    ### Key Features
    
    🔋 **Battery SOH Prediction**: Upload your PulseBat dataset to predict State of Health using U1-U21 cell data
    
    🤖 **Linear Regression Model**: Advanced machine learning model with comprehensive evaluation metrics
    
    📊 **Data Preprocessing**: Multiple sorting techniques (None, Ascending, Descending) for cell data analysis
    
    ⚡ **Health Classification**: Configurable threshold-based classification (default: 0.6)
    
    📈 **Rich Visualizations**: Interactive charts showing predicted vs actual SOH, residuals, and distributions
    
    💬 **AI Assistant**: ChatGPT-powered assistant for battery insights and maintenance tips
    
    ### Getting Started:
    1. 📁 Upload your PulseBat CSV file using the sidebar
    2. ⚙️ Configure your model settings (sorting method, threshold, etc.)
    3. 🎯 Explore the data analysis, model training, and visualizations
    4. 💭 Chat with our AI assistant for insights and battery tips
    
    ---
    
    **🎯 Don't have data? Try our demo mode above!**
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
    <p><strong>SOFE3370 Final Project - Group 18</strong></p>
    <p>Battery Pack SOH Prediction using Linear Regression & Chatbot Integration</p>
    <p>Pranav Ashok Chaudhari | Tarun Modekurty | Leela Alagala | Hannah Albi</p>
</div>
""", unsafe_allow_html=True)