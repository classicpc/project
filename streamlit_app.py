# 🔋 Advanced Battery SOH Prediction & AI Assistant Platform
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# -------------------------------
# Advanced Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🔋 SmartBattery AI - Advanced SOH Prediction Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTab > div > div > div > div {
        padding: 2rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🔋 SmartBattery AI</h1>
    <h3>Advanced State of Health Prediction Platform</h3>
    <p>Leveraging Machine Learning & AI for Sustainable Battery Management</p>
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
        help="Upload your battery pulse data in CSV format"
    )
    
    st.markdown("---")
    
    # Model Selection
    model_type = st.selectbox(
        "🤖 Select ML Model:",
        ["Linear Regression", "Random Forest", "Support Vector Machine"],
        help="Choose the machine learning algorithm"
    )
    
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
        
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if uploaded_file is not None:
        # We'll populate this after loading data
        pass

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
    """Enhanced data loading with comprehensive preprocessing"""
    with st.spinner("🔄 Loading and preprocessing data..."):
        df = pd.read_csv(file)
        
        # Identify cell columns
        u_cols = [col for col in df.columns if col.startswith("U")]
        
        # Create multiple aggregation methods
        df["Pack_SOH_Mean"] = df[u_cols].mean(axis=1)
        df["Pack_SOH_Median"] = df[u_cols].median(axis=1)
        df["Pack_SOH_Min"] = df[u_cols].min(axis=1)
        df["Pack_SOH_Max"] = df[u_cols].max(axis=1)
        df["Pack_SOH_Std"] = df[u_cols].std(axis=1)
        
        # Use mean as primary target
        df["Pack_SOH"] = df["Pack_SOH_Mean"]
        
        # Add data quality metrics
        df["Missing_Cells"] = df[u_cols].isnull().sum(axis=1)
        df["Cell_Range"] = df["Pack_SOH_Max"] - df["Pack_SOH_Min"]
        
    return df

@st.cache_data
def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate U1-U21 columns
    data = {}
    for i in range(1, 22):
        # Simulate realistic battery cell voltages with some correlation
        base_voltage = np.random.normal(0.7, 0.15, n_samples)
        noise = np.random.normal(0, 0.05, n_samples)
        data[f'U{i}'] = np.clip(base_voltage + noise, 0.1, 1.0)
    
    # Generate SOH based on cell voltages with some realistic patterns
    df = pd.DataFrame(data)
    u_cols = [f'U{i}' for i in range(1, 22)]
    df["SOH"] = df[u_cols].mean(axis=1) * np.random.normal(1, 0.1, n_samples)
    df["SOH"] = np.clip(df["SOH"], 0.1, 1.0)
    
    # Add some metadata
    df["ID"] = range(1, n_samples + 1)
    df["Mat"] = np.random.choice(['LiCoO2', 'LiFePO4', 'NMC'], n_samples)
    
    return df

# -------------------------------
# Train Linear Regression
# -------------------------------
def prepare_features(df, sort_method):
    """Prepare features with advanced preprocessing"""
    feature_cols = [col for col in df.columns if col.startswith("U")]
    X = df[feature_cols].copy()
    
    # Apply sorting if selected
    if sort_method == "Ascending":
        X = X.apply(lambda row: np.sort(row.values), axis=1, result_type="expand")
        X.columns = [f"U{i+1}_sorted_asc" for i in range(len(X.columns))]
    elif sort_method == "Descending":
        X = X.apply(lambda row: -np.sort(-row.values), axis=1, result_type="expand")
        X.columns = [f"U{i+1}_sorted_desc" for i in range(len(X.columns))]
    
    return X, feature_cols

def train_advanced_model(df, sort_method, model_type, test_size):
    """Train advanced ML models with comprehensive evaluation"""
    
    X, feature_cols = prepare_features(df, sort_method)
    y = df["Pack_SOH"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)  # RF doesn't need scaling
        X_train_scaled, X_test_scaled = X_train, X_test
    else:  # SVM
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
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
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "feature_cols": feature_cols
    }

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
def predict_single_sample(model_results, df, sort_method):
    """Predict SOH for a single sample"""
    sample_row = df.sample(1)
    X_sample, _ = prepare_features(sample_row, sort_method)
    
    if model_results["model"].__class__.__name__ != "RandomForestRegressor":
        X_sample = model_results["scaler"].transform(X_sample)
    
    prediction = model_results["model"].predict(X_sample)[0]
    return prediction

def ask_chatgpt(prompt, context_data=None):
    """Enhanced ChatGPT integration with rich context"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OpenAI API key not found. Please set OPENAI_API_KEY as an environment variable."
        
        system_msg = """You are an expert AI assistant specializing in battery technology, state of health prediction, 
        and sustainable energy storage systems. You have deep knowledge of:
        - Battery chemistry and degradation mechanisms
        - Machine learning applications in battery management
        - Sustainable battery lifecycle management
        - Battery safety and maintenance best practices
        
        Provide detailed, technical, and actionable insights."""
        
        if context_data:
            system_msg += f"\n\nCurrent Analysis Context:\n{context_data}"
        
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

def create_advanced_visualizations(model_results, df):
    """Create comprehensive visualizations"""
    
    # 1. Performance Comparison Chart
    fig1 = go.Figure()
    
    # Training performance
    fig1.add_trace(go.Scatter(
        x=model_results["y_train"], 
        y=model_results["y_train_pred"],
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Test performance
    fig1.add_trace(go.Scatter(
        x=model_results["y_test"], 
        y=model_results["y_test_pred"],
        mode='markers',
        name='Test Data',
        marker=dict(color='red', opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(model_results["y_train"].min(), model_results["y_test"].min())
    max_val = max(model_results["y_train"].max(), model_results["y_test"].max())
    fig1.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='green', dash='dash')
    ))
    
    fig1.update_layout(
        title="Model Performance: Predicted vs Actual SOH",
        xaxis_title="Actual SOH",
        yaxis_title="Predicted SOH",
        template="plotly_white"
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
        template="plotly_white"
    )
    
    # 3. Battery Health Distribution
    fig3 = px.histogram(
        df, x="Pack_SOH", 
        title="Battery SOH Distribution",
        nbins=30,
        template="plotly_white"
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
        df = generate_sample_data()
        st.success("✅ Demo dataset loaded! (1000 synthetic battery samples)")
    else:
        df = load_data(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
    
    # Update sidebar stats
    with st.sidebar:
        st.markdown("### 📊 Dataset Statistics")
        st.metric("Total Samples", len(df))
        st.metric("Battery Cells", len([col for col in df.columns if col.startswith("U")]))
        st.metric("Avg SOH", f"{df['Pack_SOH'].mean():.3f}")
        st.metric("SOH Range", f"{df['Pack_SOH'].min():.3f} - {df['Pack_SOH'].max():.3f}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🤖 ML Models", "📈 Visualizations", "💬 AI Assistant"])
    
    with tab1:
        st.markdown("## 📊 Comprehensive Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Overview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("### Data Quality Report")
            u_cols = [col for col in df.columns if col.startswith("U")]
            quality_metrics = {
                "Missing Values": df[u_cols].isnull().sum().sum(),
                "Complete Records": len(df) - df[u_cols].isnull().any(axis=1).sum(),
                "Avg Cell Voltage": df[u_cols].mean().mean(),
                "Voltage Std Dev": df[u_cols].std().mean()
            }
            for metric, value in quality_metrics.items():
                st.metric(metric, f"{value:.3f}" if isinstance(value, float) else value)
        
        with col2:
            st.markdown("### SOH Distribution Analysis")
            fig_hist = px.histogram(df, x="Pack_SOH", nbins=30, 
                                  title="Battery SOH Distribution",
                                  template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Health classification stats
            healthy_count = (df["Pack_SOH"] >= threshold).sum()
            unhealthy_count = len(df) - healthy_count
            
            fig_pie = px.pie(
                values=[healthy_count, unhealthy_count],
                names=['Healthy', 'Problematic'],
                title=f"Battery Health Classification (Threshold: {threshold})",
                color_discrete_sequence=['#2E8B57', '#DC143C']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.markdown("## 🤖 Machine Learning Model Training")
        
        with st.spinner("🔄 Training advanced ML models..."):
            model_results = train_advanced_model(df, sort_method, model_type, test_size)
        
        st.success(f"✅ {model_type} model trained successfully!")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
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
        
        # Model comparison
        st.markdown("### 🔄 Quick Model Comparison")
        comparison_data = []
        
        for model_name in ["Linear Regression", "Random Forest", "Support Vector Machine"]:
            temp_results = train_advanced_model(df, sort_method, model_name, test_size)
            comparison_data.append({
                "Model": model_name,
                "Test R²": temp_results["test_metrics"]["R²"],
                "Test RMSE": temp_results["test_metrics"]["RMSE"],
                "Test MAE": temp_results["test_metrics"]["MAE"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Feature importance (for Random Forest)
        if model_type == "Random Forest":
            st.markdown("### 🌳 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': model_results["X_train"].columns,
                'Importance': model_results["model"].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10), 
                x='Importance', y='Feature',
                title="Top 10 Most Important Features",
                template="plotly_white"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab3:
        st.markdown("## 📈 Advanced Visualizations")
        
        # Create visualizations
        fig1, fig2, fig3 = create_advanced_visualizations(model_results, df)
        
        st.plotly_chart(fig1, width='stretch', key='fig1')
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig2, width='stretch', key='fig2')
        with col2:
            st.plotly_chart(fig3, width='stretch', key='fig3')
        
        # Cell correlation heatmap
        st.markdown("### 🔥 Cell Correlation Matrix")
        u_cols = [col for col in df.columns if col.startswith("U")]
        corr_matrix = df[u_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            title="Battery Cell Correlation Heatmap",
            template="plotly_white",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.markdown("## 💬 AI-Powered Battery Assistant")
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Analyze Current Battery"):
                prediction = predict_single_sample(model_results, df, sort_method)
                health_status = classify_soh(prediction, threshold)
                
                st.session_state.chat_history.append(("user", "Analyze current battery"))
                reply = f"🔮 **Battery Analysis Result:**\n\nPredicted SOH: **{prediction:.3f}**\n\n{health_status}\n\n📊 Model used: {model_type}\n🎯 Confidence: {model_results['test_metrics']['R²']:.3f}"
                st.session_state.chat_history.append(("assistant", reply))
        
        with col2:
            if st.button("📊 Get Model Insights"):
                context = f"""Model Performance: R² = {model_results['test_metrics']['R²']:.3f}, 
                RMSE = {model_results['test_metrics']['RMSE']:.3f}. 
                Dataset: {len(df)} samples, SOH range: {df['Pack_SOH'].min():.3f}-{df['Pack_SOH'].max():.3f}"""
                
                st.session_state.chat_history.append(("user", "Provide insights about the model performance"))
                reply = ask_chatgpt("Analyze this battery prediction model performance and provide insights.", context)
                st.session_state.chat_history.append(("assistant", reply))
        
        with col3:
            if st.button("💡 Battery Tips"):
                st.session_state.chat_history.append(("user", "Give me battery maintenance tips"))
                reply = ask_chatgpt("Provide comprehensive battery maintenance and longevity tips.")
                st.session_state.chat_history.append(("assistant", reply))
        
        # Chat interface
        user_input = st.chat_input("Ask anything about batteries, SOH prediction, or get analysis insights...")
        
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            
            if "soh" in user_input.lower() or "predict" in user_input.lower():
                prediction = predict_single_sample(model_results, df, sort_method)
                health_status = classify_soh(prediction, threshold)
                bot_reply = f"🔮 **Battery SOH Prediction:**\n\nPredicted Pack SOH: **{prediction:.3f}**\n\n{health_status}\n\n📈 Model: {model_type} (R² = {model_results['test_metrics']['R²']:.3f})"
            else:
                context_data = f"""Current Analysis Context:
                - Model: {model_type}
                - Performance: R² = {model_results['test_metrics']['R²']:.3f}, RMSE = {model_results['test_metrics']['RMSE']:.3f}
                - Dataset: {len(df)} battery samples
                - SOH Range: {df['Pack_SOH'].min():.3f} - {df['Pack_SOH'].max():.3f}
                - Health Threshold: {threshold}
                - Sorting Method: {sort_method}"""
                bot_reply = ask_chatgpt(user_input, context_data)
            
            st.session_state.chat_history.append(("assistant", bot_reply))
        
        # Display chat history with enhanced styling
        st.markdown("### 💭 Conversation History")
        for i, (role, message) in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if role == "user":
                st.chat_message("user").markdown(f"**You:** {message}")
            else:
                st.chat_message("assistant").markdown(f"**AI Assistant:** {message}")

else:
    st.markdown("""
    ## 🚀 Welcome to SmartBattery AI
    
    ### What can you do here?
    
    🔋 **Battery SOH Prediction**: Upload your PulseBat dataset or try our demo to predict State of Health
    
    🤖 **Multiple ML Models**: Compare Linear Regression, Random Forest, and SVM algorithms
    
    📊 **Advanced Analytics**: Comprehensive data analysis with interactive visualizations
    
    💬 **AI Assistant**: Chat with our intelligent assistant for battery insights and maintenance tips
    
    ### Getting Started:
    1. 📁 Upload your PulseBat CSV file using the sidebar
    2. ⚙️ Configure your model and preprocessing settings
    3. 🎯 Explore the analysis, models, and visualizations
    4. 💭 Chat with our AI assistant for insights
    
    ---
    
    **🎯 Don't have data? Try our demo mode above!**
    """)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔬 Advanced ML
        - Multiple algorithms
        - Cross-validation
        - Feature importance
        - Performance metrics
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Rich Analytics  
        - Interactive charts
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
