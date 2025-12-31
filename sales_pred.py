import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import pickle
import io
import base64
from datetime import datetime

# Page config
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide", page_icon="🛒")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0071ce;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_embedded_data():
    """Load the embedded Walmart sales dataset"""
    try:
        # Try loading from the file path first
        df = pd.read_csv("C:/Users/Haji/PycharmProjects/PortfolioMLProject/walmart_data.csv")

        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please ensure '1767142983508_Walmart_Sales.csv' is in the same directory.")
        return None


@st.cache_data
def preprocess_data(df, target_col='Weekly_Sales'):
    """Preprocess the Walmart sales data"""
    df = df.copy()

    # Drop original Date column
    df = df.drop('Date', axis=1)

    # Handle missing values (if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Create interaction features
    df['Temp_Fuel_Interaction'] = df['Temperature'] * df['Fuel_Price']
    df['CPI_Unemployment_Ratio'] = df['CPI'] / (df['Unemployment'] + 0.01)

    return df


@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    """Train XGBoost and LightGBM models with GridSearchCV"""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # XGBoost
    st.write("### 🔄 Training XGBoost Model with GridSearchCV...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    xgb_params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    status_text.text("XGBoost: Searching for best hyperparameters...")
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='neg_mean_squared_error',
                            verbose=0, n_jobs=-1)
    xgb_grid.fit(X_train_scaled, y_train)
    progress_bar.progress(50)

    xgb_pred = xgb_grid.predict(X_test_scaled)
    results['xgboost'] = {
        'model': xgb_grid.best_estimator_,
        'predictions': xgb_pred,
        'best_params': xgb_grid.best_params_,
        'mse': mean_squared_error(y_test, xgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'mae': mean_absolute_error(y_test, xgb_pred),
        'r2': r2_score(y_test, xgb_pred),
        'mape': np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100
    }
    status_text.text("✅ XGBoost training completed!")

    # LightGBM
    st.write("### 🔄 Training LightGBM Model with GridSearchCV...")
    lgb_params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'num_leaves': [31, 50],
        'min_child_samples': [20, 30],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    status_text.text("LightGBM: Searching for best hyperparameters...")
    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=3, scoring='neg_mean_squared_error',
                            verbose=0, n_jobs=-1)
    lgb_grid.fit(X_train_scaled, y_train)
    progress_bar.progress(100)

    lgb_pred = lgb_grid.predict(X_test_scaled)
    results['lightgbm'] = {
        'model': lgb_grid.best_estimator_,
        'predictions': lgb_pred,
        'best_params': lgb_grid.best_params_,
        'mse': mean_squared_error(y_test, lgb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred)),
        'mae': mean_absolute_error(y_test, lgb_pred),
        'r2': r2_score(y_test, lgb_pred),
        'mape': np.mean(np.abs((y_test - lgb_pred) / y_test)) * 100
    }
    status_text.text("✅ LightGBM training completed!")

    return results, scaler


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]

    fig = go.Figure(go.Bar(
        x=importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=importance[indices],
            colorscale='Viridis',
            showscale=True
        )
    ))
    fig.update_layout(
        title=f'Top 15 Feature Importances - {model_name}',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500,
        template='plotly_white'
    )
    return fig


def plot_predictions(y_test, predictions, model_name):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=predictions,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, opacity=0.6, color='#667eea')
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        title=f'Actual vs Predicted Sales - {model_name}',
        xaxis_title='Actual Sales ($)',
        yaxis_title='Predicted Sales ($)',
        height=500,
        template='plotly_white'
    )
    return fig


def plot_residuals(y_test, predictions, model_name):
    """Plot residuals"""
    residuals = y_test - predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        marker=dict(size=8, opacity=0.6, color='#764ba2')
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=f'Residual Plot - {model_name}',
        xaxis_title='Predicted Sales ($)',
        yaxis_title='Residuals',
        height=400,
        template='plotly_white'
    )
    return fig


def plot_sales_distribution(df):
    """Plot sales distribution"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Weekly_Sales'],
        nbinsx=50,
        marker_color='#667eea',
        opacity=0.7
    ))
    fig.update_layout(
        title='Distribution of Weekly Sales',
        xaxis_title='Weekly Sales ($)',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    return fig


def plot_sales_by_store(df):
    """Plot average sales by store"""
    avg_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=avg_sales.index,
        y=avg_sales.values,
        marker_color='#764ba2'
    ))
    fig.update_layout(
        title='Average Weekly Sales by Store',
        xaxis_title='Store',
        yaxis_title='Average Sales ($)',
        template='plotly_white'
    )
    return fig


def plot_sales_trend(df):
    """Plot sales trend over time"""
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], format='%d-%m-%Y', errors='coerce')
    monthly_sales = df_copy.groupby(df_copy['Date'].dt.to_period('M'))['Weekly_Sales'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales.index.astype(str),
        y=monthly_sales.values,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title='Monthly Average Sales Trend',
        xaxis_title='Month',
        yaxis_title='Average Sales ($)',
        template='plotly_white'
    )
    return fig


# Main App
def main():
    st.markdown('<p class="main-header">🛒 Walmart Sales Prediction System</p>', unsafe_allow_html=True)
    st.markdown("### Predict weekly sales using XGBoost and LightGBM ensemble models")

    # Sidebar
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/2560px-Walmart_logo.svg.png",
        width=200)
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Overview", "🤖 Model Training", "🔮 Make Predictions"])

    # Load data
    df = load_embedded_data()
    if df is None:
        return

    st.sidebar.success(f"✅ Data loaded successfully!")
    st.sidebar.info(f"📦 Records: {df.shape[0]:,}\n\n📋 Features: {df.shape[1]}")

    if page == "🏠 Home":
        st.header("Welcome to Walmart Sales Prediction System")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("### 📊 Data\n6,435 records\n8 features")
        with col2:
            st.success("### 🤖 Models\nXGBoost\nLightGBM")
        with col3:
            st.warning("### 🎯 Target\nWeekly Sales")

        st.markdown("---")
        st.subheader("📌 Dataset Features")

        features_info = {
            'Store': 'Store number (1-45)',
            'Date': 'Week of sales',
            'Weekly_Sales': 'Sales for the given store (Target Variable)',
            'Holiday_Flag': 'Whether the week contains a holiday (1 = Yes, 0 = No)',
            'Temperature': 'Temperature on the day of sale',
            'Fuel_Price': 'Cost of fuel in the region',
            'CPI': 'Consumer Price Index',
            'Unemployment': 'Unemployment rate'
        }

        for feature, description in features_info.items():
            st.markdown(f"**{feature}**: {description}")

        st.markdown("---")
        st.subheader("🚀 How to Use")
        st.markdown("""
        1. **Data Overview**: Explore the dataset with visualizations and statistics
        2. **Model Training**: Train XGBoost and LightGBM models with automatic hyperparameter tuning
        3. **Make Predictions**: Input your data and get sales predictions
        """)

    elif page == "📊 Data Overview":
        st.header("📊 Data Overview & Exploratory Analysis")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{df.shape[0]:,}")
        col2.metric("Unique Stores", df['Store'].nunique())
        col3.metric("Avg Weekly Sales", f"${df['Weekly_Sales'].mean():,.2f}")
        col4.metric("Max Weekly Sales", f"${df['Weekly_Sales'].max():,.2f}")

        st.markdown("---")

        # Dataset Preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Statistical Summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)

        # Visualizations
        st.subheader("📊 Data Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs(["Sales Distribution", "Sales by Store", "Sales Trend", "Correlations"])

        with tab1:
            st.plotly_chart(plot_sales_distribution(df), use_container_width=True)

        with tab2:
            st.plotly_chart(plot_sales_by_store(df), use_container_width=True)

        with tab3:
            st.plotly_chart(plot_sales_trend(df), use_container_width=True)

        with tab4:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title='Feature Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

        # Missing Values
        st.subheader("❓ Missing Values Analysis")
        missing = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if len(missing) > 0:
            st.dataframe(missing, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    elif page == "🤖 Model Training":
        st.header("🤖 Model Training & Evaluation")

        st.info("📌 This will train both XGBoost and LightGBM models with GridSearchCV for hyperparameter optimization")

        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        with col2:
            random_state = st.number_input("Random State", 0, 100, 42)

        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing data and training models..."):
                # Preprocess
                df_processed = preprocess_data(df)

                st.success("✅ Data preprocessing completed!")
                st.write(f"**Features created**: {df_processed.shape[1] - 1}")

                # Prepare features and target
                X = df_processed.drop('Weekly_Sales', axis=1)
                y = df_processed['Weekly_Sales']

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                st.info(f"📊 Training samples: {len(X_train):,} | Testing samples: {len(X_test):,}")

                # Train models
                results, scaler = train_models(X_train, y_train, X_test, y_test)

                # Save to session state
                st.session_state['results'] = results
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test

                st.success("✅ Training complete!")

                # Model Comparison
                st.markdown("---")
                st.subheader("📊 Model Performance Comparison")

                comparison_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R² Score', 'MAPE (%)'],
                    'XGBoost': [
                        f"{results['xgboost']['rmse']:.2f}",
                        f"{results['xgboost']['mae']:.2f}",
                        f"{results['xgboost']['r2']:.4f}",
                        f"{results['xgboost']['mape']:.2f}"
                    ],
                    'LightGBM': [
                        f"{results['lightgbm']['rmse']:.2f}",
                        f"{results['lightgbm']['mae']:.2f}",
                        f"{results['lightgbm']['r2']:.4f}",
                        f"{results['lightgbm']['mape']:.2f}"
                    ]
                })

                st.dataframe(comparison_df, use_container_width=True)

                # Determine best model
                best_model = 'XGBoost' if results['xgboost']['r2'] > results['lightgbm']['r2'] else 'LightGBM'
                st.success(
                    f"🏆 Best Model: **{best_model}** (R² Score: {max(results['xgboost']['r2'], results['lightgbm']['r2']):.4f})")

                # Detailed Results
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔵 XGBoost Results")
                    st.metric("RMSE", f"${results['xgboost']['rmse']:,.2f}")
                    st.metric("MAE", f"${results['xgboost']['mae']:,.2f}")
                    st.metric("R² Score", f"{results['xgboost']['r2']:.4f}")
                    st.metric("MAPE", f"{results['xgboost']['mape']:.2f}%")
                    with st.expander("View Best Parameters"):
                        st.json(results['xgboost']['best_params'])

                with col2:
                    st.markdown("### 🟢 LightGBM Results")
                    st.metric("RMSE", f"${results['lightgbm']['rmse']:,.2f}")
                    st.metric("MAE", f"${results['lightgbm']['mae']:,.2f}")
                    st.metric("R² Score", f"{results['lightgbm']['r2']:.4f}")
                    st.metric("MAPE", f"{results['lightgbm']['mape']:.2f}%")
                    with st.expander("View Best Parameters"):
                        st.json(results['lightgbm']['best_params'])

                # Feature Importance
                st.markdown("---")
                st.subheader("🎯 Feature Importance Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    fig_xgb = plot_feature_importance(
                        results['xgboost']['model'],
                        st.session_state['feature_names'],
                        'XGBoost'
                    )
                    st.plotly_chart(fig_xgb, use_container_width=True)

                with col2:
                    fig_lgb = plot_feature_importance(
                        results['lightgbm']['model'],
                        st.session_state['feature_names'],
                        'LightGBM'
                    )
                    st.plotly_chart(fig_lgb, use_container_width=True)

                # Prediction Visualizations
                st.markdown("---")
                st.subheader("📉 Prediction Analysis")

                tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residual Plots", "Error Distribution"])

                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pred_xgb = plot_predictions(
                            y_test,
                            results['xgboost']['predictions'],
                            'XGBoost'
                        )
                        st.plotly_chart(fig_pred_xgb, use_container_width=True)

                    with col2:
                        fig_pred_lgb = plot_predictions(
                            y_test,
                            results['lightgbm']['predictions'],
                            'LightGBM'
                        )
                        st.plotly_chart(fig_pred_lgb, use_container_width=True)

                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_res_xgb = plot_residuals(
                            y_test,
                            results['xgboost']['predictions'],
                            'XGBoost'
                        )
                        st.plotly_chart(fig_res_xgb, use_container_width=True)

                    with col2:
                        fig_res_lgb = plot_residuals(
                            y_test,
                            results['lightgbm']['predictions'],
                            'LightGBM'
                        )
                        st.plotly_chart(fig_res_lgb, use_container_width=True)

                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        errors_xgb = np.abs(y_test - results['xgboost']['predictions'])
                        fig_err_xgb = go.Figure(go.Histogram(x=errors_xgb, nbinsx=50, marker_color='#667eea'))
                        fig_err_xgb.update_layout(title='XGBoost Error Distribution',
                                                  xaxis_title='Absolute Error',
                                                  yaxis_title='Frequency')
                        st.plotly_chart(fig_err_xgb, use_container_width=True)

                    with col2:
                        errors_lgb = np.abs(y_test - results['lightgbm']['predictions'])
                        fig_err_lgb = go.Figure(go.Histogram(x=errors_lgb, nbinsx=50, marker_color='#764ba2'))
                        fig_err_lgb.update_layout(title='LightGBM Error Distribution',
                                                  xaxis_title='Absolute Error',
                                                  yaxis_title='Frequency')
                        st.plotly_chart(fig_err_lgb, use_container_width=True)

    elif page == "🔮 Make Predictions":
        st.header("🔮 Make Sales Predictions")

        if 'results' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Model Training' section!")
            return

        st.success("✅ Models loaded and ready for predictions!")

        # Model Selection
        col1, col2 = st.columns([1, 2])
        with col1:
            model_choice = st.selectbox(
                "Choose Model",
                ["XGBoost", "LightGBM"],
                help="Select which model to use for prediction"
            )

        with col2:
            r2_xgb = st.session_state['results']['xgboost']['r2']
            r2_lgb = st.session_state['results']['lightgbm']['r2']
            st.info(f"**Model Performance** - XGBoost R²: {r2_xgb:.4f} | LightGBM R²: {r2_lgb:.4f}")

        st.markdown("---")
        st.subheader("📝 Enter Feature Values")

        # Input fields
        col1, col2, col3 = st.columns(3)

        with col1:
            store = st.number_input("Store Number", min_value=1, max_value=45, value=1, step=1)
            temperature = st.number_input("Temperature (°F)", min_value=-10.0, max_value=120.0, value=60.0, step=0.1)
            fuel_price = st.number_input("Fuel Price ($)", min_value=2.0, max_value=5.0, value=3.5, step=0.01)

        with col2:
            holiday_flag = st.selectbox("Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            cpi = st.number_input("CPI (Consumer Price Index)", min_value=100.0, max_value=250.0, value=211.0, step=0.1)
            unemployment = st.number_input("Unemployment Rate (%)", min_value=3.0, max_value=15.0, value=7.0, step=0.1)

        with col3:
            date_input = st.date_input("Date", value=datetime.now())
            st.info("💡 Date will be used to extract temporal features like month, day of week, etc.")

        if st.button("🎯 Predict Weekly Sales", type="primary", use_container_width=True):
            with st.spinner("Calculating prediction..."):
                # Create date features
                year = date_input.year
                month = date_input.month
                day = date_input.day
                dayofweek = date_input.weekday()
                quarter = (month - 1) // 3 + 1
                weekofyear = date_input.isocalendar()[1]
                isweekend = 1 if dayofweek in [5, 6] else 0

                # Create interaction features
                temp_fuel_interaction = temperature * fuel_price
                cpi_unemployment_ratio = cpi / (unemployment + 0.01)

                # Prepare input dataframe with all features in correct order
                feature_names = st.session_state['feature_names']
                input_dict = {
                    'Store': store,
                    'Holiday_Flag': holiday_flag,
                    'Temperature': temperature,
                    'Fuel_Price': fuel_price,
                    'CPI': cpi,
                    'Unemployment': unemployment,
                    'Year': year,
                    'Month': month,
                    'Day': day,
                    'DayOfWeek': dayofweek,
                    'Quarter': quarter,
                    'WeekOfYear': weekofyear,
                    'IsWeekend': isweekend,
                    'Temp_Fuel_Interaction': temp_fuel_interaction,
                    'CPI_Unemployment_Ratio': cpi_unemployment_ratio
                }

                # Ensure all features are present
                input_data = pd.DataFrame([{feature: input_dict.get(feature, 0) for feature in feature_names}])

                # Scale
                input_scaled = st.session_state['scaler'].transform(input_data)

                # Predict
                model_key = model_choice.lower().replace(' ', '')
                model = st.session_state['results'][model_key]['model']
                prediction = model.predict(input_scaled)[0]

                # Display result
                st.markdown("---")
                st.markdown("## 🎉 Prediction Result")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label=f"Predicted Weekly Sales ({model_choice})",
                        value=f"${prediction:,.2f}",
                        delta=None
                    )

                with col2:
                    model_r2 = st.session_state['results'][model_key]['r2']
                    st.metric(
                        label="Model R² Score",
                        value=f"{model_r2:.4f}"
                    )

                with col3:
                    model_rmse = st.session_state['results'][model_key]['rmse']
                    st.metric(
                        label="Model RMSE",
                        value=f"${model_rmse:,.2f}"
                    )

                # Prediction confidence interval (approximate)
                st.markdown("---")
                st.subheader("📊 Prediction Insights")

                col1, col2 = st.columns(2)

                with col1:
                    st.info(f"""
                    **Input Summary:**
                    - Store: {store}
                    - Date: {date_input.strftime('%Y-%m-%d')} ({date_input.strftime('%A')})
                    - Holiday: {'Yes' if holiday_flag == 1 else 'No'}
                    - Temperature: {temperature}°F
                    - Fuel Price: ${fuel_price}
                    - CPI: {cpi}
                    - Unemployment: {unemployment}%
                    """)

                with col2:
                    # Calculate approximate confidence interval
                    margin_error = model_rmse * 1.96  # 95% confidence
                    lower_bound = prediction - margin_error
                    upper_bound = prediction + margin_error

                    st.warning(f"""
                    **Prediction Range (95% CI):**
                    - Lower Bound: ${lower_bound:,.2f}
                    - Predicted: ${prediction:,.2f}
                    - Upper Bound: ${upper_bound:,.2f}

                    *The actual sales is expected to fall within this range with 95% confidence*
                    """)

                # Show feature values used
                with st.expander("🔍 View All Feature Values Used"):
                    st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

                # Comparison with both models
                st.markdown("---")
                st.subheader("🔄 Comparison with Both Models")

                # Get prediction from other model
                other_model_key = 'lightgbm' if model_key == 'xgboost' else 'xgboost'
                other_model = st.session_state['results'][other_model_key]['model']
                other_prediction = other_model.predict(input_scaled)[0]

                comparison_data = pd.DataFrame({
                    'Model': ['XGBoost', 'LightGBM'],
                    'Predicted Sales': [
                        f"${st.session_state['results']['xgboost']['model'].predict(input_scaled)[0]:,.2f}",
                        f"${st.session_state['results']['lightgbm']['model'].predict(input_scaled)[0]:,.2f}"
                    ],
                    'R² Score': [
                        f"{st.session_state['results']['xgboost']['r2']:.4f}",
                        f"{st.session_state['results']['lightgbm']['r2']:.4f}"
                    ],
                    'RMSE': [
                        f"${st.session_state['results']['xgboost']['rmse']:,.2f}",
                        f"${st.session_state['results']['lightgbm']['rmse']:,.2f}"
                    ]
                })

                st.dataframe(comparison_data, use_container_width=True)

                avg_prediction = (st.session_state['results']['xgboost']['model'].predict(input_scaled)[0] +
                                  st.session_state['results']['lightgbm']['model'].predict(input_scaled)[0]) / 2

                st.success(f"📊 **Ensemble Average Prediction:** ${avg_prediction:,.2f}")

        # Batch Prediction
        st.markdown("---")
        st.subheader("📦 Batch Predictions")
        st.info("Upload a CSV file with the same features to get predictions for multiple records")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Generate Batch Predictions", use_container_width=True):
                with st.spinner("Processing batch predictions..."):
                    # Preprocess batch data
                    batch_processed = preprocess_data(batch_df)

                    # Ensure same features
                    for feature in st.session_state['feature_names']:
                        if feature not in batch_processed.columns:
                            batch_processed[feature] = 0

                    batch_X = batch_processed[st.session_state['feature_names']]

                    # Scale
                    batch_scaled = st.session_state['scaler'].transform(batch_X)

                    # Predict with both models
                    model_key = model_choice.lower().replace(' ', '')
                    predictions = st.session_state['results'][model_key]['model'].predict(batch_scaled)

                    # Add predictions to original dataframe
                    result_df = batch_df.copy()
                    result_df['Predicted_Weekly_Sales'] = predictions

                    st.success("✅ Batch predictions completed!")
                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"walmart_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Summary statistics
                    st.subheader("📊 Batch Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Predictions", len(predictions))
                    col2.metric("Average Predicted Sales", f"${predictions.mean():,.2f}")
                    col3.metric("Min Predicted Sales", f"${predictions.min():,.2f}")
                    col4.metric("Max Predicted Sales", f"${predictions.max():,.2f}")


if __name__ == "__main__":
    main()