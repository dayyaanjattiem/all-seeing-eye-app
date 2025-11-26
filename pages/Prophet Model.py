import streamlit as st
import warnings
import pandas as pd
import yfinance as yf
import logging
import os
import shutil
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Prophet Stock Forecaster", layout="wide", page_icon="🔮")

# --- 0. Environment Setup & Imports ---
@st.cache_resource
def setup_prophet_environment():
    """
    Installs CmdStan to /tmp to avoid permission/path errors on Cloud.
    Safe for GitHub/Streamlit Cloud deployment.
    """
    status_container = st.empty()
    
    try:
        import cmdstanpy
        
        # 1. Force installation to /tmp (safer for Cloud environments)
        # This prevents the "No such file" error by keeping the install away from app.py
        base_tmp_dir = "/tmp" 
        cmdstan_install_dir = os.path.join(base_tmp_dir, "cmdstan_install")
        
        # Check if already installed in our temp dir
        found_paths = []
        if os.path.exists(cmdstan_install_dir):
            found_paths = [p for p in os.listdir(cmdstan_install_dir) if "cmdstan-" in p]
            
        if found_paths:
            # Found existing install in tmp
            cmdstan_path = os.path.join(cmdstan_install_dir, found_paths[0])
            os.environ['CMDSTAN'] = cmdstan_path
            cmdstanpy.set_cmdstan_path(cmdstan_path)
        else:
            # Install needed
            status_container.warning("⚠️ Installing Prophet backend... (This happens once, takes ~2 mins)")
            
            # Create dir if not exists
            os.makedirs(cmdstan_install_dir, exist_ok=True)
            
            # Install
            from cmdstanpy import install_cmdstan
            # We explicitly set the dir to /tmp/cmdstan_install
            install_cmdstan(dir=cmdstan_install_dir)
            
            # Set path explicitly after install
            found_paths = [p for p in os.listdir(cmdstan_install_dir) if "cmdstan-" in p]
            if found_paths:
                cmdstan_path = os.path.join(cmdstan_install_dir, found_paths[0])
                os.environ['CMDSTAN'] = cmdstan_path
                cmdstanpy.set_cmdstan_path(cmdstan_path)

            status_container.success("✅ Backend installation complete.")

        # 2. Suppress excessive logging
        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        # 3. Import Prophet
        from prophet import Prophet
        from prophet.plot import plot_plotly, plot_components_plotly
        return Prophet, plot_plotly, plot_components_plotly, True
        
    except Exception as e:
        status_container.error(f"Setup Error: {e}")
        return None, None, None, False

# Initialize Environment
Prophet, plot_plotly, plot_components_plotly, PROPHET_AVAILABLE = setup_prophet_environment()

warnings.filterwarnings("ignore")

# --- 1. Helper Functions (Cached) ---

@st.cache_data(ttl=3600)
def get_data_for_prophet(ticker, start_date):
    """Downloads and formats data for Prophet."""
    try:
        df = yf.download(ticker, start=start_date, progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=1, axis=1)
            except:
                df.columns = df.columns.get_level_values(0)

        if 'Close' not in df.columns:
            return None
            
        df_prophet = df[['Close']].reset_index()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

        return df_prophet
    except Exception as e:
        st.error(f"Data download error: {e}")
        return None

def run_prophet_forecast(df, forecast_days):
    """Trains the model and generates forecast."""
    try:
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        try:
            model.add_country_holidays(country_name='ZA')
        except:
            pass 

        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_days)
        future['day'] = future['ds'].dt.dayofweek
        future = future[future['day'] < 5]

        forecast = model.predict(future)
        return model, forecast

    except Exception as e:
        st.error(f"Prophet Training failed: {e}")
        return None, None

# --- 2. Main Interface ---

st.title("🔮 Prophet Stock Forecaster")
st.markdown("Automated time-series forecasting using **Facebook Prophet**.")

if not PROPHET_AVAILABLE:
    st.error("❌ The 'prophet' or 'cmdstanpy' library could not be loaded. Please ensure they are in your requirements.txt.")
    st.stop()

# Sidebar
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="STXSHA.JO")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 60)
run_btn = st.sidebar.button("Run Forecast", type="primary")

# Execution
if run_btn:
    with st.spinner(f"Downloading data for {ticker_input}..."):
        df = get_data_for_prophet(ticker_input, start_date)

    if df is None:
        st.error(f"Could not find data for {ticker_input}. Please check symbol.")
    else:
        current_price = df.iloc[-1]['y']
        
        with st.spinner("Training Prophet Model... (This uses heavy computation)"):
            model, forecast = run_prophet_forecast(df, forecast_days)

        if model is not None:
            future_data = forecast.iloc[-forecast_days:]
            start_future = future_data.iloc[0]['yhat']
            end_future = future_data.iloc[-1]['yhat']
            trend_pct = ((end_future - start_future) / start_future) * 100
            latest = forecast.iloc[-1]

            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"{current_price:.2f}")
            col2.metric("Predicted Price", f"{end_future:.2f}", delta=f"{end_future-current_price:.2f}")
            col3.metric("Expected Trend", f"{trend_pct:.2f}%")
            col4.metric("Yearly Seasonality", f"{latest['yearly']:.2f}")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["Interactive Forecast", "Components", "Raw Data"])

            with tab1:
                st.subheader("Interactive Price Forecast")
                fig_main = plot_plotly(model, forecast)
                st.plotly_chart(fig_main, use_container_width=True)

            with tab2:
                st.subheader("Seasonal Decomposition")
                st.markdown("Prophet breaks time series into trend, weekly, and yearly components.")
                fig_comp = plot_components_plotly(model, forecast)
                st.plotly_chart(fig_comp, use_container_width=True)

            with tab3:
                st.subheader("Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].tail(forecast_days))

            st.success("Forecast complete.")
