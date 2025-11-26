import streamlit as st
import warnings
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Prophet Stock Forecaster", layout="wide", page_icon="🔮")

# --- 0. Environment Setup & Imports ---
# Cache the setup so it doesn't run on every interaction
@st.cache_resource
def setup_prophet_environment():
    """
    Attempts to setup Prophet and CmdStanPy backend.
    """
    status_container = st.empty()
    try:
        import cmdstanpy
        from cmdstanpy import install_cmdstan
        
        # Check if cmdstan is installed
        try:
            cmdstan_path = cmdstanpy.cmdstan_path()
        except ValueError:
            status_container.warning("⚠️ CmdStan not found. Installing backend... (This takes ~2 mins)")
            install_cmdstan()
            status_container.success("✅ Installation complete.")
            
        # Suppress logging
        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        from prophet import Prophet
        from prophet.plot import plot_plotly, plot_components_plotly
        return Prophet, plot_plotly, plot_components_plotly, True
        
    except ImportError as e:
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

        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=1, axis=1)
            except:
                df.columns = df.columns.get_level_values(0)

        # Format for Prophet: ds (Date) and y (Value)
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

        # Try adding SA Holidays
        try:
            model.add_country_holidays(country_name='ZA')
        except:
            pass # Fail silently if holidays can't load

        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_days)
        # Filter out weekends (Stock market closed)
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

# Sidebar Configuration
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
        st.error(f"Could not find data for {ticker_input}. Please checks symbol.")
    else:
        # Display Current Price
        current_price = df.iloc[-1]['y']
        
        with st.spinner("Training Prophet Model... (This uses heavy computation)"):
            model, forecast = run_prophet_forecast(df, forecast_days)

        if model is not None:
            # --- Analysis Logic ---
            future_data = forecast.iloc[-forecast_days:]
            start_future = future_data.iloc[0]['yhat']
            end_future = future_data.iloc[-1]['yhat']
            trend_pct = ((end_future - start_future) / start_future) * 100
            
            latest = forecast.iloc[-1]

            # --- Metrics Display ---
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"{current_price:.2f}")
            col2.metric("Predicted Price", f"{end_future:.2f}", delta=f"{end_future-current_price:.2f}")
            col3.metric("Expected Trend", f"{trend_pct:.2f}%")
            col4.metric("Yearly Seasonality", f"{latest['yearly']:.2f}")

            # --- Visualizations ---
            
            # Tabbed view for cleaner UI
            tab1, tab2, tab3 = st.tabs(["Interactive Forecast", "Components", "Raw Data"])

            with tab1:
                st.subheader("Interactive Price Forecast")
                # Use Plotly for interactive zooming
                fig_main = plot_plotly(model, forecast)
                st.plotly_chart(fig_main, use_container_width=True)

            with tab2:
                st.subheader("Seasonal Decomposition")
                st.markdown("Prophet breaks time series into trend, weekly, and yearly components.")
                [Image of additive time series decomposition]
                # Use Plotly for components
                fig_comp = plot_components_plotly(model, forecast)
                st.plotly_chart(fig_comp, use_container_width=True)

            with tab3:
                st.subheader("Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].tail(forecast_days))

            st.success("Forecast complete.")
