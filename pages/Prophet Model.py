import streamlit as st
import warnings
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

# -----------------------------
# ✅ SAFE PROPHET IMPORT (NO RUNTIME INSTALLS)
# -----------------------------
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False
    st.error(f"Prophet Import Error: {e}")

warnings.filterwarnings("ignore")

# -----------------------------
# ✅ PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Prophet Stock Forecaster",
    layout="wide",
    page_icon="🔮"
)

# -----------------------------
# ✅ DATA DOWNLOAD (CACHED)
# -----------------------------
@st.cache_data(ttl=3600)
def get_data_for_prophet(ticker, start_date):
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
        st.error(f"Data Download Error: {e}")
        return None

# -----------------------------
# ✅ FORECAST FUNCTION
# -----------------------------
def run_prophet_forecast(df, forecast_days):
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
        st.error(f"Prophet Training Failed: {e}")
        return None, None

# -----------------------------
# ✅ APP UI
# -----------------------------
st.title("🔮 Prophet Stock Forecaster")
st.markdown("Automated time-series forecasting using **Facebook Prophet**.")

if not PROPHET_AVAILABLE:
    st.error("❌ Prophet is not available. Please ensure it is installed via requirements.txt.")
    st.stop()

# -----------------------------
# ✅ SIDEBAR
# -----------------------------
st.sidebar.header("Configuration")

ticker_input = st.sidebar.text_input("Ticker Symbol", value="STXSHA.JO")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 60)
run_btn = st.sidebar.button("Run Forecast", type="primary")

# -----------------------------
# ✅ EXECUTION
# -----------------------------
if run_btn:

    with st.spinner(f"Downloading data for {ticker_input}..."):
        df = get_data_for_prophet(ticker_input, start_date)

    if df is None:
        st.error("Invalid ticker or no historical data available.")
        st.stop()

    current_price = df.iloc[-1]['y']

    with st.spinner("Training Prophet Model..."):
        model, forecast = run_prophet_forecast(df, forecast_days)

    if model is None:
        st.stop()

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

    tab1, tab2, tab3 = st.tabs(["Interactive Forecast", "Components", "Raw Data"])

    with tab1:
        st.subheader("Interactive Price Forecast")
        fig_main = plot_plotly(model, forecast)
        st.plotly_chart(fig_main, use_container_width=True)

    with tab2:
        st.subheader("Seasonal Decomposition")
        fig_comp = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab3:
        st.subheader("Forecast Data")
        st.dataframe(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
            .tail(forecast_days)
        )

    st.success("✅ Forecast Complete!")
