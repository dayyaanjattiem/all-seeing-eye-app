import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
import warnings

# --- Page Configuration ---
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Title and Description ---
st.title("📈 Holt-Winters Stock Forecast")
st.markdown("""
This app downloads historical data for JSE (Johannesburg Stock Exchange) stocks 
and applies **Holt-Winters Exponential Smoothing** to forecast future prices.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

default_tickers = [
    "IMP.JO", "ANG.JO", "GFI.JO", "VAL.JO", "APN.JO", "NPH.JO", 
    "CLS.JO", "MRP.JO", "BHG.JO", "OMN.JO", "PPC.JO", "TBS.JO"
]

selected_tickers = st.sidebar.multiselect(
    "Select Tickers", 
    options=default_tickers, 
    default=default_tickers
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Trading Days)", 
    min_value=10, max_value=126, value=63, step=1
)

show_confidence = st.sidebar.checkbox("Show ±10% Confidence Interval", value=True)

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent rate limiting
def get_stock_data(ticker):
    """
    Downloads stock data. 
    """
    try:
        # We use a ticker object. 
        # On Cloud, sometimes simple downloading fails without session handling,
        # but yfinance 0.2.40+ handles this much better.
        stock = yf.Ticker(ticker)
        
        # Download 5 years of history
        data = stock.history(period="5y")
        
        if data.empty:
            return pd.DataFrame()
        
        # Ensure we have the Close column
        if 'Close' in data.columns:
            return data[['Close']]
        
        return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()

def run_forecast(data, days):
    try:
        model = ExponentialSmoothing(
            data['Close'], 
            trend='add', 
            seasonal=None, 
            initialization_method="estimated"
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(days)
        return forecast
    except Exception:
        return None

# --- Main Execution ---

if not selected_tickers:
    st.warning("Please select at least one ticker from the sidebar.")

else:
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(selected_tickers):
        progress_bar.progress((i + 1) / len(selected_tickers))
        
        st.divider()
        st.subheader(f"📊 {ticker}")

        # 1. Get Data
        data = get_stock_data(ticker)

        if data.empty:
            st.error(f"⚠️ Could not retrieve data for {ticker}. (Yahoo Finance might be rate-limiting the cloud server)")
            continue

        # 2. Run Forecast
        forecast = run_forecast(data, forecast_days)
        
        if forecast is None:
            st.warning(f"Could not generate forecast for {ticker}.")
            continue

        # 3. Prepare Dates
        last_date = data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

        # 4. Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(data.index, data['Close'], label='Historical Price', color='blue')
        ax.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='red')
        
        if show_confidence:
            ax.fill_between(
                forecast_dates, 
                forecast * 0.9, 
                forecast * 1.1, 
                color='red', 
                alpha=0.15, 
                label='Confidence Interval (±10%)'
            )

        ax.set_title(f"{ticker} - Holt-Winters Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (ZAR)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)

    progress_bar.empty()
    st.success("Processing complete.")
