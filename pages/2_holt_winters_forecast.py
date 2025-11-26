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

# Default Tickers
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
    min_value=10, 
    max_value=126, 
    value=63,
    step=1,
    help="63 days is approximately 3 months."
)

show_confidence = st.sidebar.checkbox("Show ±10% Confidence Interval", value=True)
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)

# --- Helper Functions ---

@st.cache_data
def get_stock_data(ticker, period="5y"):
    """Downloads and caches stock data."""
    data = yf.download(ticker, period=period, progress=False)
    # Handle multi-level columns if they exist (common in new yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.xs('Close', level=0, axis=1)
        except KeyError:
             # Fallback if structure is different
             pass
    
    # Ensure we have a 'Close' column or series
    if 'Close' in data.columns:
        return data[['Close']].dropna()
    elif isinstance(data, pd.Series):
        return data.to_frame(name='Close').dropna()
    else:
        return pd.DataFrame()

def run_forecast(data, days):
    """Fits the model and generates a forecast."""
    try:
        # Fit Holt-Winters Exponential Smoothing (Additive trend, no seasonality)
        model = ExponentialSmoothing(
            data['Close'], 
            trend='add', 
            seasonal=None, 
            initialization_method="estimated"
        )
        model_fit = model.fit()
        forecast = model_fit.forecast(days)
        return forecast
    except Exception as e:
        st.error(f"Model failed to fit: {e}")
        return None

# --- Main Execution ---

if not selected_tickers:
    st.warning("Please select at least one ticker from the sidebar.")

else:
    # create a placeholder for progress
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(selected_tickers):
        # Update progress bar
        progress_bar.progress((i + 1) / len(selected_tickers))
        
        st.divider()
        st.subheader(f"📊 {ticker}")

        # 1. Get Data
        with st.spinner(f"Downloading data for {ticker}..."):
            data = get_stock_data(ticker)

        if data.empty:
            st.error(f"⚠️ No data found for {ticker}")
            continue

        # 2. Run Forecast
        forecast = run_forecast(data, forecast_days)
        
        if forecast is None:
            continue

        # 3. Prepare Dates for Plotting
        last_date = data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

        # 4. Plotting using Object-Oriented Matplotlib
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical Data
        ax.plot(data.index, data['Close'], label='Historical Price', color='blue')
        
        # Forecast Data
        ax.plot(forecast_dates, forecast, label='Forecast', linestyle='--', color='red')
        
        # Confidence Interval (Visual approximation as per original code)
        if show_confidence:
            ax.fill_between(
                forecast_dates, 
                forecast * 0.9, 
                forecast * 1.1, 
                color='red', 
                alpha=0.15, 
                label='Confidence Interval (±10%)'
            )

        ax.set_title(f"{ticker} - Holt-Winters Forecast (Next {forecast_days} Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Display the plot in Streamlit
        st.pyplot(fig)

        # Optional: Show data table
        if show_raw_data:
            with st.expander(f"See raw data for {ticker}"):
                col1, col2 = st.columns(2)
                col1.write("Historical (Last 5 rows)")
                col1.dataframe(data.tail())
                col2.write("Forecast (First 5 rows)")
                col2.dataframe(pd.DataFrame({"Forecast": forecast.values}, index=forecast_dates).head())

    # Complete
    progress_bar.empty()
    st.success("All processing complete.")
