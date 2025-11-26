import streamlit as st
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import timedelta
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Page Config ---
st.set_page_config(page_title="Advanced Stock Forecaster", layout="wide", page_icon="📈")

# --- Robust Import for GARCH ---
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. HELPER FUNCTIONS (Cached)
# ==========================================

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date):
    """Downloads historical stock data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, progress=False)

        # Handle multi-index columns if they exist (common in new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try to select the ticker level
                df = df.xs(ticker, level=1, axis=1)
            except:
                # Fallback: drop the ticker level
                df.columns = df.columns.get_level_values(0)

        if df.empty:
            return None

        # Ensure we have a Close column
        if 'Close' not in df.columns:
            return None
            
        return df[['Close']].dropna()
    except Exception as e:
        return None

def train_test_split_series(series, test_ratio):
    split_idx = int(len(series) * (1 - test_ratio))
    return series.iloc[:split_idx], series.iloc[split_idx:]

def create_ml_dataset(series, window=5):
    X, y = [], []
    vals = series.values
    for i in range(len(vals) - window):
        X.append(vals[i:i+window])
        y.append(vals[i+window])
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    """Calculates RMSE and MAE."""
    common_idx = y_true.index.intersection(y_pred.index)
    if len(common_idx) == 0:
        return np.nan, np.nan

    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]

    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    return rmse, mae

# ==========================================
# 2. MODELING FUNCTIONS
# ==========================================

def run_arima_backtest(train, test, order=(5,1,0)):
    history = [x for x in train]
    predictions = []
    
    # Simple progress bar for ARIMA since it can be slow
    progress_bar = st.progress(0)
    total_steps = len(test)
    
    try:
        # Fit initial model
        model = ARIMA(history, order=order)
        model_fit = model.fit()

        for t in range(total_steps):
            yhat = model_fit.forecast(steps=1)[0]
            predictions.append(yhat)
            history.append(test.iloc[t])
            # Re-fit/Append (Fast method)
            model_fit = model_fit.append([test.iloc[t]], refit=False)
            
            # Update progress
            if t % 10 == 0:
                progress_bar.progress((t + 1) / total_steps)
                
        progress_bar.empty()

    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        return pd.Series([np.nan]*len(test), index=test.index)
        
    return pd.Series(predictions, index=test.index)

def run_ml_backtest_returns(series, test_ratio, window):
    log_returns = np.log(series / series.shift(1)).dropna()
    X, y = create_ml_dataset(log_returns, window)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = y[:split_idx]

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    pred_returns = gb.predict(X_test)

    # Reconstruct prices
    test_indices = log_returns.index[window + split_idx :]
    final_preds = []
    for i, date in enumerate(test_indices):
        prev_date_loc = series.index.get_loc(date) - 1
        prev_price = series.iloc[prev_date_loc]
        pred_price = prev_price * np.exp(pred_returns[i])
        final_preds.append(pred_price)

    return pd.Series(final_preds, index=test_indices), gb

def run_monte_carlo(series, days, simulations):
    log_returns = np.log(1 + series.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    last_price = series.iloc[-1]
    dates = [series.index[-1] + timedelta(days=i) for i in range(1, days + 1)]
    # Filter for weekdays roughly
    dates = [d for d in dates if d.weekday() < 5]
    steps = len(dates)

    simulation_df = pd.DataFrame(index=dates)
    final_prices = []

    for i in range(simulations):
        Z = norm.ppf(np.random.rand(steps))
        daily_returns = np.exp(drift + stdev * Z)
        price_path = np.zeros(steps)
        price_path[0] = last_price * daily_returns[0]
        for t in range(1, steps):
            price_path[t] = price_path[t-1] * daily_returns[t]

        simulation_df[f'Sim_{i}'] = price_path
        final_prices.append(price_path[-1])

    return simulation_df, final_prices

def run_future_ml_forecast(series, ml_model, steps, lag_window):
    last_date = series.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
    future_dates = [d for d in future_dates if d.weekday() < 5]
    actual_steps = len(future_dates)

    try:
        log_returns = np.log(series / series.shift(1)).dropna()
        current_window_ret = list(log_returns.values[-lag_window:])
        current_price = series.iloc[-1]
        ml_preds_price = []
        
        for _ in range(actual_steps):
            input_feat = np.array([current_window_ret])
            pred_ret = ml_model.predict(input_feat)[0]
            next_price = current_price * np.exp(pred_ret)
            ml_preds_price.append(next_price)
            current_price = next_price
            
            # Update window
            current_window_ret.pop(0)
            current_window_ret.append(pred_ret)
            
        return pd.Series(ml_preds_price, index=future_dates)
    except:
        return pd.Series()

# ==========================================
# 3. MAIN UI
# ==========================================

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

ticker_input = st.sidebar.text_input("Ticker Symbol (Yahoo Finance)", value="OMN.JO")
start_date_input = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
forecast_days = st.sidebar.slider("Forecast Days", 10, 120, 60)
sims = st.sidebar.slider("Monte Carlo Simulations", 50, 1000, 200)

st.sidebar.markdown("---")
use_arima = st.sidebar.checkbox("Run ARIMA (Slow)", value=False)
st.sidebar.markdown("---")

if not HAS_ARCH:
    st.sidebar.warning("⚠️ 'arch' library not found. GARCH skipped.")

run_btn = st.sidebar.button("Run Analysis", type="primary")

# --- Main Analysis Flow ---
if run_btn:
    st.title(f"📊 Analysis: {ticker_input}")
    
    # 1. Load Data
    with st.spinner(f"Downloading data for {ticker_input}..."):
        df = get_stock_data(ticker_input, start_date_input)
    
    if df is None:
        st.error(f"❌ No data found for {ticker_input}. Please check the symbol.")
    else:
        series = df['Close']
        train, test = train_test_split_series(series, 0.2)
        lag_window = 5
        
        # 2. Backtesting
        col1, col2 = st.columns(2)
        
        # ML Backtest
        with st.spinner("Training Gradient Boosting Model..."):
            bt_ml, trained_gb = run_ml_backtest_returns(series, 0.2, lag_window)
            rmse_ml, mae_ml = calculate_metrics(test, bt_ml)
            
            with col1:
                st.subheader("ML (Gradient Boost)")
                st.metric("RMSE", f"{rmse_ml:.2f}")
                st.metric("MAE", f"{mae_ml:.2f}")

        # ARIMA Backtest (Optional)
        bt_arima = None
        if use_arima:
            with st.spinner("Running ARIMA Backtest..."):
                bt_arima = run_arima_backtest(train, test)
                rmse_ar, mae_ar = calculate_metrics(test, bt_arima)
                with col2:
                    st.subheader("ARIMA")
                    st.metric("RMSE", f"{rmse_ar:.2f}")
                    st.metric("MAE", f"{mae_ar:.2f}")

        # 3. Future Forecasting
        with st.spinner("Generating Future Scenarios..."):
            future_ml = run_future_ml_forecast(series, trained_gb, forecast_days, lag_window)
            mc_sims, final_prices = run_monte_carlo(series, forecast_days, sims)

        # 4. Probability Stats
        current_price = series.iloc[-1]
        mu, std = norm.fit(final_prices)
        prob_increase = norm.sf(current_price, loc=mu, scale=std) * 100
        ci_lower = norm.ppf(0.025, mu, std)
        ci_upper = norm.ppf(0.975, mu, std)

        st.markdown("### 🎲 Probability Analysis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"{current_price:.2f}")
        m2.metric("Expected Mean", f"{mu:.2f}", delta=f"{mu-current_price:.2f}")
        m3.metric("Prob. Increase", f"{prob_increase:.1f}%")
        m4.metric("95% CI Range", f"{ci_lower:.1f} - {ci_upper:.1f}")

        # 5. Visualization (Matplotlib)
        st.markdown("### 📈 Forecast Visualization")
        
        fig = plt.figure(figsize=(14, 7))
        gs = gridspec.GridSpec(1, 4) 

        # --- Main Chart ---
        ax1 = plt.subplot(gs[0, :3])
        
        # History (Zoomed in to last 150 days)
        ax1.plot(train.index[-150:], train[-150:], label="History", color='gray', alpha=0.5)
        ax1.plot(test.index, test, label="Actual (Test)", color='black', linewidth=2)
        
        # Plot ML Backtest
        ax1.plot(bt_ml.index, bt_ml, label="ML Backtest", linestyle=':', color='green', alpha=0.9)
        
        # Plot ARIMA Backtest
        if bt_arima is not None:
             ax1.plot(bt_arima.index, bt_arima, label="ARIMA Backtest", linestyle=':', color='orange', alpha=0.9)

        # Plot Monte Carlo
        ax1.plot(mc_sims.index, mc_sims, color='blue', alpha=0.03, linewidth=1)
        # Dummy plot for legend
        ax1.plot([], [], color='blue', alpha=0.3, label=f'Monte Carlo ({sims} Sims)')

        # Plot ML Forecast
        ax1.plot(future_ml.index, future_ml, label="ML Mean Forecast", color='red', linestyle='--', linewidth=2)

        ax1.set_title(f"Forecast Overview: {ticker_input}")
        ax1.legend(loc='upper left', fontsize='small')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Price")

        # --- Density Chart ---
        ax2 = plt.subplot(gs[0, 3])
        x_range = np.linspace(min(final_prices), max(final_prices), 100)
        p = norm.pdf(x_range, mu, std)
        
        ax2.hist(final_prices, bins=15, density=True, alpha=0.6, color='blue', orientation='horizontal')
        ax2.plot(p, x_range, 'r', linewidth=2, label='Normal Dist')
        ax2.axhline(y=current_price, color='black', linestyle='--', label='Current Price')
        
        ax2.set_title(f"Price Distribution\n(+{forecast_days} Days)")
        ax2.set_xlabel("Probability")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize='small')

        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Enter a ticker symbol and click 'Run Analysis' to start.")
