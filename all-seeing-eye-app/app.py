import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import yfinance as yf # Needed for direct ticker data in TS analysis

# Import all modules
# NOTE: Ensure config.py, data_handler.py, analysis_logic.py, and plotting.py are in the same directory.
from config import DEFAULT_STOCKS, START_DATE, END_DATE, N_CLUSTERS, SHARIA_INDEX, TS_TICKERS, FORECAST_DAYS_TS
from data_handler import fetch_all_data, fetch_index_data
from analysis_logic import run_clustering, assess_valuation, run_holt_winters_forecast, analyze_correlation_beta
from plotting import (plot_clustering_2d, plot_clustering_3d, plot_valuation, 
                      plot_risk_assessment, plot_daily_returns_distribution, 
                      plot_holt_winters, plot_correlation_beta)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
plt.style.use('ggplot')

# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="All Seeing Eye - Financial Models")

# --- Sidebar Inputs ---
st.sidebar.title("📈 All Seeing Eye Config")

# Ticker Input
custom_stocks = st.sidebar.text_area(
    "Enter Tickers for K-Means/Fundamental Analysis (comma separated)", 
    ", ".join(DEFAULT_STOCKS)
)
input_tickers = [t.strip().upper() for t in custom_stocks.split(',') if t.strip()]

# K-Means Input
k_clusters = st.sidebar.slider("Number of K-Means Clusters (K)", 2, 12, N_CLUSTERS)

# Holt-Winters Ticker Select
holt_ticker = st.sidebar.selectbox("Select Ticker for Holt-Winters Forecast", TS_TICKERS)


# ==========================================
# DATA FETCHING
# ==========================================

log_returns, features_df = fetch_all_data(input_tickers, START_DATE, END_DATE)

if features_df.empty:
    st.error("Cannot proceed. No valid data for analysis after cleaning. Please check your ticker symbols and date range.")
    st.stop()
    
features_clustered_df = run_clustering(features_df, k_clusters)
report_df = assess_valuation(features_clustered_df)

# --- Extract Medians for Plotting ---
median_pe = features_clustered_df['PE_Ratio'].median()
median_pb = features_clustered_df['Price_to_Book'].median()
median_de = features_clustered_df['Debt_to_Equity'].median()
median_vol = features_clustered_df['Volatility'].median()
median_ret = features_clustered_df['Return'].median()

# Merge numeric columns for plotting functions
report_df['PE_Ratio'] = features_clustered_df['PE_Ratio']
report_df['P/B_Ratio'] = features_clustered_df['Price_to_Book']
report_df['Debt/Equity_Raw'] = features_clustered_df['Debt_to_Equity']
report_df['Volatility_Numeric'] = features_clustered_df['Volatility']
report_df['Return_Numeric'] = features_clustered_df['Return']


# ==========================================
# PAGE 1: K-MEANS & FUNDAMENTALS
# ==========================================

st.title("1️⃣ K-Means Clustering and Fundamental Analysis")

st.markdown("---")

# --- Clustering Visuals ---
st.header("1.1 Risk-Return Clustering")
col_c1, col_c2 = st.columns(2)

with col_c1:
    fig_2d = plot_clustering_2d(features_clustered_df, k_clusters)
    st.pyplot(fig_2d)

with col_c2:
    fig_3d = plot_clustering_3d(features_clustered_df, k_clusters)
    st.pyplot(fig_3d)

st.markdown("---")

# --- Valuation Visuals ---
st.header("1.2 Valuation and Risk Metrics")
col_v1, col_v2, col_v3 = st.columns(3)

with col_v1:
    fig_pe = plot_valuation(report_df, 'PE_Ratio', 'P/E Ratio', median_pe, median_ret, 'teal', 'P/E Valuation vs. Return')
    st.pyplot(fig_pe)

with col_v2:
    fig_pb = plot_valuation(report_df, 'P/B_Ratio', 'P/B Ratio', median_pb, median_ret, 'purple', 'P/B Valuation vs. Return')
    st.pyplot(fig_pb)

with col_v3:
    fig_risk = plot_risk_assessment(report_df, median_de, median_vol)
    st.pyplot(fig_risk)

st.markdown("---")

# --- Distribution Plot ---
st.header("1.3 Daily Returns Distribution (Normality Check)")
if not log_returns.empty:
    fig_kde = plot_daily_returns_distribution(log_returns)
    st.pyplot(fig_kde)

st.markdown("---")

# --- Recommendation Report ---
st.header("1.4 Investment Recommendation Summary")
st.dataframe(report_df)

# Final conclusive statement
strong_buys = report_df[report_df['Suitability'].str.contains('Strong BUY')].index.tolist()
undervalued = report_df[report_df['Valuation'].str.contains('Undervalued')].index.tolist()
overvalued = report_df[report_df['Valuation'].str.contains('Overvalued')].index.tolist()

st.success(f"🔥 **Top Strong BUY Candidates:** {', '.join(strong_buys) if strong_buys else 'None found.'}")
st.info(f"💰 **General Undervalued Candidates:** {', '.join(undervalued) if undervalued else 'None found.'}")
st.error(f"🛑 **Most Overvalued/Risky Stocks to Avoid:** {', '.join(overvalued) if overvalued else 'None found.'}")


# ==========================================
# PAGE 2: TIME SERIES & CORRELATION
# ==========================================

st.title("2️⃣ Time Series and Market Correlation Analysis")
st.markdown("---")

# --- Holt-Winters Forecast ---
st.header(f"2.1 Holt-Winters Price Forecast: {holt_ticker}")

# Line 153: Fetch full price history (FIXED WITH DEFENSIVE CHECKS)
raw_data = yf.download(holt_ticker, start="2020-01-01", progress=False, auto_adjust=False)

# Defensive Check 1: Ensure raw_data is a valid DataFrame
if raw_data is None or raw_data.empty:
    st.warning(f"Could not download any data for ticker: {holt_ticker}. Skipping Holt-Winters forecast.")
else:
    # Defensive Check 2: Select the best price column
    if 'Adj Close' in raw_data.columns:
        ticker_data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        ticker_data = raw_data['Close']
    else:
        st.error(f"Price data (Adj Close or Close) not found for {holt_ticker}. Skipping forecast.")
        ticker_data = pd.Series() # Empty series placeholder
        
    # Only run forecast if data is available
    if not ticker_data.empty:
        forecast_series = run_holt_winters_forecast(ticker_data, FORECAST_DAYS_TS)
        
        if not forecast_series.empty:
            fig_hw = plot_holt_winters(holt_ticker, ticker_data, forecast_series)
            st.pyplot(fig_hw)
        else:
            st.warning("Could not generate Holt-Winters forecast for this ticker.")
    
st.markdown("---")

# --- Correlation & Beta Analysis ---
st.header(f"2.2 Correlation and Beta vs. {SHARIA_INDEX}")

index_df = fetch_index_data(input_tickers, SHARIA_INDEX)

if not index_df.empty and SHARIA_INDEX in index_df.columns:
    normalized_df, corr_beta_df = analyze_correlation_beta(index_df, SHARIA_INDEX)

    col_corr1, col_corr2 = st.columns([2, 1])
    
    with col_corr1:
        fig_corr = plot_correlation_beta(normalized_df, SHARIA_INDEX, corr_beta_df)
        st.pyplot(fig_corr)

    with col_corr2:
        st.subheader("Correlation & Beta Metrics")
        st.dataframe(corr_beta_df.style.format("{:.4f}"))
        st.markdown(f"""
        - **Correlation**: Measures similarity of price movement (1.0 = perfect match).
        - **Beta**: Measures volatility relative to the index ({SHARIA_INDEX}). 
          * **Beta > 1.0**: More volatile/higher risk than the market.
          * **Beta < 1.0**: Less volatile/lower risk than the market.
        """)
elif not index_df.empty:
     st.warning(f"Could not fetch data for the index ticker ({SHARIA_INDEX}) from Yahoo Finance.")
else:
    st.warning("Could not fetch necessary price data for Correlation/Beta analysis.")
