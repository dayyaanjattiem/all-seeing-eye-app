import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from config import FORECAST_DAYS_TS # Ensure you have this in your config file

# --- IMPORTING LOGIC FROM SEPARATE FILE ---
from analysis_logic import (
    run_clustering,
    assess_valuation,
    run_holt_winters_forecast,
    analyze_correlation_beta,
    plot_correlation_beta
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

def load_your_data():
    """
    ⚠️ CRITICAL: THIS IS WHERE YOU LOAD YOUR DATA.
    Replace this with your actual SQL query, CSV load, or API call.
    """
    # st.write("Loading data...") # Uncomment for debugging
    
    # EXAMPLE DUMMY DATA (DELETE THIS BLOCK WHEN YOU HAVE REAL DATA)
    # This is just so the app runs right now to show you it works.
    dates = pd.bdate_range(end='2023-01-01', periods=200)
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', '^J141.JO']
    data = pd.DataFrame(index=dates)
    for t in tickers:
        data[t] = np.random.normal(100, 15, size=len(dates)).cumsum() + 100
        
    # We need a feature dataframe for clustering (usually calculated from price data)
    features_df = pd.DataFrame(index=tickers[:-1]) # Exclude index from features
    features_df['PE_Ratio'] = np.random.uniform(10, 50, len(tickers[:-1]))
    features_df['Price_to_Book'] = np.random.uniform(1, 10, len(tickers[:-1]))
    features_df['Debt_to_Equity'] = np.random.uniform(0.1, 2.0, len(tickers[:-1]))
    features_df['Volatility'] = np.random.uniform(0.1, 0.5, len(tickers[:-1]))
    features_df['Return'] = np.random.uniform(-0.2, 0.4, len(tickers[:-1]))
    features_df['Dividend_Yield'] = np.random.uniform(0, 0.05, len(tickers[:-1]))
    
    return data, features_df 

def main():
    st.title("👁️ All-Seeing Eye: Stock Analysis")
    st.markdown("---")

    # 1. LOAD DATA
    # -------------------------------------------
    try:
        # Calls the function above. Replace with your real data loader.
        price_df, features_df = load_your_data()
        
        if price_df is None or features_df is None:
            st.error("Data could not be loaded.")
            return
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar Options
    st.sidebar.header("Settings")
    benchmark_ticker = st.sidebar.text_input("Benchmark Ticker", value="^J141.JO")
    
    # 2. CLUSTERING & VALUATION
    # -------------------------------------------
    st.subheader("1. Market Segmentation & Valuation")
    
    # Run Clustering
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    clustered_df = run_clustering(features_df, n_clusters)
    
    # Assess Valuation
    valuation_report = assess_valuation(clustered_df)
    
    # Display Report
    st.dataframe(valuation_report.style.applymap(
        lambda x: 'color: green' if 'BUY' in str(x) else ('color: red' if 'Sell' in str(x) else ''),
        subset=['Suitability']
    ), use_container_width=True)

    st.markdown("---")

    # 3. TIME SERIES FORECASTING
    # -------------------------------------------
    st.subheader("2. Price Forecasting (Holt-Winters)")
    
    selected_ticker = st.selectbox("Select Stock for Forecast", options=features_df.index)
    
    if selected_ticker:
        if selected_ticker in price_df.columns:
            # Get the single column series for the stock
            stock_series = price_df[selected_ticker]
            
            # Run Forecast
            forecast = run_holt_winters_forecast(stock_series)
            
            if not forecast.empty:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.line_chart(pd.concat([stock_series.tail(100), forecast]))
                with col2:
                    st.write("Forecasted Values:")
                    st.write(forecast.head())
            else:
                st.warning("Could not generate forecast (insufficient data).")
        else:
            st.warning(f"Price data not found for {selected_ticker}")

    st.markdown("---")

    # 4. CORRELATION & BETA ANALYSIS
    # -------------------------------------------
    st.subheader(f"3. Correlation & Beta vs {benchmark_ticker}")
    
    if benchmark_ticker in price_df.columns:
        # Run Analysis
        normalized_prices, corr_beta_results = analyze_correlation_beta(price_df, benchmark_ticker)
        
        # Plot Results
        # Note: We pass the dataframes we just created into the plotting function
        fig = plot_correlation_beta(normalized_prices, benchmark_ticker, corr_beta_results)
        
        st.pyplot(fig)
    else:
        st.error(f"Benchmark ticker '{benchmark_ticker}' not found in price data columns.")

# Execution entry point
if __name__ == "__main__":
    main()
