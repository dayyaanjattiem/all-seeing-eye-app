import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from datetime import timedelta
from config import FORECAST_DAYS_TS

@st.cache_data(show_spinner="🔄 Running K-Means Clustering...")
def run_clustering(df, n_clusters):
    if len(df) < n_clusters:
        st.warning(f"Not enough stocks ({len(df)}) for K={n_clusters}. Reducing K.")
        n_clusters = max(2, len(df) // 2)

    # Separate the features
    X_2D = df[['Volatility', 'Return']].values
    X_3D = df[['Volatility', 'Return', 'Price_to_Book']].values
    
    # 2D Clustering
    scaler_2d = StandardScaler()
    X_scaled_2D = scaler_2d.fit_transform(X_2D)
    kmeans_2d = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster_2D'] = kmeans_2d.fit_predict(X_scaled_2D)

    # 3D Clustering
    scaler_3d = StandardScaler()
    X_scaled_3D = scaler_3d.fit_transform(X_3D)
    kmeans_3d = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster_3D'] = kmeans_3d.fit_predict(X_scaled_3D)

    return df.copy()

def assess_valuation(features_df):
    """Generates the text-based valuation and suitability report."""
    median_pe = features_df['PE_Ratio'].median()
    median_pb = features_df['Price_to_Book'].median()
    median_de = features_df['Debt_to_Equity'].median()
    median_vol = features_df['Volatility'].median()
    median_ret = features_df['Return'].median()

    report_data = []

    for index, row in features_df.iterrows():
        # Valuation Assessment
        if row['PE_Ratio'] < median_pe and row['Price_to_Book'] < median_pb:
            valuation = "**Undervalued**"
        elif row['PE_Ratio'] > median_pe and row['Price_to_Book'] > median_pb:
            valuation = "**Overvalued**"
        else:
            valuation = "Fairly Valued/Mixed"

        # Risk Assessment
        if row['Volatility'] < median_vol and row['Debt_to_Equity'] < median_de:
            risk = "Low Risk / Low Leverage"
        elif row['Volatility'] > median_vol and row['Debt_to_Equity'] > median_de:
            risk = "**High Risk** / High Leverage"
        else:
            risk = "Moderate Risk"

        # Investment Suitability
        suitability = "Neutral"
        if 'Undervalued' in valuation and row['Return'] > median_ret:
            suitability = "**Strong BUY** (Value + Growth)"
        elif 'Undervalued' in valuation:
            suitability = "**Potential BUY** (Value Play)"
        elif 'Overvalued' in valuation and row['Volatility'] > median_vol:
            suitability = "**Avoid/Sell** (Overvalued + High Risk)"
            
        report_data.append({
            'Ticker': index,
            'Valuation': valuation,
            'Risk/Leverage': risk,
            'Suitability': suitability,
            'P/E Ratio': f"{row['PE_Ratio']:,.2f}",
            'P/B Ratio': f"{row['Price_to_Book']:,.2f}",
            'Debt/Equity': f"{row['Debt_to_Equity']:,.2f}",
            'Div. Yield': f"{row['Dividend_Yield'] * 100:,.2f}%",
            'Annual Return': f"{row['Return'] * 100:,.2f}%",
            'Annual Volatility': f"{row['Volatility'] * 100:,.2f}%",
        })

    return pd.DataFrame(report_data).set_index('Ticker')

# --- Time Series Analysis ---

def run_holt_winters_forecast(price_series, forecast_days=FORECAST_DAYS_TS):
    """Performs Holt-Winters (Exponential Smoothing) forecast."""
    try:
        # FIX: Assume price_series is already a Pandas Series (price column). 
        # Remove .to_frame(name='Close') which caused the error.
        data = price_series.dropna() 
        if data.empty or len(data) < 100:
            return pd.Series()

        # Holt-Winters expects a Series or a DataFrame with a single column.
        model = ExponentialSmoothing(data, trend='add', seasonal=None, initialization_method="estimated")
        model_fit = model.fit()

        forecast = model_fit.forecast(forecast_days)
        
        # Create forecast date index
        last_date = data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast.index = forecast_dates
        
        return forecast
    except Exception as e:
        # Note: Statsmodels needs at least two points to fit the trend
        st.warning(f"Holt-Winters failed: {e}")
        return pd.Series()

def analyze_correlation_beta(df, index_ticker):
    """Calculates correlation and Beta of stocks vs a benchmark index."""
    
    # 1. Normalize and Calculate Correlation
    normalized = df / df.iloc[0] * 100
    correlations = df.corr()[[index_ticker]].drop(index=index_ticker)
    correlations.columns = ['Correlation with Index']

    # 2. Compute Beta
    log_returns = np.log(df / df.shift(1)).dropna()
    betas = {}
    
    # Ensure index_ticker is in log_returns columns before looping
    if index_ticker in log_returns.columns:
        market_returns = log_returns[index_ticker]
        for ticker in log_returns.columns:
            if ticker != index_ticker:
                stock_returns = log_returns[ticker]
                
                # Check for zero variance in market returns (rare but possible in small subsets)
                if market_returns.var() > 1e-8: 
                    # Use OLS for Beta calculation
                    X = sm.add_constant(market_returns)
                    model = sm.OLS(stock_returns, X).fit()
                    beta = model.params.get(index_ticker, np.nan)
                    # Check if Beta is finite before assigning
                    betas[ticker] = beta if np.isfinite(beta) else np.nan
                else:
                    betas[ticker] = np.nan
    else:
        # If the index itself failed to load, we can't calculate beta/correlation
        st.warning(f"Index ticker {index_ticker} not available for Beta calculation.")
        
    beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta vs Index']).sort_index()
    
    # Combine results
    results = correlations.join(beta_df)
    
    return normalized, results
