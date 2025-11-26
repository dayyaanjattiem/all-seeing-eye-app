import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from datetime import timedelta
import matplotlib.pyplot as plt
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

# --- FIXED Time Series Analysis ---

def run_holt_winters_forecast(price_data, forecast_days=FORECAST_DAYS_TS):
    """Performs Holt-Winters (Exponential Smoothing) forecast."""
    try:
        # Handle both Series and DataFrame inputs
        if isinstance(price_data, pd.DataFrame):
            # If DataFrame, take the first column (assuming it's the price column)
            data = price_data.iloc[:, 0].dropna()
        else:
            # If Series, use directly
            data = price_data.dropna()
            
        if data.empty or len(data) < 10:  # Reduced minimum requirement
            st.warning(f"Insufficient data for forecast: {len(data)} points")
            return pd.Series()

        # Holt-Winters expects a Series
        model = ExponentialSmoothing(data, trend='add', seasonal=None, initialization_method="estimated")
        model_fit = model.fit()

        forecast = model_fit.forecast(forecast_days)
        
        # Create forecast date index
        last_date = data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast.index = forecast_dates
        
        return forecast
        
    except Exception as e:
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

# --- ADDED Missing Function ---

def plot_correlation_beta(normalized_df, index_ticker, corr_beta_df):
    """Plots correlation and beta visualization."""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Normalized Price Performance
    for col in normalized_df.columns:
        ax1.plot(normalized_df.index, normalized_df[col], alpha=0.7, linewidth=1.5, label=col)
    ax1.set_title('Normalized Price Performance (Base = 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Correlation Heatmap
    if len(corr_beta_df) > 0 and 'Correlation with Index' in corr_beta_df.columns:
        corr_values = corr_beta_df['Correlation with Index'].values.reshape(-1, 1)
        im = ax2.imshow(corr_values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_yticks(range(len(corr_beta_df)))
        ax2.set_yticklabels(corr_beta_df.index, fontsize=10)
        ax2.set_xticks([0])
        ax2.set_xticklabels([f'Corr. with {index_ticker}'])
        ax2.set_title('Correlation with Index', fontsize=14, fontweight='bold')
        
        # Add correlation values as text
        for i, corr in enumerate(corr_values):
            ax2.text(0, i, f'{corr[0]:.2f}', ha='center', va='center', 
                    color='white' if abs(corr[0]) > 0.5 else 'black', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
    else:
        ax2.text(0.5, 0.5, 'No Correlation Data', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=14)
        ax2.set_title('Correlation with Index', fontsize=14, fontweight='bold')
    
    # 3. Beta vs Correlation Scatter
    if 'Beta vs Index' in corr_beta_df.columns:
        valid_data = corr_beta_df.dropna()
        if len(valid_data) > 0:
            ax3.scatter(valid_data['Correlation with Index'], 
                       valid_data['Beta vs Index'],
                       alpha=0.7, s=60, c='steelblue', edgecolors='black')
            ax3.set_xlabel('Correlation with Index')
            ax3.set_ylabel('Beta vs Index')
            ax3.set_title('Beta vs Correlation Analysis', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add reference lines
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Beta = 1')
            ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax3.legend()
            
            # Annotate points
            for idx, row in valid_data.iterrows():
                ax3.annotate(idx, (row['Correlation with Index'], row['Beta vs Index']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No Valid Beta Data', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=14)
    else:
        ax3.text(0.5, 0.5, 'No Beta Data', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=14)
    ax3.set_title('Beta vs Correlation Analysis', fontsize=14, fontweight='bold')
    
    # 4. Beta Distribution
    if 'Beta vs Index' in corr_beta_df.columns:
        beta_values = corr_beta_df['Beta vs Index'].dropna()
        if len(beta_values) > 0:
            ax4.hist(beta_values, bins=min(10, len(beta_values)), alpha=0.7, 
                    color='lightcoral', edgecolor='black')
            ax4.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Beta = 1 (Market)')
            ax4.axvline(x=beta_values.mean(), color='blue', linestyle='-', linewidth=2, 
                       label=f'Mean = {beta_values.mean():.2f}')
            ax4.set_xlabel('Beta')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Valid Beta Data', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'No Beta Data', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=14)
    ax4.set_title('Beta Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig
