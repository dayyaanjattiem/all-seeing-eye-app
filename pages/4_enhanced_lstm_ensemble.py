import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings

# --- Page Config ---
st.set_page_config(page_title="LSTM Stock AI", layout="wide")
warnings.filterwarnings("ignore")

# --- Dependencies & Imports ---
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

# Attempt to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                         Bidirectional, GRU)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("❌ TensorFlow is not installed. Please add 'tensorflow' to requirements.txt")

# Attempt to import Technical Analysis library
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    st.warning("⚠️ 'ta' library not found. Using basic calculation fallback.")

# --- The Enhanced Predictor Class ---
class EnhancedStockPredictor:
    def __init__(self, window_size=60, forecast_days=21, ensemble_size=3):
        self.window_size = window_size
        self.forecast_days = forecast_days
        self.ensemble_size = ensemble_size
        self.models = []
        self.scalers = []
        self.feature_names = []

    def add_technical_indicators(self, df):
        # ... (Same logic as original, adapted for silence/st.write) ...
        features_df = df.copy()
        
        # Basic Returns
        features_df['returns'] = df['Close'].pct_change()
        features_df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving Averages
        sma_periods = [5, 10, 20, 50]
        for period in sma_periods:
            features_df[f'sma_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
            features_df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            # Ratios
            features_df[f'sma_ratio_{period}'] = (df['Close'] / features_df[f'sma_{period}'] - 1)
            features_df[f'ema_ratio_{period}'] = (df['Close'] / features_df[f'ema_{period}'] - 1)

        # Volatility
        volatility_periods = [5, 10, 20]
        for period in volatility_periods:
            features_df[f'volatility_{period}'] = features_df['returns'].rolling(window=period, min_periods=1).std()
        
        base_vol = features_df['volatility_20'].replace(0, np.nan).fillna(1e-8)
        for period in [5, 10]:
            features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / base_vol

        # Momentum
        for period in [5, 10, 20]:
            features_df[f'momentum_{period}'] = (df['Close'] / df['Close'].shift(period) - 1)
            features_df[f'roc_{period}'] = df['Close'].pct_change(periods=period)

        # Advanced TA (if available)
        if TA_AVAILABLE:
            try:
                rsi = ta.momentum.RSIIndicator(df['Close'], window=14)
                features_df['rsi'] = rsi.rsi()
                features_df['rsi_normalized'] = (features_df['rsi'] - 50) / 50
                
                bb = ta.volatility.BollingerBands(df['Close'])
                bb_w = (bb.bollinger_hband() - bb.bollinger_lband()).replace(0, 1e-8)
                features_df['bb_position'] = (df['Close'] - bb.bollinger_lband()) / bb_w
                features_df['bb_width'] = bb_w / df['Close']
            except Exception:
                pass

        # Cleanup
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return features_df

    def prepare_features(self, df):
        features_df = self.add_technical_indicators(df)
        # Exclude raw prices, keep ratios/indicators
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        exclude_cols += [c for c in features_df.columns if 'sma_' in c and 'ratio' not in c]
        exclude_cols += [c for c in features_df.columns if 'ema_' in c and 'ratio' not in c]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        if not feature_cols: feature_cols = ['returns', 'log_returns']
        
        self.feature_names = feature_cols
        return np.nan_to_num(features_df[feature_cols].values, nan=0.0)

    def create_model(self, input_shape, model_type='bidirectional'):
        # ... (Model architecture from original code) ...
        if model_type == 'bidirectional':
            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True, dropout=0.1), input_shape=input_shape),
                BatchNormalization(),
                Bidirectional(LSTM(64, return_sequences=True, dropout=0.1)),
                BatchNormalization(),
                Bidirectional(LSTM(32, dropout=0.1)),
                BatchNormalization(),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.1),
                Dense(self.forecast_days)
            ])
        # Simple fallback for speed if desired, but sticking to original architecture
        else:
             model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.1),
                BatchNormalization(),
                LSTM(64, dropout=0.1),
                BatchNormalization(),
                Dense(32, activation='relu'),
                Dense(self.forecast_days)
            ])
        return model

    def train(self, df, progress_callback=None):
        if not TF_AVAILABLE: return None
        
        try:
            # Data Prep
            feature_matrix = self.prepare_features(df)
            prices = df['Close'].values
            log_returns = np.log(prices[1:] / prices[:-1]).reshape(-1, 1)
            feature_matrix = feature_matrix[1:] # Align
            
            all_data = np.column_stack([log_returns, feature_matrix])
            
            # Split & Scale
            train_size = int(len(all_data) * 0.85)
            train_data = all_data[:train_size]
            test_data = all_data[train_size:]
            
            scaler = RobustScaler()
            scaled_train = scaler.fit_transform(train_data)
            scaled_test = scaler.transform(test_data)
            self.scalers.append(scaler)
            
            # Sequences
            def create_seq(data):
                X, y = [], []
                for i in range(self.window_size, len(data) - self.forecast_days + 1):
                    X.append(data[i - self.window_size:i])
                    y.append(data[i:i + self.forecast_days, 0]) # Predict 0th col (returns)
                return np.array(X), np.array(y)

            X_train, y_train = create_seq(scaled_train)
            X_test, y_test = create_seq(scaled_test)

            if len(X_train) == 0: return None
            
            # Train Ensemble
            model_types = ['bidirectional', 'standard', 'gru'][:self.ensemble_size]
            
            for i, m_type in enumerate(model_types):
                if progress_callback: progress_callback(f"Training model {i+1}/{len(model_types)}: {m_type}")
                
                model = self.create_model(X_train.shape[1:], m_type)
                model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
                
                # Reduced patience for Streamlit speed
                es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
                model.fit(X_train, y_train, validation_split=0.2, 
                          epochs=30, batch_size=32, callbacks=[es], verbose=0)
                self.models.append(model)
                
            # Eval
            preds = np.mean([m.predict(X_test, verbose=0) for m in self.models], axis=0)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            return {'MAE': mae, 'RMSE': rmse}

        except Exception as e:
            st.error(f"Training failed: {e}")
            return None

    def predict_future(self, df):
        if not self.models: return []
        
        feature_matrix = self.prepare_features(df)
        prices = df['Close'].values
        log_returns = np.log(prices[1:] / prices[:-1]).reshape(-1, 1)
        feature_matrix = feature_matrix[1:]
        
        all_data = np.column_stack([log_returns, feature_matrix])
        scaled_data = self.scalers[0].transform(all_data)
        
        last_sequence = scaled_data[-self.window_size:].reshape(1, self.window_size, -1)
        
        # Ensemble predict
        scaled_forecast = np.mean([m.predict(last_sequence, verbose=0) for m in self.models], axis=0)[0]
        
        # Inverse Transform
        dummy = np.zeros((len(scaled_forecast), scaled_data.shape[1]))
        dummy[:, 0] = scaled_forecast
        forecast_returns = self.scalers[0].inverse_transform(dummy)[:, 0]
        
        # Reconstruct Prices
        forecast_prices = [prices[-1]]
        for ret in forecast_returns:
            forecast_prices.append(forecast_prices[-1] * np.exp(ret))
            
        return np.array(forecast_prices[1:])

# --- Main App Interface ---

st.title("🧠 Enhanced LSTM Stock Predictor")
st.markdown("Deep Learning model using **Ensemble LSTMs/GRUs** and **Technical Analysis**.")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Stock", 
    ["IMP.JO", "ANG.JO", "GFI.JO", "VAL.JO", "APN.JO", "NPH.JO", 
     "SOL.JO", "SBK.JO", "NED.JO", "ABG.JO", "FSR.JO", "CPI.JO"])
window_size = st.sidebar.slider("Lookback Window (Days)", 30, 90, 60)
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 21)
ensemble_size = st.sidebar.slider("Ensemble Size (Models)", 1, 3, 2)

@st.cache_data(ttl=3600)
def get_data(t):
    stock = yf.Ticker(t)
    return stock.history(period="5y")

data = get_data(ticker)

if not data.empty:
    st.subheader(f"Historical Data: {ticker}")
    st.line_chart(data['Close'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Last Price: {data['Close'].iloc[-1]:.2f}")
    with col2:
        st.info(f"Data Points: {len(data)}")

    # TRAINING BUTTON
    if st.button("🚀 Train Model & Forecast"):
        if not TF_AVAILABLE:
            st.error("TensorFlow not found. Cannot train.")
            st.stop()
            
        st.divider()
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        predictor = EnhancedStockPredictor(window_size, forecast_days, ensemble_size)
        
        def update_status(msg):
            status_text.text(f"Running: {msg}")
        
        with st.spinner("Training Ensemble Model... (This uses CPU and may take 1-2 mins)"):
            metrics = predictor.train(data, progress_callback=update_status)
        
        if metrics:
            progress_bar.progress(100)
            status_text.text("Training Complete!")
            
            st.success(f"Model Trained. Validation MAE: {metrics['MAE']:.4f}")
            
            # Forecast
            future_prices = predictor.predict_future(data)
            
            # Dates
            last_date = data.index[-1]
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            # Plotting
            st.subheader("🔮 Forecast Results")
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # History (zoom in on last 6 months)
            subset = data.iloc[-150:]
            ax.plot(subset.index, subset['Close'], label='Historical', color='blue', alpha=0.6)
            
            # Forecast
            ax.plot(future_dates, future_prices, label='LSTM Ensemble Forecast', color='red', linestyle='--', linewidth=2)
            
            # Uncertainty bands (Rough estimation based on volatility)
            std_dev = np.std(future_prices) * 0.5
            ax.fill_between(future_dates, future_prices - std_dev, future_prices + std_dev, color='red', alpha=0.1)
            
            ax.set_title(f"{ticker} - {forecast_days} Day Forecast")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Analysis
            exp_return = (future_prices[-1] / data['Close'].iloc[-1] - 1) * 100
            st.metric("Expected Return (End of Period)", f"{exp_return:.2f}%", delta_color="normal")
            
        else:
            st.error("Training failed. Data might be insufficient or model failed to converge.")
else:
    st.error("Could not fetch data.")
