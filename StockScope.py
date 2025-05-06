import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Set page config for Streamlit
st.set_page_config(page_title="Stock Price Tracker & Predictor", layout="wide")

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Feature engineering functions
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def create_features(data, window_size=5):
    df = data.copy()
    
    # Lag features
    for i in range(1, window_size + 1):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
    
    # Rolling statistics
    df['Rolling_Mean_5'] = df['Close'].rolling(window=5).mean()
    df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
    df['Rolling_Std_5'] = df['Close'].rolling(window=5).std()
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Technical indicators
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    
    # Target variable (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    return df.dropna()

def prepare_data(df, test_size=0.2):
    features = [col for col in df.columns if col not in ['Target', 'Close']]
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return predictions, mse, mae, feature_importance

# Streamlit app
def main():
    st.title("Stock Price Tracker & Predictor")
    
    # Sidebar for user inputs
    st.sidebar.header("Stock Selection")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", "AAPL")
    
    # Date selection with default as 1 year back
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date:", start_date)
    end_date = st.sidebar.date_input("End Date:", end_date)
    
    # Check if start date is before end date
    if start_date >= end_date:
        st.sidebar.error("Error: End date must be after start date.")
        return
    
    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Downloading stock data..."):
            try:
                stock_data = download_stock_data(ticker, start_date, end_date)
                
                if stock_data.empty:
                    st.error("No data found for this ticker symbol. Please try another one.")
                    return
                
                # Display raw data
                st.subheader(f"Raw Stock Data for {ticker}")
                st.dataframe(stock_data.tail())
                
                # Plot closing price
                st.subheader(f"Closing Price for {ticker}")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(stock_data['Close'], label='Close Price')
                ax.set_title(f"{ticker} Closing Price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.grid(True)
                st.pyplot(fig)
                
                # Feature engineering
                featured_data = create_features(stock_data)
                
                # Prepare data for ML
                X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(featured_data)
                
                # Train model
                with st.spinner("Training prediction model..."):
                    rf_model = train_random_forest(X_train, y_train)
                
                # Evaluate model
                predictions, mse, mae, feature_importance = evaluate_model(rf_model, X_test, y_test, feature_names)
                
                # Display model performance
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                col1.metric("Mean Squared Error", f"{mse:.2f}")
                col2.metric("Mean Absolute Error", f"{mae:.2f}")
                
                # Plot predictions vs actual
                st.subheader("Actual vs Predicted Prices")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(y_test.values, label='Actual')
                ax2.plot(predictions, label='Predicted')
                ax2.set_title("Actual vs Predicted Prices")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Price ($)")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)
                
                # Feature importance
                st.subheader("Feature Importance")
                st.dataframe(feature_importance)
                
                # Latest prediction
                latest_data = featured_data.iloc[-1][feature_names].values.reshape(1, -1)
                latest_data_scaled = scaler.transform(latest_data)
                next_day_pred = rf_model.predict(latest_data_scaled)[0]
                
                st.subheader("Next Day Price Prediction")
                st.metric("Predicted Close Price", f"${next_day_pred:.2f}")
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
    
    # Add some info
    st.sidebar.info("""
    **Note:** 
    - This app uses Yahoo Finance API to fetch stock data.
    - The prediction model uses Random Forest with technical indicators.
    - Predictions are for educational purposes only.
    """)

if __name__ == "__main__":
    main()