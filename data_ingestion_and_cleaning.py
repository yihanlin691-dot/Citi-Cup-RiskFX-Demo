import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simulate a real API call
def fetch_mock_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Simulates fetching financial time series data (e.g., historical stock prices).
    Returns a DataFrame with 'Date' and 'Close_Price'.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    num_days = len(dates)
    
    # Generate simulated price data
    initial_price = 100
    returns = np.random.normal(0.0005, 0.01, num_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Close_Price': prices
    })
    print(f"--- Data Fetched for {ticker}: {num_days} days ---")
    return df

def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs standard data cleaning and feature engineering:
    1. Ensures all columns are numeric/datetime.
    2. Calculates daily logarithmic returns (crucial for risk modeling).
    3. Handles potential missing values (using ffill/bfill).
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Data Cleaning: Handle potential missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Feature Engineering: Calculate Log Returns
    # R_t = ln(P_t / P_{t-1})
    df['Log_Returns'] = np.log(df['Close_Price'] / df['Close_Price'].shift(1))
    
    # Drop the first row which contains NaN log return
    df.dropna(inplace=True) 
    
    print("--- Data Preprocessing Complete: Log Returns Calculated ---")
    return df

if __name__ == '__main__':
    TICKER = 'MOCK_FX'
    START_DATE = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # 1. Ingestion
    raw_data = fetch_mock_stock_data(TICKER, START_DATE, END_DATE)
    
    # 2. Cleaning and Preprocessing
    processed_data = clean_and_preprocess_data(raw_data)
    
    # Display results
    print("\nProcessed Data Head:")
    print(processed_data.head())
    print("\nStatistical Summary of Log Returns:")
    print(processed_data['Log_Returns'].describe())


