import pandas as pd
import numpy as np

# 1. Data Loading Function: Reads Local CSV File (FX_Historical_Data.csv)
def load_historical_fx_data(file_path: str = "FX_Historical_Data.csv") -> pd.DataFrame:
    """
    Loads historical FX price data from a local CSV file.
    
    Args:
        file_path (str): The path to the historical data CSV file. Defaults to 'FX_Historical_Data.csv'.
        
    Returns:
        pd.DataFrame: A DataFrame containing 'Date' and 'Close' prices.
    """
    print(f"--- 1. Data Ingestion: Attempting to read file '{file_path}' ---")
    try:
        # Load data from the local CSV file
        df = pd.read_csv(file_path)
        
        # Validation checks
        if 'Date' not in df.columns or 'Close' not in df.columns:
             raise KeyError("CSV must contain 'Date' and 'Close' columns.")

        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"Successfully loaded file: {file_path}")
        return df
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please ensure it is uploaded to the root directory.")
        return pd.DataFrame() 
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return pd.DataFrame()


# 2. Data Cleaning and Preprocessing Function 
def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs standard data cleaning and feature engineering:
    1. Sets the index and ensures correct data types.
    2. Calculates daily logarithmic returns (R_t = ln(P_t / P_{t-1})).
    3. Handles potential missing values.
    """
    if df.empty:
        return pd.DataFrame()
        
    df = df.set_index('Date')
    
    # Ensure price column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Data Cleaning (if any)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Calculate Log Returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Drop the first row which contains NaN log return
    df.dropna(inplace=True) 
    
    print("--- 2. Data Preprocessing Complete: Log Returns Calculated ---")
    return df

# 3. Parameter Calculation Function 
def calculate_risk_parameters(processed_data: pd.DataFrame, annual_trading_days: int = 252) -> tuple:
    """
    Calculates the annualized GBM parameters (mu, sigma) required for Monte Carlo simulation.
    
    Args:
        processed_data (pd.DataFrame): DataFrame containing 'Log_Returns'.
        annual_trading_days (int): Number of days used for annualization (default 252).
        
    Returns:
        tuple: (mu, sigma) - Annualized drift and volatility.
    """
    if processed_data.empty:
        return None, None
        
    log_returns = processed_data['Log_Returns']
    T = annual_trading_days 
    
    # Drift (mu) 
    mu = log_returns.mean() * T
    
    # Volatility (sigma) 
    sigma = log_returns.std() * np.sqrt(T)

    print("-" * 30)
    print("--- 3. Risk Parameter Calculation Complete ---")
    print(f" - Annualized Drift (mu): {mu:.6f}")
    print(f" - Annualized Volatility (sigma): {sigma:.6f}")
    print("-" * 30)

    return mu, sigma

if __name__ == '__main__':
    # 1. Load data from the local CSV file
    raw_data = load_historical_fx_data()
    
    # 2. Clean data and calculate log returns
    processed_data = clean_and_preprocess_data(raw_data)
    
    # 3. Calculate GBM parameters
    mu, sigma = calculate_risk_parameters(processed_data)
    
    if not processed_data.empty:
        print("\nProcessed Data Head (with Log Returns):")
        print(processed_data.head())
        print("\nStatistical Summary of Log Returns:")
        print(processed_data['Log_Returns'].describe())

    if mu is not None:
        print("\n--- Success! Data Cleaning and Parameter Extraction completed.---")

