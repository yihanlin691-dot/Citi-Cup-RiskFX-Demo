# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 22:12:33 2025

@author: admin
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# Assuming the data processing from the previous script is available
from data_ingestion_and_cleaning import clean_and_preprocess_data, fetch_mock_stock_data

def run_monte_carlo_simulation(log_returns: pd.Series, initial_investment: float, 
                               forecast_days: int = 30, num_simulations: int = 10000) -> np.ndarray:
    """
    Performs Monte Carlo simulation to forecast future portfolio value.
    The model assumes daily log returns follow a normal distribution.
    
    Args:
        log_returns: Historical daily log returns.
        initial_investment: The starting portfolio value.
        forecast_days: The number of days to forecast into the future.
        num_simulations: The number of simulation paths to run.
        
    Returns:
        A NumPy array containing the simulated final portfolio values.
    """
    # 1. Calculate key statistics from historical data
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    print(f"\n--- Running Monte Carlo Simulation ---")
    print(f"Historical Mean Log Return (mu): {mu:.4f}")
    print(f"Historical Standard Deviation (sigma): {sigma:.4f}")
    
    # 2. Setup simulation environment
    simulated_values = np.zeros(num_simulations)
    
    # 3. Core simulation loop
    for i in range(num_simulations):
        # Generate random daily returns for the forecast period (Geometric Brownian Motion)
        daily_returns = np.exp(np.random.normal(loc=mu, scale=sigma, size=forecast_days))
        
        # Calculate the final value of the investment path
        final_value = initial_investment * np.product(daily_returns)
        simulated_values[i] = final_value
        
    return simulated_values

def calculate_value_at_risk(simulated_values: np.ndarray, confidence_level: float = 0.99) -> tuple:
    """
    Calculates Value at Risk (VaR) and Conditional VaR (CVaR) 
    from the simulated final portfolio values.
    """
    # VaR is the loss corresponding to the confidence level (e.g., 1% quantile for 99% CL)
    # We look for the 1-Confidence Level percentile of the simulated values
    var_quantile = 1 - confidence_level
    
    # 1. Calculate VaR (the value at the specific percentile)
    var_value = np.percentile(simulated_values, var_quantile * 100)
    
    # 2. Calculate Loss at VaR (the actual potential loss)
    var_loss = INITIAL_INVESTMENT - var_value
    
    # 3. Calculate CVaR (Expected Shortfall - the average loss beyond VaR)
    cvar_value = simulated_values[simulated_values <= var_value].mean()
    cvar_loss = INITIAL_INVESTMENT - cvar_value
    
    print(f"\n--- Risk Metrics Calculation (Confidence Level: {confidence_level*100:.0f}%) ---")
    print(f"Initial Investment: ${INITIAL_INVESTMENT:,.2f}")
    print(f"Value at Risk (VaR) - Max Value at {var_quantile*100:.1f}th Percentile: ${var_value:,.2f}")
    print(f"Potential Loss (VaR Loss): ${var_loss:,.2f}")
    print(f"Conditional VaR (CVaR) - Average Loss beyond VaR: ${cvar_loss:,.2f}")
    
    return var_loss, cvar_loss

if __name__ == '__main__':
    # 1. Prepare Data (using functions from the first script)
    TICKER = 'MOCK_FX'
    START_DATE = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    raw_data = fetch_mock_stock_data(TICKER, START_DATE, END_DATE)
    processed_data = clean_and_preprocess_data(raw_data)
    log_returns = processed_data['Log_Returns']
    
    # 2. Set Parameters
    INITIAL_INVESTMENT = 100000 
    FORECAST_DAYS = 30
    NUM_SIMULATIONS = 10000
    CONFIDENCE_LEVEL = 0.99
    
    # 3. Run Simulation
    final_values = run_monte_carlo_simulation(log_returns, INITIAL_INVESTMENT, FORECAST_DAYS, NUM_SIMULATIONS)
    
    # 4. Calculate Risk
    var_loss, cvar_loss = calculate_value_at_risk(final_values, CONFIDENCE_LEVEL)

