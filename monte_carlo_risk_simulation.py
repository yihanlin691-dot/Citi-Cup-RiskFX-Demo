import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_ingestion_and_cleaning import (
    load_historical_fx_data, 
    clean_and_preprocess_data, 
    calculate_risk_parameters
)

# Global Parameters
INITIAL_INVESTMENT = 100000 
FORECAST_DAYS = 30
NUM_SIMULATIONS = 10000
CONFIDENCE_LEVEL = 0.99
ANNUAL_TRADING_DAYS = 252 # Added constant for clarity

def run_monte_carlo_simulation(mu_annual: float, sigma_annual: float, initial_investment: float, 
                               forecast_days: int = FORECAST_DAYS, num_simulations: int = NUM_SIMULATIONS) -> np.ndarray:
    """
    Performs Monte Carlo simulation for future portfolio value based on Geometric Brownian Motion (GBM).
    Uses Annualized Mu and Sigma, converting them to daily scale for simulation.
    
    Args:
        mu_annual: Annualized drift calculated from historical data.
        sigma_annual: Annualized volatility calculated from historical data.
        initial_investment: The starting portfolio value.
        
    Returns:
        A NumPy array containing the simulated final portfolio values.
    """
    # 1. Convert Annualized Parameters to Daily Scale (De-annualization)
    T = ANNUAL_TRADING_DAYS 
    
    # Calculate the daily drift (mu_daily) and volatility (sigma_daily)
    # The drift term is adjusted using the GBM discrete formula for log returns:
    # mu_daily = (mu_annual / T) - (0.5 * sigma_annual^2 / T)
    mu_daily = (mu_annual / T) - (0.5 * (sigma_annual ** 2) / T) 
    sigma_daily = sigma_annual / np.sqrt(T)
    
    print(f"\n--- Running Monte Carlo Simulation ---")
    print(f"Daily Drift (mu_daily) used: {mu_daily:.6f}")
    print(f"Daily Volatility (sigma_daily) used: {sigma_daily:.6f}")
    
    # 2. Vectorized simulation setup
    
    # Generate random daily log returns for all simulations and all forecast days simultaneously
    # Shape: (num_simulations, forecast_days)
    daily_log_returns = np.random.normal(loc=mu_daily, 
                                         scale=sigma_daily, 
                                         size=(num_simulations, forecast_days))
    
    # Sum the log returns over the forecast period (axis=1)
    # This gives the final log return: ln(P_T / P_0)
    final_log_returns = daily_log_returns.sum(axis=1)
    
    # Convert final log returns back to price and calculate final value
    # P_T = P_0 * exp(ln(P_T / P_0))
    simulated_values = initial_investment * np.exp(final_log_returns)
        
    return simulated_values

def calculate_value_at_risk(simulated_values: np.ndarray, confidence_level: float = CONFIDENCE_LEVEL) -> tuple:
    """
    Calculates Value at Risk (VaR) and Conditional VaR (CVaR) 
    from the simulated final portfolio values.
    """
    # VaR is calculated at the (1 - Confidence Level) percentile (e.g., 1% quantile for 99% CL)
    var_quantile = 1 - confidence_level
    
    # 1. Calculate VaR (The portfolio value at the worst-case percentile)
    var_value = np.percentile(simulated_values, var_quantile * 100)
    
    # 2. Calculate Loss at VaR
    var_loss = INITIAL_INVESTMENT - var_value
    
    # 3. Calculate CVaR (Expected Shortfall - the average loss beyond VaR)
    # Average of all simulated values that are below or equal to the VaR value
    cvar_value = simulated_values[simulated_values <= var_value].mean()
    cvar_loss = INITIAL_INVESTMENT - cvar_value
    
    print(f"\n--- Risk Metrics Calculation (Confidence Level: {confidence_level*100:.0f}%) ---")
    print(f"Initial Investment: ${INITIAL_INVESTMENT:,.2f}")
    print(f"VaR ({confidence_level*100:.0f}%) Portfolio Value: ${var_value:,.2f}")
    print(f"Potential VaR Loss: ${var_loss:,.2f}")
    print(f"Conditional VaR (CVaR) Loss: ${cvar_loss:,.2f}")
    
    return var_loss, var_value, cvar_loss # Return var_value for plotting


if __name__ == '__main__':
    # 1. Data Ingestion and Parameter Calculation
    print("\n--- Starting Data and Parameter Extraction ---")
    
    # Load data from CSV and preprocess
    raw_data = load_historical_fx_data()
    processed_data = clean_and_preprocess_data(raw_data)
    
    # Calculate the Annualized parameters (mu and sigma) from the processed data
    mu_annual, sigma_annual = calculate_risk_parameters(processed_data, ANNUAL_TRADING_DAYS)
    
    # Check for calculation errors
    if mu_annual is None:
        print("Fatal Error: Could not calculate risk parameters. Simulation aborted.")
        exit()
    
    # 2. Run Simulation
    final_values = run_monte_carlo_simulation(
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        initial_investment=INITIAL_INVESTMENT,
        forecast_days=FORECAST_DAYS,
        num_simulations=NUM_SIMULATIONS
    )
    
    # 3. Calculate Risk
    var_loss, var_value, cvar_loss = calculate_value_at_risk(final_values, CONFIDENCE_LEVEL)

    # 4. Visualization and Saving the Result
    
    plt.figure(figsize=(12, 7))
    
    # Plot the histogram of the final portfolio values
    plt.hist(final_values, bins=100, density=True, alpha=0.6, color='skyblue', 
             label='Simulated Final Value Distribution')
    
    # Mark the VaR value line 
    plt.axvline(var_value, color='red', linestyle='dashed', linewidth=3, 
                label=f'VaR ({CONFIDENCE_LEVEL*100:.0f}% CL) Value: ${var_value:,.2f}\nPotential Loss: ${var_loss:,.2f}')
    
    # Mark the initial investment value
    plt.axvline(INITIAL_INVESTMENT, color='green', linestyle='solid', linewidth=2, 
                label=f'Initial Investment: ${INITIAL_INVESTMENT:,.2f}')
    
    plt.title(f'Monte Carlo Simulation: Portfolio Value Distribution after {FORECAST_DAYS} Days', 
              fontsize=16)
    plt.xlabel('Simulated Portfolio Final Value ($)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', alpha=0.5)
    
    # CRITICAL: Save the figure as result.png
    plt.savefig('result.png')
    
    print("\n--- Visualization Complete ---")
    print("Successfully generated and saved result.png file.")
