import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from data_ingestion_and_cleaning import clean_and_preprocess_data, fetch_mock_stock_data

# Global Parameters
INITIAL_INVESTMENT = 100000 
FORECAST_DAYS = 30
NUM_SIMULATIONS = 10000
CONFIDENCE_LEVEL = 0.99

def run_monte_carlo_simulation(log_returns: pd.Series, initial_investment: float, 
                               forecast_days: int = FORECAST_DAYS, num_simulations: int = NUM_SIMULATIONS) -> np.ndarray:
    """
    Performs Monte Carlo simulation to forecast future portfolio value (based on GBM).
    The model assumes daily log returns follow a normal distribution.
    
    Args:
        log_returns: Historical daily log returns.
        initial_investment: The starting portfolio value.
        ... (omitted other args for brevity)
        
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
        # Generate random daily returns for the forecast period
        daily_returns = np.exp(np.random.normal(loc=mu, scale=sigma, size=forecast_days))
        
        # Calculate the final value of the investment path
        final_value = initial_investment * np.product(daily_returns)
        simulated_values[i] = final_value
        
    return simulated_values

def calculate_value_at_risk(simulated_values: np.ndarray, confidence_level: float = CONFIDENCE_LEVEL) -> tuple:
    """
    Calculates Value at Risk (VaR) and Conditional VaR (CVaR) 
    from the simulated final portfolio values.
    """
    # VaR is calculated at the (1 - Confidence Level) percentile
    var_quantile = 1 - confidence_level
    
    # 1. Calculate VaR (The portfolio value at the worst-case percentile)
    var_value = np.percentile(simulated_values, var_quantile * 100)
    
    # 2. Calculate Loss at VaR
    var_loss = INITIAL_INVESTMENT - var_value
    
    # 3. Calculate CVaR (Expected Shortfall - the average loss beyond VaR)
    cvar_loss = INITIAL_INVESTMENT - simulated_values[simulated_values <= var_value].mean()
    
    print(f"\n--- Risk Metrics Calculation (Confidence Level: {confidence_level*100:.0f}%) ---")
    print(f"Initial Investment: ${INITIAL_INVESTMENT:,.2f}")
    print(f"VaR (99%) Portfolio Value: ${var_value:,.2f}")
    print(f"Potential VaR Loss: ${var_loss:,.2f}")
    print(f"Conditional VaR (CVaR) Loss: ${cvar_loss:,.2f}")
    
    return var_loss, var_value, cvar_loss # Return var_value for plotting

if __name__ == '__main__':
    # 1. Prepare Data 
    TICKER = 'MOCK_FX'
    START_DATE = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    raw_data = fetch_mock_stock_data(TICKER, START_DATE, END_DATE)
    processed_data = clean_and_preprocess_data(raw_data)
    log_returns = processed_data['Log_Returns']
    
    # 2. Run Simulation
    final_values = run_monte_carlo_simulation(log_returns, INITIAL_INVESTMENT, FORECAST_DAYS, NUM_SIMULATIONS)
    
    # 3. Calculate Risk
    var_loss, var_value, cvar_loss = calculate_value_at_risk(final_values, CONFIDENCE_LEVEL)

    # 4. Visualization and Saving the Result
    
    plt.figure(figsize=(12, 7))
    
    # Plot the histogram of the final portfolio values
    plt.hist(final_values, bins=100, density=True, alpha=0.6, color='skyblue', 
             label='Simulated Final Value Distribution')
    
    # Mark the VaR value line (the maximum loss at 99% CL)
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
