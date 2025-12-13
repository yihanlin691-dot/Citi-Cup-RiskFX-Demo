import numpy as np  
from typing import List, Any, Dict 

# run the Monte Carlo model
try:
    from monte_carlo_risk_simulation import run_simulation_and_get_results
except ImportError:
    # Fallback for local testing if the file structure is not right
    print("WARNING: Could not import run_simulation_and_get_results. Using mock data.")
    def run_simulation_and_get_results():
        return np.random.randint(1000, 5000), 100000 - np.random.randint(1000, 5000), np.random.randint(5000, 8000)

# 1. Define Tools 

def determine_risk_level(var_loss: float, initial_investment: float = 100000) -> str:
    """Helper function to classify risk based on VaR percentage."""
    var_percent = var_loss / initial_investment
    if var_percent > 0.04:  # Loss over 4% is High Risk (e.g., VaR Loss > $4,000)
        return "HIGH"
    elif var_percent > 0.01: # Loss between 1% and 4% is Moderate Risk
        return "MODERATE"
    else: # Loss less than 1% is Low Risk
        return "LOW"

def risk_analysis_tool(ticker: str, forecast_days: int) -> str:
    """
    Executes the actual Monte Carlo model and returns the risk data.
    """
    print(f"-> TOOL CALL: Executing REAL Monte Carlo Risk Analysis for {ticker} over {forecast_days} days...")
    
    # CORE INTEGRATION POINT
    var_loss, var_value, cvar_loss = run_simulation_and_get_results()
    
    risk_level = determine_risk_level(var_loss)
    
    return (
        f"RESULT: 99% VaR Loss for {ticker} is ${var_loss:,.0f} (Value: ${var_value:,.0f}). "
        f"CVaR Loss is ${cvar_loss:,.0f}. Risk is considered {risk_level}."
    )

def strategy_lookup_tool(risk_level: str) -> str:
    """Simulates querying a database or LLM for suitable strategies."""
    print(f"-> TOOL CALL: Consulting Strategy Database for risk level '{risk_level}'...")
    if 'LOW' in risk_level:
        return "RESULT: Strategy suggests: Diversify into safe government bonds and reduce equity exposure."
    elif 'MODERATE' in risk_level:
        return "RESULT: Strategy suggests: Maintain current diversified portfolio; rebalance if any sector exceeds 15%."
    else: # HIGH or other
        return "RESULT: Strategy suggests: Immediately reduce high-beta assets and raise cash reserves to 20%."

# 2. Define Agent State and Nodes

class AgentState:
    """Represents the mutable state passed between agent steps."""
    def __init__(self, ticker: str, initial_prompt: str):
        self.ticker = ticker
        self.prompt = initial_prompt
        self.risk_data = None
        self.strategy_advice = None
        self.final_decision = None

def decision_node(state: AgentState) -> str:
    """The central 'Router' node that decides the next step based on current state."""
    print(f"\n--- NODE: Decision Node (Input: '{state.prompt}') ---")
    if not state.risk_data:
        print("Decision: Risk data is missing. Calling Risk Analysis Tool.")
        return "call_risk_tool"
    elif not state.strategy_advice:
        print("Decision: Risk data available. Proceeding to call Strategy Lookup Tool.")
        return "call_strategy_tool"
    else:
        print("Decision: All required data available. Proceeding to Final Synthesis.")
        return "final_synthesis"

# 3. Define Execution Nodes

def execute_risk_tool(state: AgentState) -> AgentState:
    """Executes the risk_analysis_tool and updates the state."""
    # Ensure forecast_days parameter is handled if needed, here we use default (30)
    result = risk_analysis_tool(state.ticker, forecast_days=30) 
    state.risk_data = result
    return state

def execute_strategy_tool(state: AgentState) -> AgentState:
    """Executes the strategy_lookup_tool and updates the state."""
    
    if state.risk_data and 'Risk is considered ' in state.risk_data:
        try:
            risk_segment = state.risk_data.split('Risk is considered ')[1]
            risk_level = risk_segment.split('.')[0].strip()
            
            if risk_level in ['MODERATE', 'HIGH', 'LOW']:
                result = strategy_lookup_tool(risk_level)
                state.strategy_advice = result
            else:
                print(f"WARNING: Extracted risk level '{risk_level}' is invalid. Defaulting to MODERATE.")
                result = strategy_lookup_tool('MODERATE')
                state.strategy_advice = result
                
        except (IndexError, AttributeError):
            print("ERROR: Parsing failed due to unexpected tool output format or missing data.")
            state.strategy_advice = "ERROR: Failed to parse risk level."
            
    else:
        print("WARNING: state.risk_data is missing or in unexpected format.")
        
    return state

# 4. Final Synthesis Node

def final_synthesis(state: AgentState) -> AgentState:
    """Generates the final human-readable decision."""
    print("\n--- NODE: Final Synthesis ---")
    decision_text = (
        f"Based on the initial request for '{state.prompt}', the system performed a multi-step analysis.\n"
        f"1. Risk Assessment: {state.risk_data}\n"
        f"2. Strategic Guidance: {state.strategy_advice}\n"
        f"FINAL DECISION: The agent recommends proceeding with the suggested strategy to maintain diversification and monitor sector weighting."
    )
    state.final_decision = decision_text
    return state

# 5. Simplified Agent Loop

def run_agent_workflow(initial_state: AgentState):
    current_state = initial_state
    
    for step in range(5): # Limit steps to prevent infinite loop
        next_step = decision_node(current_state)
        
        if next_step == "call_risk_tool":
            current_state = execute_risk_tool(current_state)
        elif next_step == "call_strategy_tool":
            current_state = execute_strategy_tool(current_state)
        elif next_step == "final_synthesis":
            current_state = final_synthesis(current_state)
            print("\n--- WORKFLOW COMPLETE ---")
            print(current_state.final_decision)
            break
        else:
            print("ERROR: Unknown decision state or workflow breakdown.")
            break
            
if __name__ == '__main__':
    initial_request = "Assess the current risk profile of the FX portfolio and propose a rebalancing strategy."
    initial_state = AgentState(ticker="EUR/USD", initial_prompt=initial_request)
    
    run_agent_workflow(initial_state)
