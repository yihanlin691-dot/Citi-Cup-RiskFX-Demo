# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 22:14:08 2025

@author: admin
"""

import numpy as np  
from typing import List, Any, Dict 

# 1. Define Tools (Functions the Agent can call) 

def risk_analysis_tool(ticker: str, forecast_days: int) -> str:
    """Simulates calling the Monte Carlo model to get risk data."""
    print(f"-> TOOL CALL: Executing Monte Carlo Risk Analysis for {ticker} over {forecast_days} days...")
    # Mock Risk Data (using numpy for random generation)
    mock_var_loss = np.random.randint(1000, 5000)
    # The output format is crucial for the parser in the decision loop
    return f"RESULT: 99% VaR Loss for {ticker} is ${mock_var_loss:,.0f}. Risk is considered MODERATE."

def strategy_lookup_tool(risk_level: str) -> str:
    """Simulates querying a database or LLM for suitable strategies."""
    print(f"-> TOOL CALL: Consulting Strategy Database for risk level '{risk_level}'...")
    if 'LOW' in risk_level:
        return "RESULT: Strategy suggests: Diversify into safe government bonds and reduce equity exposure."
    elif 'MODERATE' in risk_level:
        return "RESULT: Strategy suggests: Maintain current diversified portfolio; rebalance if any sector exceeds 15%."
    else: # HIGH or other
        return "RESULT: Strategy suggests: Immediately reduce high-beta assets and raise cash reserves to 20%."

# 2. Define Agent State and Nodes (Simplified LangGraph Concept) 

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
        # Note: We need to parse the risk level from risk_data before calling strategy tool
        print("Decision: Risk data available. Proceeding to call Strategy Lookup Tool.")
        return "call_strategy_tool"
    else:
        print("Decision: All required data available. Proceeding to Final Synthesis.")
        return "final_synthesis"

# 3. Define Execution Nodes (Tool Callers) 

def execute_risk_tool(state: AgentState) -> AgentState:
    """Executes the risk_analysis_tool and updates the state."""
    result = risk_analysis_tool(state.ticker, forecast_days=30)
    state.risk_data = result
    return state

def execute_strategy_tool(state: AgentState) -> AgentState:
    """Executes the strategy_lookup_tool and updates the state."""
    
    if state.risk_data and 'Risk is considered ' in state.risk_data:
        try:
            # Robust Parsing Logic to prevent 'list index out of range'
            # 1. Get the segment after the main separator
            risk_segment = state.risk_data.split('Risk is considered ')[1]
            # 2. Get the risk level (e.g., MODERATE) before the first period
            risk_level = risk_segment.split('.')[0].strip()
            
            # 3. Validation and execution
            if risk_level in ['MODERATE', 'HIGH', 'LOW']:
                result = strategy_lookup_tool(risk_level)
                state.strategy_advice = result
            else:
                print(f"WARNING: Extracted risk level '{risk_level}' is invalid. Defaulting.")
                result = strategy_lookup_tool('MODERATE')
                state.strategy_advice = result
                
        except IndexError:
            # Catches the 'list index out of range' error if the format changes
            print("ERROR: Parsing failed due to unexpected tool output format.")
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

# 5. Simplified Agent Loop (Simulating LangGraph execution flow)

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

