# Citi-Cup-RiskFX-Demo

## Project Overview

This repository contains a streamlined and runnable demonstration of key technical components developed for a quantitative finance project. The primary goal is to showcase expertise in the integration of **data engineering, core financial algorithms, and advanced agentic decision-making workflows** for real-time portfolio risk analysis.

The architecture is deliberately modular, focusing on three critical phases of a modern FinTech system: Data Ingestion, Algorithmic Modeling, and AI-Powered Actioning.

## Core Technical Highlights

This demonstration is composed of three independent, runnable modules that represent my **individual contribution** to the project's technical architecture.

### 1. Data Ingestion and Cleaning (`data_ingestion_and_cleaning.py`)

* **Function:** Simulates a robust data pipeline for fetching raw financial time series, such as FX or stock prices.
* **Key Capability:** Implements essential **data cleaning**, such as handling missing values using interpolation and **feature engineering** by calculating daily **logarithmic returns**. Log returns are the necessary input for advanced risk models.
* **Technologies:** Python, Pandas, NumPy.

### 2. Algorithmic Risk Modeling (`monte_carlo_risk_simulation.py`)

* **Function:** Executes the core quantitative finance algorithm.
* **Key Capability:** Performs **Monte Carlo Simulation** based on historical log returns to forecast future portfolio value paths. This module calculates industry-standard risk metrics: **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** at a 99% confidence level.
* **Technologies:** Python, NumPy (for high-performance numerical computation), Statistical Methods.

### 3. Agentic Decision-Making Simulation (`agentic_decision_simulation.py`)

* **Function:** Simulates an advanced **LLM-Agent workflow** (similar to LangGraph) that processes data and makes complex decisions.
* **Key Capability:** Demonstrates an agent that operates in a multi-step loop:
    1.  **Tool Calling:** Automatically triggers the Risk Analysis Tool (Module 2).
    2.  **State Management:** Updates its internal state based on tool results.
    3.  **Synthesizing Advice:** Uses external lookups (simulated) to generate final and actionable strategic advice, such as rebalancing recommendations.
* **Technologies:** Python Classes, Modular Function Design (Agentic Pattern).

## How to Run the Demo

To execute the core components, you only need Python and the necessary dependencies.

**1. Dependencies**

```bash
pip install numpy pandas
