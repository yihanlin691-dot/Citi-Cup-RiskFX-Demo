# Citi-Cup-RiskFX-Demo

## 📖 1. Project Overview
This repository represents my personal contribution to a comprehensive risk management system developed for the **Citi Cup**. 

The project focuses on **quantifying Foreign Exchange  risk exposure** using stochastic calculus and simulation techniques. By integrating a **Geometric Brownian Motion** model with an agent-based decision engine, the system enables proactive risk assessment and dynamic position adjustment.

**Key Problem Solved:**
How to accurately forecast the **Value at Risk (VaR)** and **Conditional VaR (CVaR)** for a Spot FX portfolio under volatile market conditions to minimize downside risk.

---

## 📂 2. Repository Structure
This simplified version isolates the core quantitative modules I designed and implemented:

```text
├── data_ingestion_and_cleaning.py   # [Data Layer] Preprocessing & Return Calculation
├── monte_carlo_risk_simulation.py   # [Model Layer] Stochastic Simulation & VaR Engine
├── agentic_decision_simulation.py   # [Decision Layer] Automated Risk Mitigation Strategy
├── result.png                       # [Output] Visualization of Risk Distribution
└── README.md                        # Project Documentation
```

---

## ⚙️ 3. Core Modules & Methodology
### A. Data Processing (data_ingestion_and_cleaning.py)
Responsible for transforming raw time-series data into a format suitable for stochastic modeling.
 * **Method**: Fetches historical FX data and calculates Daily Log Returns ($$r_t = \ln(P_t / P_{t-1})$$).
 * **Key Output**: Historical volatility and drift parameters.
### B. Monte Carlo Risk Engine (monte_carlo_risk_simulation.py)
This is the core computational engine. It projects 10,000 potential future price paths using the **Geometric Brownian Motion (GBM)** model: $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
 * **Simulation**: Runs 10,000 iterations over a 30-day forecast horizon.
 * **Risk Metrics**: Calculates VaR (99%) to identify the threshold of worst-case losses.
 * **Visualization**: Generates a probability distribution of the final portfolio value (result.png).
### C. Agentic Decision System (agentic_decision_simulation.py)
Simulates a risk-aware trading agent that utilizes the calculated VaR metrics.
 * **Logic**: If the projected potential loss exceeds the risk appetite threshold, the agent automatically triggers hedging or position-sizing adjustments.

---

## 📊 4. Simulation Results
The following visualization demonstrates the probabilistic distribution of the portfolio's future value. The red dashed line highlights the Value at Risk (VaR) threshold at a 99% confidence level.
<img width="1200" height="700" alt="result" src="https://github.com/user-attachments/assets/36c86fae-8666-4a7c-967e-ea3415966c64" />

### Key Findings
 * **Initial Investment**: $100,000
 * **Forecast Horizon**: 30 Days
 * **Confidence Level**: 99%
 * **Insight**: The simulation suggests that while the portfolio has an expected positive drift, there is a quantifiable probability (1%) of significant downside, which is captured by the VaR metric shown in the chart.

---

## 🛠️ 5. Technologies & Dependencies

### 5.1. Core Dependencies

The project is developed using **Python 3.12** and relies on the following major scientific computing and data analysis libraries:

* **numpy**: High-performance numerical computing and array operations.
* **pandas**: Data structuring, manipulation, and time-series handling.
* **matplotlib**: Visualization for generating the `result.png` risk distribution chart.

### 5.2. Installation Guide

It is highly recommended to set up a virtual environment (e.g., using `venv` or `conda`) before installation.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yihanlin691-dot/Citi-Cup-RiskFX-Demo.git](https://github.com/yihanlin691-dot/Citi-Cup-RiskFX-Demo.git)
    cd Citi-Cup-RiskFX-Demo
    ```

2.  **Install Required Libraries:**
    Install all dependencies mentioned above: 
    ```bash
    pip install numpy pandas matplotlib
    ```

*Note: For this project, a GPU (CUDA) installation is NOT required.*

---

## 🧠 6. Future Work & Feedback
This project serves as a strong foundation and can be extended in several ways:
* **Risk Modeling**: Incorporate alternative risk models (e.g., Extreme Value Theory) or optimize Monte Carlo sampling strategies to improve estimation accuracy.
* **Agent Optimization**: Apply advanced machine learning (e.g., Reinforcement Learning) to the agent to optimize decision-making beyond simple threshold rules.
* **External Data**: Integrate real-time market data APIs for live risk monitoring.
Any constructive feedback is highly welcome! You can reach the author below.

## 📧 6. Contact
 * **Author**: Yihan Lin
 * **Context**: Citi Cup (Personal Module Demo)
<!-- end list -->


