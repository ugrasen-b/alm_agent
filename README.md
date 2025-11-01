# Neural ALM Agent – Interest Rate Risk Management Prototype

### Overview
This project aims to develop a **neural network–based agent** that supports a bank’s **Asset–Liability Management (ALM)** by dynamically managing **interest rate risk** on its **savings account portfolio**.

Banks collect deposits in savings accounts and pay clients a variable interest rate (“client rate”). To earn income, these funds are invested in **government bonds** of different maturities. The challenge lies in allocating funds across maturities while managing **interest rate uncertainty** and **liability behavior**.

This project builds a **simulation and learning environment** where a neural agent is trained via **Monte Carlo scenarios** to learn the optimal investment strategy.

---

### Problem Setup

At each time step (monthly):
- The bank observes:
  - Total **savings account volume**
  - The **client rate** paid to customers
- The bank decides how to allocate this volume into available **government bonds**:
  - 1-Month, 3-Month, 6-Month, and 1-Year maturities
- Future yield curves, client rates, and volumes are **unknown** and simulated through stochastic processes (Monte Carlo paths).

The simulation runs over a **12-month horizon**, generating **10,000 yield curve and balance trajectories**.  
The agent learns a strategy that performs well across these simulated futures.

---

### Objectives
- Build a **Monte Carlo ALM simulator** that produces realistic yield curve, client rate, and volume paths.
- Train a **neural agent** (using PyTorch) that learns an optimal **bond allocation strategy** under uncertainty.
- Compare the learned strategy with **static benchmarks** (e.g., fixed 50%-50% allocation).
- Extend to longer horizons (e.g., 120 months) and richer instrument sets in later phases.

---

### Project Structure

--to--do

