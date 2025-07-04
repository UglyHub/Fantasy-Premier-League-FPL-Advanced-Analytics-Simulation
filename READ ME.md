# Fantasy Premier League (FPL) Advanced Analytics & Simulation

## Overview

This project provides a comprehensive suite of advanced analytics and simulation tools for Fantasy Premier League (FPL) player data. Leveraging state-of-the-art machine learning, natural language processing, reinforcement learning, and generative modeling, the notebook uncovers hidden player archetypes, predicts injury risk, simulates dynamic pricing, forecasts gameweek difficulty, and generates realistic FPL team selections.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Key Features](#key-features)
  - [1. Player Archetype Clustering with Multimodal Embeddings](#1-player-archetype-clustering-with-multimodal-embeddings)
  - [2. Dynamic Player Valuation with Reinforcement Learning](#2-dynamic-player-valuation-with-reinforcement-learning)
  - [3. Gameweek Difficulty Predictor with Graph Neural Networks](#3-gameweek-difficulty-predictor-with-graph-neural-networks)
  - [4. Injury Risk Prediction with Time-Series Analysis](#4-injury-risk-prediction-with-time-series-analysis)
  - [5. FPL Manager Behavior Simulation with Generative AI](#5-fpl-manager-behavior-simulation-with-generative-ai)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Results & Outputs](#results--outputs)
- [Acknowledgements](#acknowledgements)

---

## Project Structure

- `FPL_24_25.ipynb` — Main Jupyter notebook containing all code, experiments, and visualizations.
- `fpl_playerstats_2024-25.csv` — Cleaned dataset of FPL player statistics for the 2024/25 season.
- `FPL playerstats data dictionary.pdf` — Data dictionary describing all columns in the dataset.
- `*.json` — Altair-generated visualizations for team composition and cost breakdowns.

---

## Key Features

### 1. Player Archetype Clustering with Multimodal Embeddings

- **Objective:** Cluster players into novel archetypes using both numerical stats (e.g., goals, assists, influence) and semantic embeddings of player/team names.
- **Approach:**
  - Numerical features are standardized.
  - Player and team names are embedded using a pre-trained BERT model.
  - Features are concatenated and reduced in dimension using UMAP.
  - K-Means clustering groups players into archetypes, visualized in 2D.
- **Outcome:** Reveals hidden patterns in player roles, going beyond traditional position-based groupings.

### 2. Dynamic Player Valuation with Reinforcement Learning

- **Objective:** Simulate a dynamic pricing system for FPL players, optimizing for points per million.
- **Approach:**
  - Custom Gymnasium environment models pricing decisions as actions (increase, decrease, keep).
  - RL agent (PPO) learns to adjust player costs based on form, points, and team strength.
- **Outcome:** Demonstrates how dynamic pricing could optimize value, compared to FPL’s static system.

### 3. Gameweek Difficulty Predictor with Graph Neural Networks

- **Objective:** Predict individualized gameweek difficulty for each player, modeling team and player interactions as a graph.
- **Approach:**
  - Players are nodes; edges connect players from the same team.
  - Node features include form, expected goals, team strength, and points.
  - A Graph Convolutional Network (GCN) predicts difficulty scores.
- **Outcome:** Captures complex dependencies, aiding transfer and captaincy decisions.

### 4. Injury Risk Prediction with Time-Series Analysis

- **Objective:** Forecast player injury risk using time-series data (gameweek points, minutes, etc.).
- **Approach:**
  - LSTM model trained on sequences of recent gameweek data.
  - Binary classification predicts risk of injury/unavailability.
- **Outcome:** Provides probabilities to help managers avoid risky players.

### 5. FPL Manager Behavior Simulation with Generative AI

- **Objective:** Simulate realistic FPL team selections using a Variational Autoencoder (VAE).
- **Approach:**
  - Synthetic teams are generated under FPL constraints (budget, positions).
  - VAE learns latent patterns in team selection.
  - Teams are assigned starting 11 and substitutes for common formations.
  - Altair visualizations show player distribution and cost breakdowns.
- **Outcome:** Offers insights into community selection trends and optimal team structures.

---

## Installation & Requirements

**Python Version:** 3.8+

**Required Libraries:**
- pandas, numpy, scikit-learn, matplotlib, torch, transformers, umap-learn, gymnasium, stable-baselines3, tensorflow, altair, pytorch-geometric

**Installation:**
Install all dependencies using pip:
```sh
pip install pandas numpy scikit-learn matplotlib torch transformers umap-learn gymnasium stable-baselines3 tensorflow altair torch-geometric
```
Some code cells in the notebook also include `!pip install ...` commands for Colab compatibility.

---

## Usage

1. **Open `FPL_24_25.ipynb` in Jupyter or VS Code.**
2. **Run cells sequentially** to execute each module. Each section is self-contained and can be run independently if the required data is loaded.
3. **Visualizations** are saved as JSON files and can be viewed with Altair or Vega viewers.

---

## Results & Outputs

- **Cluster Visualizations:** 2D scatter plots of player archetypes.
- **Dynamic Pricing Table:** Shows original and RL-adjusted player costs.
- **Difficulty Scores:** Top players ranked by predicted gameweek difficulty.
- **Injury Risk Table:** Players ranked by predicted injury risk.
- **Synthetic Teams:** Starting 11 and substitutes for multiple formations, with cost breakdowns and position distributions.
- **Altair Charts:** JSON files for interactive visualizations of team composition and costs.

---

## Acknowledgements

- Player data sourced from the official FPL API and curated for the 2024/25 season.
- NLP models from HuggingFace Transformers.
- RL and GNN frameworks from Stable-Baselines3 and PyTorch Geometric.
- Visualization powered by Altair.

---

**Author:** Chiderah Onwumelu  
**Contact:** chiderahonwumelu@gmail.com  


---

*This project is for research and educational purposes. Not affiliated with the official Fantasy Premier League.*