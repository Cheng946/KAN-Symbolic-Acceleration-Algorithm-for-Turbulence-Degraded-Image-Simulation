# Section 3.3: K-Fold Comparison and Pruning Results

This folder contains the code and experimental settings for **Section 3.3** of the paper:
> Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks

## Content
- K-Fold Cross-Validation Training
- Pruning Analysis

## Usage
torchrun --nproc_per_node=2 MyKANTrain.py

torchrun --nproc_per_node=2 PruneKAN.py
