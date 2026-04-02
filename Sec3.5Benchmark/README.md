# Section 3.5: Benchmark Comparison with Deep Learning and Traditional Methods

This folder contains the code and experimental settings for **Section 3.5** of the paper:
> Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks

## Content
- Training of Baseline Models
- Fitting accuracy comparison of various models (Table 5)

## Structure
torchrun --nproc_per_node=2 MyKANTrain.py

torchrun --nproc_per_node=2 Test_MSE.py
