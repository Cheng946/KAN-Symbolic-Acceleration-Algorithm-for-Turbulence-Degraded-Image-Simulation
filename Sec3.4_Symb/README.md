# Section 3.4: Analysis of Symbolization Results

This folder contains the code and experimental settings for **Section 3.4** of the paper:
> Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks
## Content
- Symbolic KAN
- Weight distribution of elementary functions in the symbolic network (Table 2, Figure 2)
- Most significant input variables and their gradient rankings (Table 3)
- The test set Lfit of symbolic expressions with different numbers of cl,i,j and wl,i,j b in each layer (Table 4)
- Symbolic network outputs closed-form expressions

## Usage
torchrun --nproc_per_node=2 SymbolicKAN_Finetune.py

python SymbolicKAN_LayerWeightAnalysis.py

torchrun --nproc_per_node=2 SymbolicKAN_grad.py

python SymbolicKAN_Equation_Prune.py

python SymbolicKAN_Equation.py
