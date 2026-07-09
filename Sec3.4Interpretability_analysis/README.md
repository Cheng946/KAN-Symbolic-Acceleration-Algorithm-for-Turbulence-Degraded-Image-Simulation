# Section 3.4: Interpretability analysis Results
This folder contains the code and experimental settings for Section 3.4 of the paper: **Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks**.

## Content
- **3.4.1 Layer-wise representational similarity analysis**: `Compute_RSA.py`
- **3.4.2 Neuron-wise physical attribution**: `Compute_RSA_neuron.py`
- **3.4.3 Intra-class parameter diversity of physically attributed nodes**: `Compute_RSA_analyze_symbolic_kan_group_weights.py`
- **3.4.4 Inter-class parameter discrepancy analysis**: `Compute_RSA_analyze_symbolic_kan_group_weights.py`
- **3.4.5 RSA-based representation distance analysis**: `Compute_RSA_group_by_class.py`

## Usage
Run the following commands to execute the corresponding interpretability analysis experiments:
```bash
python Compute_RSA.py
python Compute_RSA_neuron.py
python Compute_RSA_analyze_symbolic_kan_group_weights.py
python Compute_RSA_group_by_class.py
