# Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks

## Paper Overview
Existing physics-based simulations of atmospheric turbulence-degraded images often suffer from high computational cost, which limits their use in large-scale synthetic dataset generation. To address this problem, this paper proposes an accelerated simulation framework integrating Principal Component Analysis (PCA) with Kolmogorov-Arnold Networks (KANs). PCA decomposes spatially varying optical transfer functions (OTFs) into a set of compact fixed basis functions, converting anisoplanatic image degradation into weighted superposition of space-invariant convolutions. Instead of a single compact closed-form formula, KAN constructs a decomposable symbolic computation graph to learn the nonlinear mapping from Zernike coefficient vectors to OTF basis coefficients, forming a physically auditable surrogate simulator. Layer-wise representation similarity analysis, neuron-wise physical attribution and parameter-level analysis jointly reveal the layer-by-layer evolution rule that shallow layers mainly encode phase and pupil information while deep layers gradually form OTF-oriented representations, supporting interpretable physical diagnosis of the surrogate model. Experimental results demonstrate that the proposed KAN achieves higher fitting accuracy than several benchmark neural networks under comparable parameter budgets. Compared with traditional phase screen segmentation and Zernike polynomial-based algorithms, the PCA–KAN framework achieves acceleration ratios of 42.61 and 5.89 respectively. All results verify that the PCA–KAN framework provides an efficient and physically auditable surrogate for the simulation of atmospheric turbulence-degraded images.

**Model Architecture**：
- Backbone Network： Kolmogorov-Arnold Network
- Core module：KAN
- Input/Output Dimensions：33，70

## 环境配置,Environment Setup
### 软件要求, Software Requirements
torch==1.13.1+cu116
numpy==1.26.4

### 硬件要求
- NVIDIA GPU A40 ×2 
- CUDA ≥11.6

### 依赖安装
```bash
conda create -n [env_name] python=3.10.10
conda activate [env_name]
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu116/torch_stable.html
