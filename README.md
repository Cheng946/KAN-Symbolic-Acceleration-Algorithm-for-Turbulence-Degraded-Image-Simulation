# Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks

## Paper Overview
Existing physics-based simulations of atmospheric turbulence-degraded images often suffer from high computational cost, which limits their use in large-scale synthetic dataset generation. To address this problem, this paper proposes an accelerated simulation framework that integrates Principal Component Analysis (PCA) with Kolmogorov-Arnold Networks (KANs). PCA is used to decompose spatially varying optical transfer functions into a compact set of fixed basis functions, thereby reformulating anisoplanatic image degradation as a weighted combination of spatially invariant convolutions. A standalone KAN is then employed to learn the nonlinear mapping from Zernike coefficients to the corresponding PCA basis coefficients. Beyond acceleration, the proposed model is designed to provide a physically auditable surrogate simulator. Instead of treating the symbolic KAN expression as a single compact closed-form formula, we use its decomposable computation graph to examine whether the learned mapping follows the optical forward process from Zernike-induced phase perturbations to pupil functions and OTF coefficients. Layer-wise representational similarity analysis, neuron-wise physical attribution, and parameter-level analysis show that shallow layers mainly encode phase- and pupil-related information, whereas deeper layers gradually form OTF-oriented representations. Experimental results demonstrate that the proposed KAN achieves higher fitting accuracy than several benchmark neural networks under comparable parameter budgets. Compared with traditional phase screen segmentation and Zernike polynomial-based algorithms, the proposed framework achieves speedup factors of 42.61 and 5.89, respectively. These results indicate that the PCA–KAN framework provides an efficient and physically auditable surrogate for atmospheric turbulence-degraded image simulation.

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
