# Accelerated Simulation of Atmospheric Turbulence-Degraded Images via Interpretable Kolmogorov-Arnold Networks

## Paper Overview
Current simulations of image degradation caused by atmospheric turbulence struggle to meet the efficiency demands of large-scale datasets. This paper proposes an accelerated simulation algorithm by integrating Principal Component Analysis (PCA) and the Kolmogorov-Arnold Network (KAN). Unlike black-box models, KAN offers an interpretable physical mechanism through its learnable spline activation functions and symbolic computation capabilities. Leveraging its compact architecture, KAN expresses the degradation mapping as a closed-form function of individual Zernike coefficients. This enables an in-depth interpretable analysis of key phenomena, such as the dominance of low-order aberrations and local control
effects. Experimental results demonstrate that KAN outperforms various benchmark neural networks in fitting accuracy given the same parameter budget. Furthermore, compared with traditional phase screen segmentation and Zernike polynomial-based algorithms, the proposed method achieves speedup factors of 42.61 and 5.89, respectively. These findings confirm that KAN, as an interpretable acceleration framework, possesses significant potential for advancing turbulence-degraded imaging tasks. 

**Model Architecture**：
- Backbone Network： Kolmogorov-Arnold Network
- Core module：KAN
- Input/Output Dimensions：33，70

## 环境配置,Environment Setup
### 软件要求, Software Requirements
torch==1.13.1+cu116
numpy==1.26.4

### 硬件要求
- NVIDIA GPU A40
- CUDA ≥11.6

### 依赖安装
```bash
conda create -n [env_name] python=3.10.10
conda activate [env_name]
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu116/torch_stable.html
