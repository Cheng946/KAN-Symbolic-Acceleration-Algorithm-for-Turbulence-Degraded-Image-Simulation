# Acceleration Algorithm Simulation for Atmospheric Turbulence Degraded Remote Sensing Images Based on Kolmogorov-Arnold Network

## Paper Overview
In the process of acquiring images from spaceborne and airborne remote sensing platforms, image degradation caused by atmospheric turbulence seriously restricts the efficient processing and application of remote sensing data. Aiming at the current challenge that atmospheric turbulence degraded image simulation struggles to meet the processing efficiency requirements of modern large-scale remote sensing datasets, this paper applies Principal Component Analysis (PCA), Kolmogorov-Arnold Net-work (KAN), and Feature-wise Linear Modulation (FiLM) structures to simulation computations, proposing an accelerated algorithm for atmospheric turbulence degra-dation simulation specifically designed for remote sensing images. Experimental re-sults demonstrate that the algorithm achieves an average processing time of 3.08 sec-onds per remote sensing image, significantly outperforming the traditional phase screen segmentation algorithm (131.23 seconds per image) and the Zernike polynomi-als algorithm (18.14 seconds per image). The computational efficiency of the proposed method is 42.61 times and 5.89 times that of the traditional methods, respectively. While accelerating the algorithm, it introduces only a 0.96% mean absolute percentage error. This algorithm provides an efficient atmospheric turbulence degradation simu-lation scheme for the preprocessing of remote sensing images, contributing to im-proving the processing quality and efficiency of remote sensing images in geometric reconstruction, data fusion, and other remote sensing applications.

**Model Architecture**：
- Backbone Network： Kolmogo-rov-Arnold Network+FiLM
- Core module：KANLinear、b_splines、FilM
- Input/Output Dimensions：33，70

## 环境配置
### 硬件要求
- NVIDIA GPU A40
- CUDA ≥11.6

### 依赖安装
```bash
conda create -n [env_name] python=3.10.10
conda activate [env_name]
pip install -r requirements.txt
