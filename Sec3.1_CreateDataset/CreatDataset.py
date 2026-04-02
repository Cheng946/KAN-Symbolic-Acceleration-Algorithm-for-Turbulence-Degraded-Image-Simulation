import numpy as np
import os
import sympy as sp
import math
from scipy.signal import correlate
import h5py
from typing import Tuple, Dict, Optional

from sklearn.cluster import MiniBatchKMeans  # 适合大数据量的聚类
from sklearn.model_selection import train_test_split

import multiprocessing as mp
from tqdm import tqdm
import gc

from scipy.fft import fft2,ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from multiprocessing import Pool,cpu_count



def genZernikeCoeff(args):
    '''
    Just a simple function to generate random coefficients as needed, conforms to Zernike's Theory. The nollCovMat()
    function is at the heart of this function.

    A note about the function call of nollCovMat in this function. The input (..., 1, 1) is done for the sake of
    flexibility. One can call the function in the typical way as is stated in its description. However, for
    generality, the D/r0 weighting is pushed to the "b" random vector, as the covariance matrix is merely scaled by
    such value.

    :param num_zern: This is the number of Zernike basis functions/coefficients used. Should be numbers that the pyramid
    rows end at. For example [1, 3, 6, 10, 15, 21, 28, 36]
    :param D_r0:
    :return:
    '''
    kappa, num_zern = args
    C = nollCovMat(num_zern, 1, 1)
    e_val, e_vec = np.linalg.eig(C)
    R = np.real(e_vec * np.sqrt(e_val))

    b = np.random.randn(int(num_zern), 1)
    a = np.matmul(R, b)
    a=a[3:]
    a=np.squeeze(a)
    a[np.isinf(a)] = 0
    a[np.isnan(a)] = 0

    return a*kappa

def nollCovMat(Z, D, fried):
    """
    This function generates the covariance matrix for a single point source. See the associated paper for details on
    the matrix itself.

    :param Z: Number of Zernike basis functions/coefficients, determines the size of the matrix.
    :param D: The diameter of the aperture (meters)
    :param fried: The Fried parameter value
    :return:
    """
    C = np.zeros((Z,Z))
    for i in range(Z):
        for j in range(Z):
            ni, mi = nollToZernInd(i+1)
            nj, mj = nollToZernInd(j+1)
            if (abs(mi) == abs(mj)) and (np.mod(i - j, 2) == 0):
                num = math.gamma(14.0/3.0) * math.gamma((ni + nj - 5.0/3.0)/2.0)
                den = math.gamma((-ni + nj + 17.0/3.0)/2.0) * math.gamma((ni - nj + 17.0/3.0)/2.0) * \
                      math.gamma((ni + nj + 23.0/3.0)/2.0)
                coef1 = 0.0072 * (np.pi ** (8.0/3.0)) * ((D/fried) ** (5.0/3.0)) * np.sqrt((ni + 1) * (nj + 1)) * \
                        ((-1) ** ((ni + nj - 2*abs(mi))/2.0))
                C[i, j] = coef1*num/den
            else:
                C[i, j] = 0
    C[0,0] = 1
    return C

def nollToZernInd(j):
    """
    This function maps the input "j" to the (row, column) of the Zernike pyramid using the Noll numbering scheme.

    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))

    return n, m

def zernikeGen(coeff,ZernPoly36, **kwargs):

    result = coeff[:,np.newaxis,np.newaxis] * ZernPoly36

    return result

def genZernPoly(index, x_grid, y_grid):
    """
    This function simply

    :param index:
    :param x_grid:
    :param y_grid:
    :return:
    """
    n,m = nollToZernInd(index)
    radial = radialZernike(x_grid, y_grid, (n,m))
    #print(n,m)
    if m < 0:
        return np.multiply(radial, np.sin(-m * np.arctan2(y_grid, x_grid)))
    else:
        return np.multiply(radial, np.cos(m * np.arctan2(y_grid, x_grid)))

def radialZernike(x_grid, y_grid, z_ind):
    rho = np.sqrt(x_grid ** 2 + y_grid ** 2)
    radial = np.zeros(rho.shape)
    n = z_ind[0]
    m = np.abs(z_ind[1])

    for k in range(int((n - m)/2 + 1)):
        #print(k)
        temp = (-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + m) / 2 - k)
                                                       * np.math.factorial((n - m) / 2 - k))
        radial += temp * rho ** (n - 2*k)

    # radial = rho ** np.reshape(np.asarray([range(int((n - m)/2 + 1))]), (int((n - m)/2 + 1), 1, 1))

    return radial

def RandCreatOTF(Cn2):
    wvl = 0.525e-6;  # 波长
    L = 7000;  # 传播距离
    D = 0.305;  # 观测口径
    k = 2 * np.pi / wvl; #波矢

    # 采样网格数
    N = 128;
    # 目标的采样间隔
    delta0 = L * wvl / (2 * D);

    # 孔径函数
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1


    # 获取当前工作目录
    current_directory = os.getcwd()

    # 要检查的文件名
    file_name = "36—128ZernPoly.npy";

    # 拼接文件路径
    file_path = os.path.join(current_directory, file_name)
    ZernPoly36 = np.load(file_path);


    Cn2 = Cn2;  # 湍流强度1000*1e-16+5e-16=1005*1e-16=1.005e-13
    # 定义符号变量
    z = sp.symbols('z')
    # 定义积分表达式
    expression = (z / L) ** (5 / 3);
    r0 = ((0.423 * (2 * np.pi / wvl) ** 2) * Cn2 * sp.integrate(expression, (z, 0, L))) ** (-3 / 5);

    # 系数跟湍流强度的缩放因子
    kappa = ((D / r0) ** (5 / 3) / (2 ** (5 / 3)) * (2 * wvl / (np.pi * D)) ** 2 * 2 * np.pi) ** (0.5) * L / delta0;
    OTF=np.zeros((1,128,128),dtype=complex)

    a = genZernikeCoeff(36);
    a = a * kappa;
    zernike_stack = zernikeGen(N, a, ZernPoly36);
    # 计算相位
    Fai = np.sum(zernike_stack, axis=2);
    # 根据傅里叶变换得到对偶定理，F(F(w()))计算OTF
    wave = mask * np.exp(1j * 2 * np.pi * Fai);
    # wave进行归一化,离散Parseval定理np.sum(PSF)/(128**2)=np.sum(np.abs(wave)**2)
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / 128 ** 2) / p) ** 0.5)

    cor = correlate(wave, wave, mode='same') * N ** 2
    cor = cor[::-1, ::-1]

    OTF[0,:,:]=cor

    return OTF

def init_worker(zern_poly, mask_data):
    """初始化工作进程的全局变量"""
    global ZernPoly36, mask
    ZernPoly36 = zern_poly
    mask = mask_data

def ComputOTF(a):
    Zernike_stack = zernikeGen(a,ZernPoly36);
    Fai = np.sum(Zernike_stack, axis=0)

    # print(type(Fai),type(mask))

    # 根据傅里叶变换得到对偶定理，F(F(w()))计算OTF
    wave = mask * np.exp(1j * 2 * np.pi * Fai);

    # wave进行归一化,离散Parseval定理np.sum(PSF)/(PSF.shape[0]**2)=np.sum(np.abs(wave)**2)
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / wave.shape[0] ** 2) / p) ** 0.5)

    cor = correlate(wave, wave, mode='same') * N ** 2
    OTF = cor[::-1, ::-1]

    return OTF


# ---------------------- 核心工具函数：加载、预处理、逆变换、保存 ----------------------
def load_and_merge(batch_indices) -> np.ndarray:
    """加载指定批次的NPZ数据并合并"""
    base_path = '/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_PCA70_coff_compressed_batch_{}.npz'
    total_samples = 0
    features = None

    # 第一遍：统计总样本数和特征数
    for idx in batch_indices:
        file_path = base_path.format(idx)
        try:
            with np.load(file_path, mmap_mode='r') as data:
                pca_data = data['data']
                total_samples += pca_data.shape[0]
                if features is None:
                    features = pca_data.shape[1]
                print(f"检测批次 {idx}：形状 {pca_data.shape}")
        except Exception as e:
            print(f"检测批次 {idx} 失败：{str(e)}")

    if features is None:
        raise ValueError("未加载到有效数据，无法确定特征数量")

    # 预分配内存并填充数据
    merged = np.empty((total_samples, features), dtype=np.float32)
    current_idx = 0
    for idx in batch_indices:
        file_path = base_path.format(idx)
        try:
            with np.load(file_path, mmap_mode='r') as data:
                pca_data = data['data']
                batch_size = pca_data.shape[0]
                merged[current_idx:current_idx + batch_size] = pca_data
                current_idx += batch_size
                print(f"加载批次 {idx}：累计样本 {current_idx}")
        except Exception as e:
            print(f"加载批次 {idx} 失败：{str(e)}")

    print(f"合并完成：形状 {merged.shape}\n")
    return merged


def fit_preprocess_params(raw_data: np.ndarray, method: str = 'standard') -> Dict[str, np.ndarray]:
    """仅基于训练集计算预处理参数（关键：避免数据泄露）"""
    if method not in ['standard', 'minmax']:
        raise ValueError("预处理处理方法仅支持 'standard' 或 'minmax'")

    params = {'method': method}
    if method == 'standard':
        mean = np.mean(raw_data, axis=0, keepdims=True)
        std = np.std(raw_data, axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # 避免除零
        params.update({'mean': mean, 'std': std})
    elif method == 'minmax':
        min_val = np.min(raw_data, axis=0, keepdims=True)
        max_val = np.max(raw_data, axis=0, keepdims=True)
        range_val = np.where(max_val == min_val, 1.0, max_val - min_val)
        params.update({'min': min_val, 'max': max_val, 'range': range_val})

    print(f"预处理处理参数计算完成（方法：{method}）")
    return params


def apply_preprocess(data: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """用训练集参数处理任意数据（训练/验证/测试集通用）"""
    if params['method'] == 'standard':
        return (data - params['mean']) / params['std']
    elif params['method'] == 'minmax':
        return (data - params['min']) / params['range']
    else:
        raise ValueError(f"不支持的预处理方法：{params['method']}")


def inverse_preprocess(processed_data: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """逆变换复原原始数据"""
    if params['method'] == 'standard':
        return processed_data * params['std'] + params['mean']
    elif params['method'] == 'minmax':
        return processed_data * params['range'] + params['min']
    else:
        raise ValueError(f"不支持的预处理方法：{params['method']}")


def save_data_with_params(data: np.ndarray, save_path: str, description: str,
                          params: Optional[Dict[str, np.ndarray]] = None):
    """保存数据及预处理参数"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        chunk_size = min(10000, data.shape[0])
        dset = f.create_dataset(
            'data', data=data, dtype='float32',
            chunks=(chunk_size, data.shape[1]),
            compression='gzip', compression_opts=4
        )
        dset.attrs['description'] = description
        dset.attrs['samples'] = data.shape[0]
        dset.attrs['features'] = data.shape[1]

        if params is not None:
            param_group = f.create_group('preprocess_params')
            param_group.attrs['method'] = params['method']
            for key, val in params.items():
                if key != 'method':
                    param_group.create_dataset(key, data=val, dtype='float32')

    print(f"数据已保存至：{save_path}")
    print(f"是否包含预处理参数：{'是' if params is not None else '否'}\n")


def load_data_with_params(load_path: str) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """加载数据及预处理参数"""
    with h5py.File(load_path, 'r') as f:
        data = f['data'][:]
        params = None
        if 'preprocess_params' in f:
            param_group = f['preprocess_params']
            params = {'method': param_group.attrs['method']}
            for key in param_group.keys():
                params[key] = param_group[key][:]

    print(f"数据加载完成：形状 {data.shape}")
    print(f"是否加载到预处理参数：{'是' if params is not None else '否'}")
    return data, params


def normalize_data(data, mean, std, epsilon=1e-8):
    """标准化数据：(x - mean) / (std + epsilon)"""
    return (data - mean) / (std + epsilon)

def denormalize_data(normalized_data, mean, std, epsilon=1e-8):
    """逆标准化：x * (std + epsilon) + mean"""
    return normalized_data * (std + epsilon) + mean


# ---------------------- 新增：预处理参数的单独保存和加载 ----------------------
def save_preprocess_params(params: Dict[str, np.ndarray], save_path: str):
    """专门用于保存预处理参数的方法"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        # 保存方法类型
        f.attrs['method'] = params['method']
        # 保存所有数值参数
        for key, val in params.items():
            if key != 'method':  # 方法已作为属性保存
                f.create_dataset(key, data=val, dtype='float32')

    print(f"预处理参数已单独保存至：{save_path}")


def load_preprocess_params(load_path: str) -> Dict[str, np.ndarray]:
    """专门用于加载预处理参数的方法"""
    with h5py.File(load_path, 'r') as f:
        # 重建参数字典
        params = {'method': f.attrs['method']}
        # 加载所有数值参数
        for key in f.keys():
            params[key] = f[key][:]

    print(f"已加载预处理参数（方法：{params['method']}）")
    return params


if __name__ == "__main__":
#################################参数准备，Parameter preparation#######################################
    wvl = 0.525e-6;  # 波长，wavelength
    L = 7000;  # 传播距离，propagation distance
    D = 0.305;  # 观测孔径， aperture of observation
    k = 2 * np.pi / wvl; #波矢， wave vector
    N = 128; # 采样网格数， number of sampling grid points
    delta0 = L * wvl / (2 * D); # 目标的采样间隔， sampling interval for the target

    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1
    mask=mask.astype(int)
    mask=np.array(mask) # 孔径函数， aperture function

    current_directory = os.getcwd() # 获取当前工作目录， get the current working directory

    file_name = "36—128ZernPoly.npy"; # 保存前36阶Zernike多项式的文件， File for the first 36 Zernike polynomia

    file_path = os.path.join(current_directory, file_name) # 获取Zernike多项式文件路径， get the file path of the Zernike polynomials
    ZernPoly36 = np.load(file_path);
    ZernPoly36 = np.transpose(ZernPoly36, (2, 0, 1))
    ZernPoly36=ZernPoly36[3:,:,:] #获取4到36阶Zernike多项式，数据维度[36,128,128],Get 4th–36th Zernike polynomials, data shape: [36, 128, 128]


    Cn2 = 15 * (1e-16);  # 折射率结构常数， refractive index structure constan
    z = sp.symbols('z') # 定义符号变量， define symbolic variable
    expression = (z / L) ** (5 / 3); # 定义积分表达式，define the integral expression
    r0 = ((0.423 * (2 * np.pi / wvl) ** 2) * Cn2 * sp.integrate(expression, (z, 0, L))) ** (-3 / 5); #计算大气相干长度， Calculate the atmospheric coherence length

    kappa = ((D / r0) ** (5 / 3) / (2 ** (5 / 3)) * (2 * wvl / (np.pi * D)) ** 2 * 2 * np.pi) ** (0.5) * L / delta0; # 湍流强度的缩放因子, Turbulence Intensity Scaling Factor
#########################################################################################################################
###########################生成10^7组Zernike系数a_i, Generate 1e7 Zernike coefficient sets a_i#############################

    num=int(1e7) #设定生成的a_i的组数， Set the number of generated a_i sets

    # 创建一个数据生成器，Create a data generator
    def input_generator():
        for _ in range(num):
            yield (kappa, 36)

    # 使用进程池处理数据， Process data using a process pool
    processes = mp.cpu_count()
    with mp.Pool(processes=processes) as pool:
        # 使用imap_unordered提高效率， Use imap_unordered to improve efficiency
        results_iter = pool.imap_unordered(genZernikeCoeff, input_generator())

        # 分块写入文件以减少内存占用， Write data to file in chunks to reduce memory usage
        chunk_size = 10000
        # a_i保存的路径， Saving path of a_i
        output_path = "/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_aj_compressed.npz"

        # 创建初始数组 ,Create initial array
        first_chunk = np.array([next(results_iter) for _ in range(chunk_size)], dtype=np.float64)
        all_data = [first_chunk]

        # 计算总进度, Calculate total progress
        pbar = tqdm(total=num, unit='units', desc='Data processing progress')
        pbar.update(chunk_size)

        # 处理剩余数据， Process remaining data
        remaining = num - chunk_size
        while remaining > 0:
            current_size = min(chunk_size, remaining)
            chunk = np.array([next(results_iter) for _ in range(current_size)], dtype=np.float64)
            all_data.append(chunk)
            pbar.update(current_size)
            remaining -= current_size

        pbar.close()

    # 合并所有块并保存， Merge all chunks and save
    results = np.vstack(all_data)
    print(f"Result shape: {results.shape}")

    # 保存压缩文件， Save compressed file
    np.savez_compressed(output_path, data=results)
    print(f"Data saved to: {output_path}")
######################################################################################################################################
######################################可视化a_i重构PSF(可注释，不影响计算流程)############################################################
###################################Visualize PSF reconstructed from a_i (can be commented out, does not affect computation flow)######
    # data = np.load("/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_aj_compressed.npz") #加载a_i数据
    # a = data['data']  # 从npz文件中获取数组, Load arrays from NPZ file
    # a = np.array(a, dtype=np.float64)
    # a = a[3256, :]  # 随机选择第3257个样本, Randomly select the 3257th sample
    #
    # # 计算相位， Calculate phase
    # Zernike_stack = zernikeGen(a, ZernPoly36)
    # Fai = np.sum(Zernike_stack, axis=0)
    #
    # print(type(Fai), type(mask))  # 确保mask已定义， Ensure mask is defined
    #
    # # 根据傅里叶变换得到PSF, Obtain PSF via Fourier transform
    # wave = np.exp(1j * 2 * np.pi * Fai)
    # wave = mask * wave
    # PSF = fftshift(fft2(wave))
    #
    # # 显示PSF图像, Display PSF
    # plt.figure(figsize=(10, 8))
    # plt.imshow(np.abs(PSF) ** 2, cmap='viridis')
    # plt.colorbar(label='Intensity')
    # plt.title('Point Spread Function (PSF)')
    # plt.show()

#######################################################################################################################
####################################    生成OTF， Generate OTF        ##################################################
    init_worker(ZernPoly36, mask)
    data = np.load("/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_aj_compressed.npz")
    a = data['data']  # 从npz文件中获取数组, Load arrays from npz file

    total_samples = len(a)
    batch_size = 200000  # 每批处理20万组数据, Process 200,000 data groups per batch
    num_batches = (total_samples + batch_size - 1) // batch_size  # 计算总批次数, Calculate total number of batches

    print(f"Total data volume: {total_samples}, Batch size: {batch_size}, Total batches: {num_batches}")

    # 预获取单个结果的形状和数据类型，用于预分配数组, Pre-fetch shape and dtype of a single result for array pre-allocation
    if total_samples > 0:
        sample_result = ComputOTF(a[0])
        result_shape = sample_result.shape
        result_dtype = sample_result.dtype
        del sample_result  # 释放内存
        gc.collect()
    else:
        print("No data to process")

    # 使用进程池, Use process pool
    with Pool() as pool:
        # 遍历所有批次, Iterate over all batches
        for batch_idx in range(num_batches):
            # 计算当前批次的起始和结束索引, Calculate start and end indices for the current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            batch_data = a[start_idx:end_idx]
            current_batch_size = end_idx - start_idx

            print(f"\nProcess batch {batch_idx + 1}/{num_batches}，Data range: {start_idx}-{end_idx}")

            # 预分配结果数组，避免列表转换的开销, Pre-allocate result array to avoid overhead of list conversion
            batch_OTF = np.empty((current_batch_size,) + result_shape, dtype=result_dtype)

            # 使用tqdm显示当前批次的处理进度，并直接写入预分配的数组,Use tqdm to display batch processing progress and write directly to the pre-allocated array
            for i, result in enumerate(tqdm(pool.imap(ComputOTF, batch_data),
                                            total=current_batch_size,
                                            desc=f"Batch {batch_idx + 1} Progress")):
                batch_OTF[i] = result

            # 保存当前批次结果，Save current batch results
            save_path = f'/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_OTF_compressed_batch_{batch_idx + 1}.npz'
            np.savez_compressed(save_path, data=batch_OTF)
            print(f"Batch {batch_idx + 1} saved to: {save_path}")

            # 清理内存,Clear memory
            del batch_OTF
            gc.collect()

    print("\nAll data processing completed!")

#####################################################################################################################
##################################  可视化OTF,可注释掉，不影响进程 #######################################################
##################################  Visualize OTF (can be commented out, does not affect process)   #################
    # data = np.load("/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_OTF_compressed_batch_1.npz")
    # OTF = data['data']  # 从npz文件中获取数组
    # OTF = np.array(OTF, dtype=complex)
    #
    # plt.imshow(np.abs(OTF[4200,:,:]))
    # plt.show()


#############################################################################################################
###################################获取PCA模型################################################################
###################################Get PCA model############################################################
    data = np.load("/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_OTF_compressed_batch_1.npz")
    OTF = data['data']  # 从npz文件中获取数组, Load arrays from npz file
    OTF = np.array(OTF, dtype=complex)
    Crop_OTF = OTF[:100000, :, :]

    num_matrix = Crop_OTF.shape[0]
    matrix_size = Crop_OTF.shape[1]

    #将复数矩阵转换为实数矩阵, Convert complex matrix to real matrix
    real_matrix = np.hstack((Crop_OTF.real, Crop_OTF.imag)).reshape(num_matrix, 2 * matrix_size * matrix_size)

    print(real_matrix.shape)
    # 使用Kernel PCA降维，Use Kernel PCA for dimensionality reduction
    n_componets = 70  # 降维后的维度，Reduced dimension

    pca = PCA(n_components=n_componets)
    reduced_matrices = pca.fit_transform(real_matrix)

    #保存模型参数，Save model parameters
    model_filename = '/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl'
    joblib.dump(pca, model_filename)

##########################################################################################################################################
##########################################展示PCA分解得到基函数结果(可注释，不影响运行)##############################################################
#########################Display basis functions obtained by PCA decomposition (can be commented out, does not affect execution)############

    # pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl')
    # decomp=pca.components_
    # matrix_size=128
    # # print(decomp.shape)
    #
    # approx_real_part=decomp[:,:matrix_size*matrix_size].reshape(70,matrix_size,matrix_size)
    # approx_imag_part=decomp[:,matrix_size*matrix_size:].reshape(70,matrix_size,matrix_size)
    # approx_complex_matrix=approx_real_part+1j*approx_imag_part
    #
    # # plt.imshow(np.abs(approx_complex_matrix[3,:,:]))
    #
    # #这里望远镜为MEADE的LX600-ACF焦距为f=2.438m,焦比=8,波长为namuda=0.525e-6m,振幅场网格128*128，网格大小为2.38*10^-3,傅里叶变换后d_fx=3.2787m^-1,
    # #The telescope used is the MEADE LX600-ACF with a focal length f=2.438m,
    # #focal ratio f/#=8, and wavelength λ=0.525×10−6m.
    # #The amplitude field grid is 128×128 with a grid spacing of 2.38×10−3m,
    # #and the spatial frequency interval after Fourier transform is d_fx=3.2787m−1.

    # #PSF网格大小为d_u=d_fx*namuda*f=4.197*10^-6m,OTF网格大小为d_m=1/(N*d_u)=1.8614*10^-3μm
    # #PSF grid size:d_u=d_fx*namuda*f=4.197*10^-6m， OTF grid size:d_m=1/(N*d_u)=1.8614*10^-3μm
    # PSF = np.abs(ifftshift(ifft2(approx_complex_matrix[0, :, :])))
    #
    # # 物理参数设置， Physical Parameter Setup
    # grid_size = 4.197  # 每个网格的微米数，Micrometers per grid
    # total_pixels = matrix_size  # 总像素数，Total number of pixels
    # half_pixels = total_pixels // 2  # 半像素数，Half the number of pixels
    #
    # # 计算从中心向两边扩展的实际物理范围，Calculate the actual physical range extending from the center to both sides
    # # 中心为原点(0,0)，向左右上下各扩展的距离，With the center as the origin (0,0), calculate the extension distances to the left, right, top, and bottom.
    # max_extent = half_pixels * grid_size  # 约为269微米，Approximately 269 micrometers.
    #
    # # 绘制图像，设置坐标范围以中心为基准向两边扩展，Plot the image and set the coordinate range to extend outward from the center.
    # # extent参数定义了图像四个边界的物理坐标，The `extent` parameter defines the physical coordinates of the four boundaries of the image.
    # # 创建图形，Create the figure.
    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    # # 绘制图像，设置坐标范围以中心为基准向两边扩展，Plot the image and set the coordinate range to extend symmetrically from the center.
    # im = ax.imshow(PSF,
    #                extent=[-max_extent, max_extent, -max_extent, max_extent],
    #                origin='lower',
    #                cmap='viridis')
    #
    # # 设置坐标轴标签，包含单位， Set axis labels with physical units
    # ax.set_xlabel('X (μm)', fontsize=20)
    # ax.set_ylabel('Y (μm)', fontsize=20)
    #
    #
    # # 设置网格，Set the grid
    # # 计算网格刻度位置 - 以grid_size为间距， Calculate grid ticks at intervals of grid_size
    # # 生成从 -max_extent 到 max_extent 步长为 grid_size 的刻度， Generate ticks: -max_extent → max_extent, step = grid_size
    # ticks = np.arange(-max_extent, max_extent + grid_size, grid_size)
    #
    # # 为了避免刻度过多导致重叠，我们可以每隔n个网格显示一个刻度，Display one tick every n grids to prevent overlap
    # # 这里选择每隔10个网格显示一个刻度，你可以根据需要调整，Display one tick every 10 grids (adjustable)
    # tick_interval = 20
    # show_ticks = ticks[::tick_interval]
    #
    # # 设置坐标轴刻度，Set axis ticks
    # ax.set_xticks(show_ticks)
    # ax.set_yticks(show_ticks)
    #
    # # 添加网格线，Add grid lines
    # ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    #
    # # 调整刻度标签字体大小，Adjust the font size of tick labels
    # ax.tick_params(axis='both', which='major', labelsize=20)
    #
    # # 添加colorbar并设置标签，Add colorbar and set its label
    # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #
    # cbar.ax.tick_params(labelsize=20)
    #
    # # 调整布局，防止标签重叠，Adjust layout to prevent label overlap
    # plt.tight_layout()
    #
    # # 显示图像，Show the image
    # plt.show()


###############################################################################################################
########################################## PCA分解OTF ##########################################################
########################################## PCA decomposition of OTF ############################################
    pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl')
    chunk_size = 50000  # 每块处理的数据量，可根据内存大小调整，Data volume per block is adjustable based on memory size

    for batch_idx in range(50):
        OTF_path = f'/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_OTF_compressed_batch_{batch_idx + 1}.npz'
        data = np.load(OTF_path)
        OTF = data['data']
        print(f'已载入{OTF_path}')
        del data
        gc.collect()

        num_matrix = OTF.shape[0]
        matrix_size = OTF.shape[1]

        # 将复数矩阵转换为实数矩阵，Convert complex-valued matrix to real-valued matrix
        real_matrix = np.hstack((OTF.real, OTF.imag)).reshape(num_matrix, 2 * matrix_size * matrix_size)

        # 初始化存储结果的列表，Initialize result storage list
        reduced_chunks = []

        # 分块处理，Block processing
        for i in range(0, num_matrix, chunk_size):
            # 计算当前块的结束索引，Calculate end index for current block
            end_idx = min(i + chunk_size, num_matrix)

            # 处理当前块，Process current block
            chunk = real_matrix[i:end_idx]
            reduced_chunk = pca.transform(chunk)
            reduced_chunks.append(reduced_chunk)

            # 打印进度，Print progress
            print(f'Batch {batch_idx + 1} Processed {end_idx}/{num_matrix} entries', end='\r')

        # 合并所有块的结果,Merge results from all blocks
        reduced_matrices = np.vstack(reduced_chunks)
        print(f'\nBatch {batch_idx + 1} dimensionality reduction completed, result shape: {reduced_matrices.shape}')

        save_path = f'/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_PCA70_coff_compressed_batch_{batch_idx + 1}.npz'
        np.savez_compressed(save_path, data=reduced_matrices)

        print(f"Batch {batch_idx + 1} saved to: {save_path}")
        # 释放real_matrix和reduced_chunks,Free memory: real_matrix, reduced_chunks
        del OTF,real_matrix, chunk, reduced_chunk, reduced_chunks
        gc.collect()

#################################################################################################################
############################################## 展示重构效果 ################################################
    # data = np.load('/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_PCA70_coff_compressed_batch_2.npz')
    # PCA_70_coff = data['data']  # 从npz文件中获取数组
    # matrix_size = 128
    #
    # pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl')
    # approx_complex_matrix=pca.inverse_transform(PCA_70_coff[126:127])
    # approx_real_part=approx_complex_matrix[:,:matrix_size*matrix_size].reshape(matrix_size,matrix_size)
    # approx_imag_part=approx_complex_matrix[:,matrix_size*matrix_size:].reshape(matrix_size,matrix_size)
    # approx_complex_matrix = approx_real_part + 1j * approx_imag_part
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.abs(approx_complex_matrix))
    #
    # data = np.load('/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_OTF_compressed_batch_2.npz')
    # OTF=data['data']  # 从npz文件中获取数组
    #
    # OTF = np.array(OTF, dtype=complex)
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.abs(OTF[126,:,:]))
    # plt.show()


    # PCA_70=np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_PCA70.npy')
    # PCA_70 = np.load('/media/aiofm/F/20250723_k-fold-cross-validation-KAN/100000DataSet/50000-100000_PCA70.npy')
    # PCA_70=np.array(PCA_70,dtype=np.float64)
    # matrix_size=128
    #
    # pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/pca_model-70.pkl')
    # approx_complex_matrix=pca.inverse_transform(PCA_70[126:127])
    # approx_real_part=approx_complex_matrix[:,:matrix_size*matrix_size].reshape(matrix_size,matrix_size)
    # approx_imag_part=approx_complex_matrix[:,matrix_size*matrix_size:].reshape(matrix_size,matrix_size)
    # approx_complex_matrix = approx_real_part + 1j * approx_imag_part
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.abs(approx_complex_matrix))
    #
    #
    # OTF = np.load('/media/aiofm/F/20250723_k-fold-cross-validation-KAN/100000DataSet/50000-100000_OTF.npy')
    # OTF = np.array(OTF, dtype=complex)
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.abs(OTF[126,:,:]))
    # plt.show()

#########################################################################################################
########################################## k折交叉分割数据集 #####################################################
########################################## K-fold cross-validation split ################################

    input_source = '/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_aj_compressed.npz'
    output_batch_template = '/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_PCA70_coff_compressed_batch_{}.npz'
    samples_per_batch = 200000  # 每批样本数，Batch size
    total_samples = 5000000  # 前500万样本，First 5 million samples
    total_batches = 25  # 1-25号输出文件，Output files 1 to 25

    # K折参数，K-fold parameters
    K = 8  # 总折数，Total folds
    current_fold = 1  # 当前生成第1折，Current fold: 1 (1-based)
    assert current_fold <= K, f"Current fold {current_fold} Cannot exceed total folds{K}"

    # 独立测试集占比（20%），Independent test set ratio (20%)
    test_ratio = 0.2
    test_target = int(total_samples * test_ratio)  # 100万，1 million
    kfold_total = total_samples - test_target  # 400万， 4 million

    # 输出路径，Output path
    base_dir = '/media/aiofm/F/20250723_k-fold-cross-validation-KAN'
    train_path = f'{base_dir}/{current_fold}-fold/Train'
    val_path = f'{base_dir}/{current_fold}-fold/Val'
    test_path = f'{base_dir}/{current_fold}-fold/Test'

    # 模型和统计量保存路径，Model and statistics save path
    kmeans_model_save_path = f'{base_dir}/kmeans_models/kmeans_cluster_20.pkl'
    cluster_stats_save_path = f'{base_dir}/kmeans_models/cluster_stats_20.npz'

    # 创建目录，Create directory
    os.makedirs(os.path.dirname(kmeans_model_save_path), exist_ok=True)
    for path in [train_path, val_path, test_path]:
        os.makedirs(path, exist_ok=True)

    # 数据输出路径，Data output path
    train_input = f'{train_path}/Train_input.h5'
    train_output = f'{train_path}/Train_output.h5'
    val_input = f'{val_path}/Val_input.h5'
    val_output = f'{val_path}/Val_output.h5'
    test_input = f'{test_path}/Test_input.h5'
    test_output = f'{test_path}/Test_output.h5'
    stats_path = f'{base_dir}/{current_fold}-fold/input_preprocessing_stats.h5'


    # 标准化函数，Standardization function
    def normalize_data(data, mean, std):
        return (data - mean) / (std + 1e-8)


    # ---------------------- Tool function ----------------------
    #######################加载单个输出文件, Load single output file##############
    def load_output_batch(batch_id):

        try:
            path = output_batch_template.format(batch_id)
            with np.load(path, mmap_mode='r') as f:
                return batch_id, f['data'][:].astype('float32')
        except Exception as e:
            print(f"Load output files {batch_id} failed：{e}")
            return batch_id, None


    # ---------------------- 核心步骤1：训练并保存KMeans模型（独立流程） ----------------------
######################Core Step 1: Train and save the KMeans model (independent process) #####################
    def train_and_save_kmeans(data, save_model_path, save_stats_path):
        ############训练KMeans模型并保存到指定路径,Train & save KMeans model to path #############################
        print("===== Start training KMeans=====")
        sample_size = 500000
        sample_idx = np.random.choice(total_samples, size=sample_size, replace=False)
        sample_data = data[sample_idx]

        # 计算聚类用的标准化参数,Calculate standardization parameters for clustering
        sample_mean = np.mean(sample_data, axis=0)
        sample_std = np.std(sample_data, axis=0)
        normalized_sample = (sample_data - sample_mean) / (sample_std + 1e-8)

        # 训练模型,Train the model
        n_clusters = 20
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=10000,
            random_state=42,
            n_init='auto'
        )
        kmeans.fit(normalized_sample)

        # 保存模型和统计量,Save model and statistics
        joblib.dump(kmeans, save_model_path)
        np.savez(save_stats_path, sample_mean=sample_mean, sample_std=sample_std)
        print(f"✅ Model saved to：{save_model_path}")
        print(f"✅ Clustering statistics saved to：{save_stats_path}")

        return kmeans, sample_mean, sample_std


    # ---------------------- 核心步骤2：加载KMeans模型 ----------------------
######################Core Step 2: Load KMeans Model ########################
    def load_kmeans_model(load_model_path, load_stats_path):
##############Load KMeans model and statistics from the specified path#######
        print("===== Start loading the saved KMeans model =====")
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Model file does not exist：{load_model_path}")
        if not os.path.exists(load_stats_path):
            raise FileNotFoundError(f"Statistics file does not exist：{load_stats_path}")

        # 加载模型（核心操作）
        kmeans_loaded = joblib.load(load_model_path)
        cluster_stats = np.load(load_stats_path)
        sample_mean_loaded = cluster_stats['sample_mean']
        sample_std_loaded = cluster_stats['sample_std']

        print(f"✅ Model loaded successfully：{load_model_path}")
        print(f"✅ Clustering statistics loaded successfully：{load_stats_path}")
        return kmeans_loaded, sample_mean_loaded, sample_std_loaded


    # 加载原始数据,Load raw data
    with np.load(input_source, mmap_mode='r') as input_data:
        A_j = input_data['data']
        input_data_5m = A_j[:total_samples]
        assert input_data_5m.shape == (total_samples, 33), f"Input data shape error：{input_data_5m.shape}"

        # # 步骤1：训练并保存模型（仅首次运行需要，后续可注释）,Step 1: Train & save model (first run only, comment later)
        # kmeans_trained, sample_mean_trained, sample_std_trained = train_and_save_kmeans(
        #     data=input_data_5m,
        #     save_model_path=kmeans_model_save_path,
        #     save_stats_path=cluster_stats_save_path
        # )

        # 步骤2：强制加载已保存的模型（后续所有操作都用这个加载后的模型）,Step 2: Force load saved model (use for all later steps)
        kmeans, sample_mean, sample_std = load_kmeans_model(
            load_model_path=kmeans_model_save_path,
            load_stats_path=cluster_stats_save_path
        )

        # 步骤3：使用加载后的模型进行簇标签预测（核心验证）,Predict cluster labels using the loaded model (core validation)
        print("===== Predict cluster labels using the loaded model =====")
        cluster_labels = np.zeros(total_samples, dtype=int)
        batch_size = 100000
        for i in tqdm(range(0, total_samples, batch_size), desc="Batch predict cluster labels"):
            end = min(i + batch_size, total_samples)
            batch_data = input_data_5m[i:end]
            normalized_batch = (batch_data - sample_mean) / (sample_std + 1e-8)
            # 使用加载后的模型predict,Predict using the loaded model
            cluster_labels[i:end] = kmeans.predict(normalized_batch)

        # 拆分独立测试集（使用加载模型的结果）,Split independent test set
        print("===== Split independent test set =====")
        test_indices = []
        kfold_indices = []
        for cluster_id in range(20):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            idx_kfold, idx_test = train_test_split(
                cluster_indices, test_size=test_ratio, random_state=42, shuffle=True
            )
            test_indices.extend(idx_test)
            kfold_indices.extend(idx_kfold)
        test_indices = np.array(test_indices)[:test_target]
        test_indices.sort()
        kfold_indices = np.array(kfold_indices)

        # K折划分（使用加载模型的结果）
        print(f"===== Split data into {K} folds =====")
        fold_indices = [[] for _ in range(K)]
        for cluster_id in tqdm(range(20), desc=f"Split into {K} folds"):
            cluster_mask = (cluster_labels[kfold_indices] == cluster_id)
            cluster_in_kfold = kfold_indices[cluster_mask]
            n_cluster = len(cluster_in_kfold)
            if n_cluster == 0:
                continue
            np.random.shuffle(cluster_in_kfold)
            fold_sizes = [n_cluster // K] * K
            for i in range(n_cluster % K):
                fold_sizes[i] += 1
            start = 0
            for fold in range(K):
                end = start + fold_sizes[fold]
                fold_indices[fold].extend(cluster_in_kfold[start:end])
                start = end

        val_indices = np.array(fold_indices[current_fold - 1])
        train_indices = np.concatenate([fold_indices[i] for i in range(K) if i != current_fold - 1])
        train_indices.sort()
        val_indices.sort()
        print(f"Number of training samples for fold {current_fold}：{len(train_indices)}")
        print(f"Number of validation samples for fold {current_fold}：{len(val_indices)}")

    # 处理输入数据,Process input data
    with np.load(input_source, mmap_mode='r') as input_data:
        A_j = input_data['data']
        input_data_5m = A_j[:total_samples]
        input_features = 33

        # 计算标准化参数,Calculate normalization parameters
        print("===== Calculate normalization parameters for input data =====")
        train_mean = np.zeros(input_features, dtype=np.float64)
        train_std = np.zeros(input_features, dtype=np.float64)
        batch_size = 100000

        # 计算均值，Calculate the mean
        count = 0
        for i in tqdm(range(0, len(train_indices), batch_size), desc="Calculate the mean"):
            end = min(i + batch_size, len(train_indices))
            batch_idx = train_indices[i:end]
            batch_data = input_data_5m[batch_idx].astype(np.float64)
            train_mean += np.sum(batch_data, axis=0)
            count += len(batch_idx)
        train_mean /= count

        # 计算标准差，Calculate the standard deviation
        count = 0
        for i in tqdm(range(0, len(train_indices), batch_size), desc="Calculate the standard deviation"):
            end = min(i + batch_size, len(train_indices))
            batch_idx = train_indices[i:end]
            batch_data = input_data_5m[batch_idx].astype(np.float64)
            train_std += np.sum((batch_data - train_mean) ** 2, axis=0)
            count += len(batch_idx)
        train_std = np.sqrt(train_std / (count - 1))

        # 保存统计量
        with h5py.File(stats_path, 'w') as f:
            f.create_dataset('input_mean', data=train_mean)
            f.create_dataset('input_std', data=train_std)


        # 保存输入数据，Save statistics
        def save_input(path, indices):
            print(f"Save input data to{path}...")
            with h5py.File(path, 'w') as f:
                dset = f.create_dataset(
                    'data', shape=(len(indices), input_features),
                    dtype='float32', chunks=(10000, input_features)
                )
                for i in tqdm(range(0, len(indices), batch_size), desc="Write input"):
                    end = min(i + batch_size, len(indices))
                    batch_idx = indices[i:end]
                    batch_data = input_data_5m[batch_idx]
                    dset[i:end] = normalize_data(batch_data, train_mean, train_std)
            return len(indices)


        train_count = save_input(train_input, train_indices)
        val_count = save_input(val_input, val_indices)
        test_count = save_input(test_input, test_indices)

    # 处理输出数据,Process output data
    print("===== Process output data =====")
    batch_ids = list(range(1, total_batches + 1))
    n_processes = max(1, cpu_count() // 2)
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(load_output_batch, batch_ids),
            total=len(batch_ids),
            desc="Load output file"
        ))

    output_batches = {batch_id: data for batch_id, data in results if data is not None}
    assert len(output_batches) == total_batches, "Failed to load some output files"


    def fast_save_output(path, indices):
        print(f"Quickly save output data to{path}...")
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset(
                'data', shape=(len(indices), 70),
                dtype='float32', chunks=(10000, 70)
            )
            batch_size = 100000
            for i in tqdm(range(0, len(indices), batch_size), desc="Write output"):
                end = min(i + batch_size, len(indices))
                batch_indices = indices[i:end]
                batch_ids = (batch_indices // samples_per_batch) + 1
                in_batch_indices = batch_indices % samples_per_batch
                batch_output = np.array([
                    output_batches[bid][idx] for bid, idx in zip(batch_ids, in_batch_indices)
                ], dtype='float32')
                dset[i:end] = batch_output
        return len(indices)


    fast_save_output(train_output, train_indices)
    fast_save_output(val_output, val_indices)
    fast_save_output(test_output, test_indices)

    del output_batches

    print("\n===== All operations completed =====")
    print(f"✅ Fold {current_fold} training set: {train_count} samples")
    print(f"✅ Fold {current_fold} validation set: {val_count} samples")
    print(f"✅ Independent test set: {test_count} samples")
    print(f"✅ Use the loaded KMeans model throughout all clustering-related operations.")





