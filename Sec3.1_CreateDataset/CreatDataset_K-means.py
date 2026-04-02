# -*- coding: utf-8 -*-
"""
完整的K-means聚类代码：对500万样本进行聚类并可视化分布
功能：加载数据 → 抽样标准化 → MiniBatchKMeans聚类 → 全量预测簇标签 → 统计分布 → 绘制+保存+展示直方图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle

# ---------------------- 1. 配置参数（根据实际路径修改） ----------------------
# 数据路径
INPUT_SOURCE = '/media/aiofm/F/20250723_k-fold-cross-validation-KAN/10^7DataSet/10^7_aj_compressed.npz'
# 输出图表路径（重点：确保此路径可写）
OUTPUT_PLOT_PATH = '/home/aiofm/PycharmProjects/MyKANNet/K-means/cluster_distribution.png'
# 核心参数
TOTAL_SAMPLES = 5000000  # 选取前500万样本
SAMPLE_SIZE = 500000  # 抽样聚类的样本量（10%）
N_CLUSTERS = 20  # 聚类簇数
RANDOM_SEED = 42  # 固定随机种子保证可复现

# ---------------------- 2. 工具函数（新增：路径校验） ----------------------
def ensure_dir_exists(file_path):
    """
    确保文件所在目录存在，不存在则创建
    :param file_path: 目标文件路径
    """
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录：{dir_path}")

# ---------------------- 3. 核心函数定义 ----------------------
def load_data(file_path, total_samples):
    """
    加载npz格式数据，返回指定数量的样本
    :param file_path: 数据文件路径
    :param total_samples: 要加载的样本总数
    :return: 二维数组 (total_samples, feature_dim)
    """
    try:
        with np.load(file_path, mmap_mode='r') as f:
            data = f['data'][:total_samples].astype(np.float32)
        assert data.shape[0] == total_samples, f"加载样本数错误：实际{data.shape[0]}，期望{total_samples}"
        print(f"数据加载完成，形状：{data.shape}")
        return data
    except Exception as e:
        raise RuntimeError(f"加载数据失败：{e}")

def sample_and_normalize(data, sample_size, random_seed):
    """
    抽样并标准化数据（避免全量计算均值/std的内存压力）
    :param data: 全量数据
    :param sample_size: 抽样量
    :param random_seed: 随机种子
    :return: 标准化后的抽样数据、抽样均值、抽样标准差
    """
    # 随机抽样
    sample_idx = np.random.choice(len(data), size=sample_size, replace=False)
    sample_data = data[sample_idx]

    # 计算均值和标准差（添加小值避免除零）
    sample_mean = np.mean(sample_data, axis=0)
    sample_std = np.std(sample_data, axis=0)
    eps = 1e-8
    normalized_sample = (sample_data - sample_mean) / (sample_std + eps)

    # 打乱数据提升聚类效果
    normalized_sample = shuffle(normalized_sample, random_state=random_seed)
    print(f"抽样标准化完成，抽样数据形状：{normalized_sample.shape}")
    return normalized_sample, sample_mean, sample_std

def train_kmeans(normalized_sample, n_clusters, random_seed):
    """
    训练MiniBatchKMeans模型（适合大规模数据）
    :param normalized_sample: 标准化抽样数据
    :param n_clusters: 簇数
    :param random_seed: 随机种子
    :return: 训练好的KMeans模型
    """
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10000,  # 小批次加速训练
        random_state=random_seed,
        n_init='auto',  # 自动选择初始化次数
        max_iter=100  # 最大迭代次数
    )
    print(f"开始训练MiniBatchKMeans（{n_clusters}簇）...")
    kmeans.fit(normalized_sample)
    print("KMeans训练完成")
    return kmeans

def predict_cluster_labels(data, kmeans, sample_mean, sample_std, batch_size=100000):
    """
    批量预测全量数据的簇标签（避免内存溢出）
    :param data: 全量数据
    :param kmeans: 训练好的模型
    :param sample_mean: 抽样均值
    :param sample_std: 抽样标准差
    :param batch_size: 批量大小
    :return: 全量数据的簇标签数组
    """
    total = len(data)
    cluster_labels = np.zeros(total, dtype=np.int32)
    eps = 1e-8

    print("批量预测簇标签...")
    for start in tqdm(range(0, total, batch_size), desc="预测进度"):
        end = min(start + batch_size, total)
        batch_data = data[start:end]
        # 标准化（使用抽样的均值/std）
        normalized_batch = (batch_data - sample_mean) / (sample_std + eps)
        # 预测并保存标签
        cluster_labels[start:end] = kmeans.predict(normalized_batch)

    print(f"簇标签预测完成，标签形状：{cluster_labels.shape}")
    return cluster_labels

def plot_save_and_show_cluster_distribution(cluster_labels, n_clusters, save_path):
    """
    绘制、保存并展示聚类分布直方图（学术论文级样式）
    :param cluster_labels: 全量簇标签
    :param n_clusters: 簇数
    :param save_path: 图表保存路径
    """
    # 前置：确保保存目录存在
    ensure_dir_exists(save_path)

    # 统计各簇样本数
    cluster_count = Counter(cluster_labels)
    cluster_ids = np.arange(n_clusters)
    cluster_sizes = [cluster_count.get(i, 0) for i in cluster_ids]
    total_samples = sum(cluster_sizes)
    cluster_percent = [size / total_samples * 100 for size in cluster_sizes]

    # 恢复默认后端（支持GUI显示），解决保存和展示冲突
    plt.switch_backend('TkAgg')  # 通用GUI后端，Windows/Linux/Mac均兼容
    # 备选后端：如果TkAgg报错，可尝试 'Qt5Agg' 或 'Agg'（仅保存）
    # plt.switch_backend('Qt5Agg')

    # 设置图表样式（适配论文）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文期刊推荐
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 如需显示中文请取消注释
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        cluster_ids, cluster_sizes,
        width=0.7, color='skyblue', edgecolor='navy', alpha=0.8
    )

    # 标注样本数和占比
    for bar, size, percent in zip(bars, cluster_sizes, cluster_percent):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5000,
            f'{size:,}\n({percent:.1f}%)',
            ha='center', va='bottom', fontweight='bold'
        )

    # 设置坐标轴和标题
    ax.set_xlabel('Cluster ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax.set_title(
        f'Distribution of {total_samples:,} Samples Across {n_clusters} K-Means Clusters',
        fontweight='bold', fontsize=14, pad=20
    )
    ax.set_xticks(cluster_ids)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(cluster_sizes) * 1.1)

    # ---------------------- 核心修改：删除总样本数标注 ----------------------
    # 移除了原有的 ax.text(...) 总样本数标注代码

    # 保存图表（先保存再展示）
    try:
        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',  # 确保背景为白色（论文要求）
            edgecolor='none'
        )
        print(f"✅ 聚类分布图已成功保存至：{save_path}")
    except Exception as e:
        raise RuntimeError(f"❌ 保存图表失败：{e}")

    # 展示图表（阻塞式，关闭窗口后程序继续执行）
    print("📊 正在展示聚类分布直方图...（关闭窗口后程序将继续）")
    plt.show()

    # 关闭画布释放内存
    plt.close(fig)

def print_cluster_stats(cluster_labels, n_clusters):
    """打印聚类统计信息（用于论文分析）"""
    cluster_count = Counter(cluster_labels)
    cluster_ids = np.arange(n_clusters)
    cluster_sizes = [cluster_count.get(i, 0) for i in cluster_ids]
    total = sum(cluster_sizes)

    print("\n=== 聚类簇统计信息 ===")
    for i in cluster_ids:
        percent = cluster_sizes[i] / total * 100
        print(f"簇{i:2d}: {cluster_sizes[i]:,} 样本 ({percent:.2f}%)")

    print(f"\n最大簇样本数: {max(cluster_sizes):,} ({max(cluster_sizes) / total * 100:.2f}%)")
    print(f"最小簇样本数: {min(cluster_sizes):,} ({min(cluster_sizes) / total * 100:.2f}%)")
    print(f"平均簇样本数: {np.mean(cluster_sizes):,.0f}")
    print(f"簇样本数变异系数: {np.std(cluster_sizes) / np.mean(cluster_sizes) * 100:.2f}%")

# ---------------------- 4. 主执行流程 ----------------------
if __name__ == '__main__':
    try:
        # 步骤1：加载数据
        data = load_data(INPUT_SOURCE, TOTAL_SAMPLES)

        # 步骤2：抽样并标准化
        normalized_sample, sample_mean, sample_std = sample_and_normalize(
            data, SAMPLE_SIZE, RANDOM_SEED
        )

        # 步骤3：训练KMeans模型
        kmeans_model = train_kmeans(normalized_sample, N_CLUSTERS, RANDOM_SEED)

        # 步骤4：预测全量簇标签
        cluster_labels = predict_cluster_labels(
            data, kmeans_model, sample_mean, sample_std
        )

        # 步骤5：绘制、保存并展示分布直方图（核心修改）
        plot_save_and_show_cluster_distribution(cluster_labels, N_CLUSTERS, OUTPUT_PLOT_PATH)

        # 步骤6：打印统计信息
        print_cluster_stats(cluster_labels, N_CLUSTERS)

    except Exception as e:
        print(f"程序执行失败：{e}")
        exit(1)
