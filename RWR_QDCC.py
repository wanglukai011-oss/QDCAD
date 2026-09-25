import os
import torch
import numpy as np
from torch_geometric.data import Data
import random
from collections import defaultdict
from tqdm import tqdm
from data_process.Community_Search import select_nodes_via_label
import time
from sklearn.mixture import GaussianMixture
from scipy.stats import hypergeom
from sklearn.mixture import GaussianMixture
from scipy.stats import hypergeom
import time
from tqdm import tqdm

# ========== 调试配置：开启CUDA同步执行，精准定位报错行，调试必备 ==========
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ====================== 全局基础配置 - 所有参数可按需调整 ======================
# 运行设备：优先使用GPU加速，无GPU自动使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 固定随机种子，保证实验结果可复现
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# 数据类型：float32，兼顾精度和显存占用，GNN标准配置
DTYPE = torch.float32
EPS = 1e-8  # 极小值，防止除零错误和NaN值，所有除法必加

# ========== 全局缓存：减少重复计算，提升运行效率 ==========
_SIM_CACHE = {}
_DEGREE_CACHE = {}


# ====================== 图数据预处理函数【GPU加速版+内存优化】- 无改动+完整注释 ======================
def preprocess_graph_gpu(edges_df, features_df):
    """
    极速版：图数据预处理，构建 CSR 格式索引实现 O(1) 邻居查询
    """
    node_id_list = features_df['node_id'].tolist()
    node2idx = {node_id: idx for idx, node_id in enumerate(node_id_list)}
    idx2node = {idx: node_id for node_id, idx in node2idx.items()}
    N = len(node_id_list)

    edge_u = [node2idx[uid] for uid in edges_df.iloc[:, 0].tolist()]
    edge_v = [node2idx[vid] for vid in edges_df.iloc[:, 1].tolist()]
    all_edge_index = torch.tensor([edge_u, edge_v], dtype=torch.long, device=DEVICE)
    all_edge_index = torch.cat([all_edge_index, all_edge_index.flip(0)], dim=1)

    # ================= 🚀 核心优化：构建 CSR (Compressed Sparse Row) 索引 =================
    src, dst = all_edge_index
    # 按照 src 排序，保证同一节点的邻居在内存中连续
    sort_idx = torch.argsort(src)
    src = src[sort_idx]
    dst = dst[sort_idx]
    all_edge_index = torch.stack([src, dst])

    # 计算 ptr (指针数组)：ptr[i] 到 ptr[i+1] 的切片就是节点 i 的所有邻居
    degrees = torch.bincount(src, minlength=N)
    ptr = torch.zeros(N + 1, dtype=torch.long, device=DEVICE)
    ptr[1:] = torch.cumsum(degrees, dim=0)
    # ====================================================================================

    feature_cols = features_df.columns[1:-1]
    node_feat_np = features_df[feature_cols].values.astype(np.float32)
    node_feat_np = node_feat_np / (np.linalg.norm(node_feat_np, axis=1, keepdims=True) + EPS)
    node_feat_tensor = torch.tensor(node_feat_np, dtype=DTYPE, device=DEVICE)

    node_label_tensor = torch.tensor(features_df['label'].values, dtype=torch.long, device=DEVICE)

    # 注意：我们返回了 ptr 和 dst，彻底淘汰了慢速的 mask 查找
    return all_edge_index, node2idx, idx2node, node_feat_tensor, node_label_tensor, ptr, dst


def simple_rwr_baseline(q_id, node2idx, idx2node, all_edge_index,
                        ptr, dst, node_feat_tensor, node_label_tensor,
                        min_community_size, max_community_size, q_quantile,
                        restart_prob=0.15, max_rwr_iter=100, rwr_tol=1e-6):
    """
    纯拓扑 RWR 社区搜索 (极简 Baseline)

    策略：
        - 所有边权重视为 1，进行标准重启随机游走
        - 直接选取 RWR 稳态得分最高的 top‑max_community_size 个节点
        - 无任何阶段扩张、阈值调节或质量守恒逻辑
    参数保留原函数签名但 q_quantile / min_community_size 在本版本中未使用
    """
    # 1. 查询节点校验
    q_idx = node2idx.get(q_id, -1)
    if q_idx < 0:
        return None
    N = node_feat_tensor.size(0)

    # 2. 构建均匀权重的转移矩阵
    deg = ptr[1:] - ptr[:-1]
    src = torch.arange(N, device=DEVICE).repeat_interleave(deg)
    edge_weights = torch.ones(dst.size(0), device=DEVICE)

    sum_w = torch.zeros(N, device=DEVICE)
    sum_w.index_add_(0, src, edge_weights)
    sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))
    norm_weights = edge_weights / sum_w[src]  # 1 / deg(src)

    # 3. 幂迭代 RWR
    p = torch.zeros(N, device=DEVICE)
    p[q_idx] = 1.0
    for _ in range(max_rwr_iter):
        p_prop = torch.zeros(N, device=DEVICE)
        p_prop.index_add_(0, dst, p[src] * norm_weights)
        p_new = (1 - restart_prob) * p_prop
        p_new[q_idx] += restart_prob
        if torch.abs(p_new - p).sum() < rwr_tol:
            p = p_new
            break
        p = p_new
    rwr_scores = p

    # 4. 直接选 top-k 节点 (保证查询节点在内)
    k = min(max_community_size, N)
    topk = torch.argsort(rwr_scores, descending=True)[:k].tolist()
    if q_idx not in topk:
        # 如果查询节点意外未入 topk，强制加入并维持总数不变
        if len(topk) >= max_community_size:
            topk[-1] = q_idx
        else:
            topk.append(q_idx)
    community_idx = topk

    # 5. 子图打包 (与原函数完全一致)
    community_idx_tensor = torch.tensor(community_idx, dtype=torch.long, device=DEVICE)

    x = node_feat_tensor[community_idx_tensor].detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    node_labels = node_label_tensor[community_idx_tensor].detach()
    local_q_idx = community_idx.index(q_idx)

    mask = torch.isin(all_edge_index[0], community_idx_tensor) & \
           torch.isin(all_edge_index[1], community_idx_tensor)
    edge_index_global = all_edge_index[:, mask]

    if edge_index_global.shape[1] > 0:
        global2local = torch.zeros(N, dtype=torch.long, device=DEVICE)
        global2local[community_idx_tensor] = torch.arange(len(community_idx), device=DEVICE)
        edge_index = global2local[edge_index_global]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    avg_quality = torch.mean(rwr_scores[community_idx_tensor]).item()

    return Data(x=x, edge_index=edge_index, node_labels=node_labels,
                q_id=torch.tensor([local_q_idx], dtype=torch.long, device=DEVICE),
                y=torch.tensor([-1], dtype=torch.long, device=DEVICE),
                community_quality=torch.tensor([avg_quality], dtype=DTYPE, device=DEVICE),
                p_val=None)


def topo_rwr_topk_baseline(q_id, node2idx, idx2node, all_edge_index,
                           ptr, dst, node_feat_tensor, node_label_tensor,
                           min_community_size, max_community_size, q_quantile,
                           restart_prob=0.15, max_rwr_iter=100, rwr_tol=1e-6):
    """
    消融基线 ①：纯拓扑 RWR (均匀边权重) + top‑k 选择

    与主实验函数接口完全一致，但：
      - 所有边权重视为 1（标准随机游走）
      - 直接取 RWR 得分最高的 max_community_size 个节点构成社区
      - min_community_size、q_quantile 在本函数中未使用
    """
    # 1. 查询节点校验
    q_idx = node2idx.get(q_id, -1)
    if q_idx < 0:
        return None
    N = node_feat_tensor.size(0)

    # 2. 均匀边权重，行归一化
    deg = ptr[1:] - ptr[:-1]
    src = torch.arange(N, device=DEVICE).repeat_interleave(deg)
    edge_weights = torch.ones(dst.size(0), device=DEVICE)

    sum_w = torch.zeros(N, device=DEVICE)
    sum_w.index_add_(0, src, edge_weights)
    sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))
    norm_weights = edge_weights / sum_w[src]  # 1/deg(src)

    # 3. 幂迭代 RWR
    p = torch.zeros(N, device=DEVICE)
    p[q_idx] = 1.0
    for _ in range(max_rwr_iter):
        p_prop = torch.zeros(N, device=DEVICE)
        p_prop.index_add_(0, dst, p[src] * norm_weights)
        p_new = (1 - restart_prob) * p_prop
        p_new[q_idx] += restart_prob
        if torch.abs(p_new - p).sum() < rwr_tol:
            p = p_new
            break
        p = p_new
    rwr_scores = p

    # 4. 取 top-k (k = max_community_size，但不超过节点总数)
    k = min(max_community_size, N)
    topk_indices = torch.argsort(rwr_scores, descending=True)[:k].tolist()
    # 确保查询节点一定在内
    if q_idx not in topk_indices:
        if len(topk_indices) >= k:
            topk_indices[-1] = q_idx
        else:
            topk_indices.append(q_idx)
    community_idx = topk_indices

    # 5. 子图打包（与原始函数完全一致）
    community_idx_tensor = torch.tensor(community_idx, dtype=torch.long, device=DEVICE)
    x = node_feat_tensor[community_idx_tensor].detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    node_labels = node_label_tensor[community_idx_tensor].detach()
    local_q_idx = community_idx.index(q_idx)

    mask = torch.isin(all_edge_index[0], community_idx_tensor) & \
           torch.isin(all_edge_index[1], community_idx_tensor)
    edge_index_global = all_edge_index[:, mask]

    if edge_index_global.shape[1] > 0:
        global2local = torch.zeros(N, dtype=torch.long, device=DEVICE)
        global2local[community_idx_tensor] = torch.arange(len(community_idx), device=DEVICE)
        edge_index = global2local[edge_index_global]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    # 社区质量：社区内平均 RWR 得分
    avg_quality = torch.mean(rwr_scores[community_idx_tensor]).item()

    return Data(x=x, edge_index=edge_index, node_labels=node_labels,
                q_id=torch.tensor([local_q_idx], dtype=torch.long, device=DEVICE),
                y=torch.tensor([-1], dtype=torch.long, device=DEVICE),
                community_quality=torch.tensor([avg_quality], dtype=DTYPE, device=DEVICE),
                p_val=None)


def attr_rwr_topk_baseline(q_id, node2idx, idx2node, all_edge_index,
                           ptr, dst, node_feat_tensor, node_label_tensor,
                           min_community_size, max_community_size, q_quantile,
                           restart_prob=0.15, max_rwr_iter=100, rwr_tol=1e-6):
    """
    消融基线 ②：属性感知 RWR (余弦相似度边权重) + top‑k 选择

    与 dap_arwr_single_query_balanced 的区别：
      - 保留了属性驱动的边权重（余弦相似度）
      - 但节点选择退化为简单的 top‑k（无动态阈值、无质量守恒）
      - min_community_size、q_quantile 未使用
    """
    # 1. 查询节点校验
    q_idx = node2idx.get(q_id, -1)
    if q_idx < 0:
        return None
    N = node_feat_tensor.size(0)

    # 2. 计算余弦相似度作为边权重
    q_feat = node_feat_tensor[q_idx].unsqueeze(0)
    cos_sim = torch.nn.functional.cosine_similarity(q_feat, node_feat_tensor, dim=1).clamp(min=0.0)
    cos_sim[q_idx] = 1.0

    deg = ptr[1:] - ptr[:-1]
    src = torch.arange(N, device=DEVICE).repeat_interleave(deg)
    edge_weights = cos_sim[dst]  # 目标节点的余弦相似度

    sum_w = torch.zeros(N, device=DEVICE)
    sum_w.index_add_(0, src, edge_weights)
    sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))
    norm_weights = edge_weights / sum_w[src]

    # 3. 幂迭代 RWR
    p = torch.zeros(N, device=DEVICE)
    p[q_idx] = 1.0
    for _ in range(max_rwr_iter):
        p_prop = torch.zeros(N, device=DEVICE)
        p_prop.index_add_(0, dst, p[src] * norm_weights)
        p_new = (1 - restart_prob) * p_prop
        p_new[q_idx] += restart_prob
        if torch.abs(p_new - p).sum() < rwr_tol:
            p = p_new
            break
        p = p_new
    rwr_scores = p

    # 4. 取 top-k
    k = min(max_community_size, N)
    topk_indices = torch.argsort(rwr_scores, descending=True)[:k].tolist()
    if q_idx not in topk_indices:
        if len(topk_indices) >= k:
            topk_indices[-1] = q_idx
        else:
            topk_indices.append(q_idx)
    community_idx = topk_indices

    # 5. 子图打包（与原始函数一致）
    community_idx_tensor = torch.tensor(community_idx, dtype=torch.long, device=DEVICE)
    x = node_feat_tensor[community_idx_tensor].detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    node_labels = node_label_tensor[community_idx_tensor].detach()
    local_q_idx = community_idx.index(q_idx)

    mask = torch.isin(all_edge_index[0], community_idx_tensor) & \
           torch.isin(all_edge_index[1], community_idx_tensor)
    edge_index_global = all_edge_index[:, mask]

    if edge_index_global.shape[1] > 0:
        global2local = torch.zeros(N, dtype=torch.long, device=DEVICE)
        global2local[community_idx_tensor] = torch.arange(len(community_idx), device=DEVICE)
        edge_index = global2local[edge_index_global]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    avg_quality = torch.mean(rwr_scores[community_idx_tensor]).item()

    return Data(x=x, edge_index=edge_index, node_labels=node_labels,
                q_id=torch.tensor([local_q_idx], dtype=torch.long, device=DEVICE),
                y=torch.tensor([-1], dtype=torch.long, device=DEVICE),
                community_quality=torch.tensor([avg_quality], dtype=DTYPE, device=DEVICE),
                p_val=None)


# ==========================================================
# 🚀 极速两阶段贪心寻路 (支持自动化敏感性测试 + 严格防越界版)
# ==========================================================
def dap_arwr_single_query_balanced(q_id, node2idx, idx2node, all_edge_index,
                                   ptr, dst, node_feat_tensor, node_label_tensor,
                                   min_community_size, max_community_size, q_quantile,
                                   restart_prob=0.15, max_rwr_iter=100, rwr_tol=1e-6):
    """
    基于重启随机游走的异配感知社区搜索 (RWR-HACS) —— 单查询节点版本

    策略：
        1. 计算查询节点与所有节点的余弦相似度（属性亲和度）
        2. 以余弦相似度作为边转移权重，执行重启随机游走，得到稳态得分 rwr_scores
        3. 使用 rwr_scores 进行两阶段社区扩张：
           阶段一：动态阈值保底生长至 min_community_size
           阶段二：全局质量 Q(C) 守恒的贪心扩张至 max_community_size

    参数：
        q_id:                 查询节点 ID (原始 id)
        node2idx:             dict, 原始 id -> 全局索引
        idx2node:             dict, 全局索引 -> 原始 id (本函数未直接使用，保留接口)
        all_edge_index:       全图边索引 (2, E) 用于最终子图提取
        ptr, dst:             CSR 格式邻接信息 (ptr 为行偏移, dst 为列索引)
        node_feat_tensor:     节点特征矩阵 (N, d)
        node_label_tensor:    节点标签向量 (N,)
        min_community_size:   最小社区规模
        max_community_size:   最大社区规模
        q_quantile:           第一阶段动态阈值分位数 (0~1)
        restart_prob:         RWR 重启概率 (默认 0.15)
        max_rwr_iter:         RWR 最大迭代次数
        rwr_tol:              RWR 收敛容差 (L1 范数)

    返回：
        Data 对象，包含子图特征、边、节点标签、查询节点局部索引、
        社区质量 (平均 RWR 得分) 等信息；若查询节点无效则返回 None
    """
    # ==================== 0. 查询节点校验 ====================
    q_idx = node2idx.get(q_id, -1)
    if q_idx < 0:
        return None

    community_nodes = {q_idx}
    N = node_feat_tensor.size(0)

    # ==================== 1. 计算余弦相似度 (保留原有属性度量) ====================
    q_feat = node_feat_tensor[q_idx].unsqueeze(0)
    cos_sim = torch.nn.functional.cosine_similarity(q_feat, node_feat_tensor, dim=1).clamp(min=0.0)
    cos_sim[q_idx] = 1.0

    # ==================== 2. 构建转移矩阵并运行 RWR ====================
    # 2.1 重建边的源节点 (利用 ptr 与 dst)
    deg = ptr[1:] - ptr[:-1]  # 每个节点的出度
    src = torch.arange(N, device=DEVICE).repeat_interleave(deg)

    # 2.2 边权重 = 目标节点的余弦相似度
    edge_weights = cos_sim[dst]

    # 2.3 行归一化，保证每个节点的出边转移概率和为 1
    sum_w = torch.zeros(N, device=DEVICE)
    sum_w.index_add_(0, src, edge_weights)  # sum_w[i] = Σ_{j ∈ N(i)} cos_sim[j]
    sum_w = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))  # 避免除零
    norm_weights = edge_weights / sum_w[src]

    # 2.4 幂迭代求解 RWR 稳态分布
    p = torch.zeros(N, device=DEVICE)
    p[q_idx] = 1.0
    for _ in range(max_rwr_iter):
        # 传播：p_new_partial = p @ M
        p_prop = torch.zeros(N, device=DEVICE)
        p_prop.index_add_(0, dst, p[src] * norm_weights)
        # 重启
        p_new = (1 - restart_prob) * p_prop
        p_new[q_idx] += restart_prob
        # 收敛检查
        if torch.abs(p_new - p).sum() < rwr_tol:
            p = p_new
            break
        p = p_new

    rwr_scores = p  # 最终稳态得分，替代原余弦相似度作为扩张度量

    # ==================== 3. 辅助函数：获取外部一阶邻居 ====================
    def get_external_neighbors(comm_set):
        if not comm_set:
            return torch.tensor([], dtype=torch.long, device=DEVICE)
        comm_tensor = torch.tensor(list(comm_set), dtype=torch.long, device=DEVICE)
        neighs = [dst[ptr[n]: ptr[n + 1]] for n in comm_tensor]
        if not neighs:
            return torch.tensor([], dtype=torch.long, device=DEVICE)
        all_n = torch.unique(torch.cat(neighs))
        return all_n[~torch.isin(all_n, comm_tensor)]

    # ==================== 4. 阶段一：保底生长至 min_community_size ====================
    while len(community_nodes) < min_community_size:
        # 硬上限保护
        if len(community_nodes) >= max_community_size:
            break
        neighbors = get_external_neighbors(community_nodes)
        if len(neighbors) == 0:
            break

        # 使用 RWR 得分
        sims = rwr_scores[neighbors]

        # 动态阈值
        if len(neighbors) <= 3:
            threshold = torch.mean(sims).item()
        else:
            threshold = torch.quantile(sims, q_quantile).item()

        valid_mask = sims >= threshold
        valid_candidates = neighbors[valid_mask]

        if len(valid_candidates) > 0:
            # 按得分降序加入，实时监控最大规模
            valid_sims = sims[valid_mask]
            sorted_idx = torch.argsort(valid_sims, descending=True)
            sorted_candidates = valid_candidates[sorted_idx]
            for candidate in sorted_candidates:
                if len(community_nodes) >= max_community_size:
                    break
                community_nodes.add(candidate.item())
        else:
            # 防止死循环：强制拉取 RWR 得分最高的邻居
            if len(community_nodes) < max_community_size:
                best_idx = torch.argmax(sims)
                community_nodes.add(neighbors[best_idx].item())

    # ==================== 5. 阶段二：质量守恒贪心扩张 ====================
    if min_community_size <= len(community_nodes) < max_community_size:
        current_comm_tensor = torch.tensor(list(community_nodes), dtype=torch.long, device=DEVICE)
        q_base = torch.mean(rwr_scores[current_comm_tensor]).item()  # 初始社区平均 RWR 得分

        while len(community_nodes) < max_community_size:
            neighbors = get_external_neighbors(community_nodes)
            if len(neighbors) == 0:
                break

            sims = rwr_scores[neighbors]
            sorted_indices = torch.argsort(sims, descending=True)

            added_in_this_round = False
            for idx in sorted_indices:
                candidate = neighbors[idx].item()
                candidate_sim = sims[idx].item()
                current_size = len(community_nodes)
                # 模拟加入后的新社区平均质量
                q_new = (q_base * current_size + candidate_sim) / (current_size + 1)
                if q_new >= q_base - EPS:  # 允许极小浮点误差
                    community_nodes.add(candidate)
                    q_base = q_new
                    added_in_this_round = True
                    break  # 每轮只加入一个最优节点，随后重新获取邻居拓扑
                else:
                    break  # 最优候选都拉低质量，直接停止

            if not added_in_this_round:
                break

    # ==================== 6. 子图打包 ====================
    community_idx = list(community_nodes)
    community_idx_tensor = torch.tensor(community_idx, dtype=torch.long, device=DEVICE)

    # 6.1 节点特征与标签
    x = node_feat_tensor[community_idx_tensor].detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    node_labels = node_label_tensor[community_idx_tensor].detach()
    local_q_idx = community_idx.index(q_idx)

    # 6.2 提取社区内部边
    mask = torch.isin(all_edge_index[0], community_idx_tensor) & \
           torch.isin(all_edge_index[1], community_idx_tensor)
    edge_index_global = all_edge_index[:, mask]

    if edge_index_global.shape[1] > 0:
        global2local = torch.zeros(N, dtype=torch.long, device=DEVICE)
        global2local[community_idx_tensor] = torch.arange(len(community_idx), device=DEVICE)
        edge_index = global2local[edge_index_global]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

    # 改为基于余弦相似度：
    # avg_quality = torch.mean(cos_sim[community_idx_tensor]).item()
    # 计算社区内部边的两端节点余弦相似度的均值 (同配性度量 S_avg)
    if edge_index.shape[1] > 0:
        # edge_index 已是局部索引，需要映射回全局索引以获得特征
        global_edge_index = edge_index_global  # 之前用来提取边的全局边索引
        src_global = global_edge_index[0]
        dst_global = global_edge_index[1]
        # 提取两端节点的特征
        feat_src = node_feat_tensor[src_global]
        feat_dst = node_feat_tensor[dst_global]
        # 计算每条边的余弦相似度
        edge_cos = torch.nn.functional.cosine_similarity(feat_src, feat_dst, dim=1)
        edge_cos = torch.clamp(edge_cos, min=0.0)  # 可选，避免负值
        avg_quality = torch.mean(edge_cos).item()
    else:
        avg_quality = 0.0  # 无边时的默认值

    # 6.4 返回 Data 对象 (标签 y 待后续标注流程填充)
    return Data(x=x, edge_index=edge_index, node_labels=node_labels,
                q_id=torch.tensor([local_q_idx], dtype=torch.long, device=DEVICE),
                y=torch.tensor([-1], dtype=torch.long, device=DEVICE),
                community_quality=torch.tensor([avg_quality], dtype=DTYPE, device=DEVICE),
                p_val=None)


def dap_arwr_first_stage_gpu(
        edges_df,
        features_df,
        num_samples,
        min_community_size=5,
        max_community_size=50,
        q_quantile=0.75,
        target_label=1,
        max_attempts=20,
        progress_callback=None,
        restart_prob=0.15,
        max_rwr_iter=100,
        rwr_tol=1e-6,
        confidence_tau=0.90,
        method='hacs',
        label_method='fisher_gmm',
        label_tau=None
):
    """
    批量生成查询社区的顶层函数。

    社区搜索方法：
        hacs
        topo_topk
        attr_topk

    标签方法仅保留三种：

    1. fisher_gmm
       - query 标签属于社区内部已知标签证据
       - 所有 n_L >= 1 的社区都计算 Fisher p-value
       - 所有有效 Fisher score 都参与 GMM 拟合
       - GMM 完成后：
            n_L < 2 -> 最终标签固定为 -1
            n_L >= 2 -> 根据 GMM posterior 输出 0 / -1 / 1
       - 注意：n_L=1 社区参与 GMM calibration，但不形成确定监督标签

    2. query_label
       - 直接使用 query 节点标签作为社区标签

    3. ratio_based
       - 使用社区全部已知节点（包括 query）
       - 计算局部异常比例 k_L / n_L
       - 与全局异常比例 K*/N* 比较
       - 不使用 n_L >= 2 规则
    """

    start_time = time.time()

    all_edge_index, node2idx, idx2node, \
        node_feat_tensor, node_label_tensor, ptr, dst = preprocess_graph_gpu(
            edges_df,
            features_df
        )

    # ==========================================================
    # 全局标签背景统计
    # ==========================================================
    all_labels_np = node_label_tensor.cpu().numpy()

    global_valid_mask = all_labels_np != -1

    N_star = int(
        np.sum(global_valid_mask)
    )

    K_star = int(
        np.sum(
            all_labels_np[global_valid_mask] == 1
        )
    )

    global_anomaly_ratio = (
        K_star / N_star
        if N_star > 0
        else 0.0
    )

    print(
        f"\n🌍 全局背景统计: "
        f"已知标签节点 N*={N_star}, "
        f"异常节点 K*={K_star}, "
        f"异常比例={global_anomaly_ratio:.6f}"
    )

    print(
        f"⚙️ 图采样参数: "
        f"k_min={min_community_size}, "
        f"k_max={max_community_size}, "
        f"q={q_quantile}"
    )

    print(
        f"🔁 RWR 参数: "
        f"restart_prob={restart_prob}, "
        f"max_iter={max_rwr_iter}, "
        f"tol={rwr_tol}"
    )

    method_names = {
        'hacs': 'RWR-HACS (完整方法)',
        'topo_topk': '纯拓扑 RWR + top-k',
        'attr_topk': '属性 RWR + top-k'
    }

    label_names = {
        'fisher_gmm': 'Fisher-GMM',
        'query_label': 'Query-label Transfer',
        'ratio_based': 'Ratio-based Labeling'
    }

    if method not in method_names:
        raise ValueError(
            f"未知 method={method}, "
            f"可选值为 {list(method_names.keys())}"
        )

    if label_method not in label_names:
        raise ValueError(
            f"未知 label_method={label_method}, "
            f"可选值为 {list(label_names.keys())}"
        )

    print(
        f"🧪 社区搜索方法: "
        f"{method_names[method]}"
    )

    print(
        f"🏷️ 标签推断方法: "
        f"{label_names[label_method]}"
    )

    # ==========================================================
    # 查询节点选择
    # 异常 / 正常 query 各半
    # ==========================================================
    anomaly_nodes, normal_nodes = select_nodes_via_label(
        features_df,
        num_samples
    )

    query_node_list = (
        anomaly_nodes + normal_nodes
    )

    data_list = []

    need_alignment = (
        method in ('topo_topk', 'attr_topk')
    )

    # ==========================================================
    # 阶段一：社区搜索
    # ==========================================================
    pbar = tqdm(
        total=len(query_node_list),
        desc="🔍 社区搜索",
        ncols=180,
        bar_format="{l_bar}{bar:60}{r_bar}",
        colour="green"
    )

    for idx, q_id in enumerate(query_node_list):

        try:
            q_id_int = int(q_id)

            # --------------------------------------------------
            # 消融方法：
            # 先运行完整 HACS 获取社区规模 S，
            # 再让 baseline 使用相同 S
            # --------------------------------------------------
            if need_alignment:

                temp_data = dap_arwr_single_query_balanced(
                    q_id_int,
                    node2idx,
                    idx2node,
                    all_edge_index,
                    ptr,
                    dst,
                    node_feat_tensor,
                    node_label_tensor,
                    min_community_size,
                    max_community_size,
                    q_quantile,
                    restart_prob=restart_prob,
                    max_rwr_iter=max_rwr_iter,
                    rwr_tol=rwr_tol
                )

                if temp_data is None:
                    continue

                S = temp_data.x.shape[0]

                if method == 'topo_topk':

                    current_data = topo_rwr_topk_baseline(
                        q_id_int,
                        node2idx,
                        idx2node,
                        all_edge_index,
                        ptr,
                        dst,
                        node_feat_tensor,
                        node_label_tensor,
                        min_community_size,
                        S,
                        q_quantile,
                        restart_prob=restart_prob,
                        max_rwr_iter=max_rwr_iter,
                        rwr_tol=rwr_tol
                    )

                else:

                    current_data = attr_rwr_topk_baseline(
                        q_id_int,
                        node2idx,
                        idx2node,
                        all_edge_index,
                        ptr,
                        dst,
                        node_feat_tensor,
                        node_label_tensor,
                        min_community_size,
                        S,
                        q_quantile,
                        restart_prob=restart_prob,
                        max_rwr_iter=max_rwr_iter,
                        rwr_tol=rwr_tol
                    )

            # --------------------------------------------------
            # 完整 RWR-HACS / RWR-QDCC
            # --------------------------------------------------
            else:

                current_data = dap_arwr_single_query_balanced(
                    q_id_int,
                    node2idx,
                    idx2node,
                    all_edge_index,
                    ptr,
                    dst,
                    node_feat_tensor,
                    node_label_tensor,
                    min_community_size,
                    max_community_size,
                    q_quantile,
                    restart_prob=restart_prob,
                    max_rwr_iter=max_rwr_iter,
                    rwr_tol=rwr_tol
                )

            if current_data is not None:
                data_list.append(
                    current_data
                )

        except Exception:
            pass

        pbar.update(1)

        if (
            (idx + 1) % 10 == 0
            and len(data_list) > 0
        ):

            cur_nodes = sum(
                d.x.shape[0]
                for d in data_list
            )

            pbar.set_postfix({
                '有效图': len(data_list),
                '均节点': (
                    f"{cur_nodes / len(data_list):.1f}"
                )
            })

    pbar.close()

    # ==========================================================
    # 阶段二：标签推断
    # ==========================================================
    print(
        f"\n🧠 正在进行标签推断 "
        f"(方法: {label_method}) ..."
    )

    pos_count = 0
    neg_count = 0
    gray_count = 0

    insufficient_evidence_count = 0
    ambiguous_posterior_count = 0

    # ==========================================================
    # 1. Fisher-GMM
    # ==========================================================
    if label_method == 'fisher_gmm':

        # ------------------------------------------------------
        # 关键：
        #
        # n_L=1 也计算 Fisher，并参与 GMM 拟合。
        #
        # 只有 GMM 完成之后，
        # n_L<2 才强制设为 -1。
        #
        # 这就是我们讨论的 Post-Gate。
        # ------------------------------------------------------

        valid_p_vals = []
        valid_indices = []

        # 保存每个参与 GMM 社区对应的 n_L，
        # 后面输出最终标签时使用。
        valid_n_labeled = []

        for i, d in enumerate(data_list):

            comm_labels = (
                d.node_labels
                .cpu()
                .numpy()
            )

            val_mask = (
                comm_labels != -1
            )

            # n_L：包括 query
            n_labeled = int(
                np.sum(val_mask)
            )

            # k_L：包括 query
            k_labeled = int(
                np.sum(
                    comm_labels[val_mask] == 1
                )
            )

            # ----------------------------------------------
            # 理论上 query 总是已标注，
            # 因此正常情况下不会出现 n_L=0。
            # 这里仍然保留安全处理。
            # ----------------------------------------------
            if n_labeled == 0:

                d.p_val = 2.0

                d.y = torch.tensor(
                    [-1],
                    dtype=torch.long,
                    device=DEVICE
                )

                gray_count += 1
                insufficient_evidence_count += 1

                continue

            # ----------------------------------------------
            # Fisher one-sided enrichment test
            # ----------------------------------------------
            if k_labeled == 0:

                p_val = 1.0

            else:

                p_val = hypergeom.sf(
                    k_labeled - 1,
                    N_star,
                    K_star,
                    n_labeled
                )

            # 防止 log(0)
            p_val = float(
                np.clip(
                    p_val,
                    1e-300,
                    1.0
                )
            )

            d.p_val = p_val

            # ----------------------------------------------
            # 注意：
            # n_L=1 的社区也进入这里！
            # 它们参与 GMM calibration。
            # ----------------------------------------------
            valid_p_vals.append(
                p_val
            )

            valid_indices.append(
                i
            )

            valid_n_labeled.append(
                n_labeled
            )

        # ------------------------------------------------------
        # GMM
        # ------------------------------------------------------
        if len(valid_p_vals) > 10:

            p_vals_np = np.array(
                valid_p_vals,
                dtype=np.float64
            )

            p_vals_np = np.clip(
                p_vals_np,
                a_min=1e-300,
                a_max=1.0
            )

            scores = (
                -np.log(p_vals_np)
            ).reshape(
                -1,
                1
            )

            # --------------------------------------------------
            # 保留你原来 sklearn GMM 的实现，
            # 不使用自写 GPU GMM。
            # --------------------------------------------------
            gmm = GaussianMixture(
                n_components=2,
                covariance_type='full',
                random_state=SEED
            )

            gmm.fit(
                scores
            )

            means = (
                gmm.means_
                .flatten()
            )

            normal_comp_idx = int(
                np.argmin(means)
            )

            anomaly_comp_idx = int(
                np.argmax(means)
            )

            post_probs = (
                gmm.predict_proba(
                    scores
                )
            )

            prob_anomaly = (
                post_probs[
                    :,
                    anomaly_comp_idx
                ]
            )

            prob_normal = (
                post_probs[
                    :,
                    normal_comp_idx
                ]
            )

            print(
                "\n📈 Fisher-GMM 分量:"
            )

            print(
                f"   Normal mean     = "
                f"{means[normal_comp_idx]:.6f}"
            )

            print(
                f"   Anomalous mean  = "
                f"{means[anomaly_comp_idx]:.6f}"
            )

            # --------------------------------------------------
            # 最终标签输出
            # --------------------------------------------------
            for (
                list_idx,
                n_labeled,
                p_anom,
                p_norm
            ) in zip(
                valid_indices,
                valid_n_labeled,
                prob_anomaly,
                prob_normal
            ):

                # ==========================================
                # Post-Gate:
                #
                # n_L=1 已经参与完 GMM，
                # 但最终不能形成确定社区标签。
                # ==========================================
                if n_labeled < 2:

                    data_list[list_idx].y = (
                        torch.tensor(
                            [-1],
                            dtype=torch.long,
                            device=DEVICE
                        )
                    )

                    gray_count += 1

                    insufficient_evidence_count += 1

                    continue

                # ==========================================
                # n_L >= 2:
                # 正常按 GMM posterior 判断
                # ==========================================
                if p_anom >= confidence_tau:

                    data_list[list_idx].y = (
                        torch.tensor(
                            [1],
                            dtype=torch.long,
                            device=DEVICE
                        )
                    )

                    pos_count += 1

                elif p_norm >= confidence_tau:

                    data_list[list_idx].y = (
                        torch.tensor(
                            [0],
                            dtype=torch.long,
                            device=DEVICE
                        )
                    )

                    neg_count += 1

                else:

                    data_list[list_idx].y = (
                        torch.tensor(
                            [-1],
                            dtype=torch.long,
                            device=DEVICE
                        )
                    )

                    gray_count += 1

                    ambiguous_posterior_count += 1

        # ------------------------------------------------------
        # 有效样本太少，无法拟合 GMM
        # ------------------------------------------------------
        else:

            for list_idx in valid_indices:

                data_list[list_idx].y = (
                    torch.tensor(
                        [-1],
                        dtype=torch.long,
                        device=DEVICE
                    )
                )

                gray_count += 1

                ambiguous_posterior_count += 1

    # ==========================================================
    # 2. Query-label Transfer
    # ==========================================================
    elif label_method == 'query_label':

        # ------------------------------------------------------
        # baseline：
        # 整个社区直接继承 query 标签
        #
        # y_C = y_q
        # ------------------------------------------------------
        for d in data_list:

            q_local_idx = int(
                d.q_id.item()
            )

            query_label = int(
                d.node_labels[
                    q_local_idx
                ].item()
            )

            if query_label == 1:

                d.y = torch.tensor(
                    [1],
                    dtype=torch.long,
                    device=DEVICE
                )

                # 仅兼容已有 Data 字段
                d.p_val = 0.0

                pos_count += 1

            elif query_label == 0:

                d.y = torch.tensor(
                    [0],
                    dtype=torch.long,
                    device=DEVICE
                )

                d.p_val = 1.0

                neg_count += 1

            else:

                d.y = torch.tensor(
                    [-1],
                    dtype=torch.long,
                    device=DEVICE
                )

                d.p_val = 2.0

                gray_count += 1

    # ==========================================================
    # 3. Ratio-based Labeling
    # ==========================================================
    elif label_method == 'ratio_based':

        # ------------------------------------------------------
        # baseline：
        #
        # local_ratio = k_L / n_L
        #
        # 默认：
        # tau = K* / N*
        #
        # local_ratio > tau -> anomalous
        # otherwise         -> normal
        #
        # 注意：
        # 这里不使用 n_L>=2 规则。
        # ------------------------------------------------------

        tau = (
            label_tau
            if label_tau is not None
            else (
                K_star / N_star
                if N_star > 0
                else 0.5
            )
        )

        print(
            f"\n📊 Ratio-based 阈值:"
            f" tau={tau:.6f}"
        )

        for d in data_list:

            comm_labels = (
                d.node_labels
                .cpu()
                .numpy()
            )

            val_mask = (
                comm_labels != -1
            )

            n_labeled = int(
                np.sum(
                    val_mask
                )
            )

            k_labeled = int(
                np.sum(
                    comm_labels[
                        val_mask
                    ] == 1
                )
            )

            if n_labeled == 0:

                d.y = torch.tensor(
                    [-1],
                    dtype=torch.long,
                    device=DEVICE
                )

                d.p_val = 2.0

                gray_count += 1

                continue

            local_ratio = (
                k_labeled
                / n_labeled
            )

            # 只是为了兼容已有可视化代码，
            # Ratio-based 本身并不使用 p-value。
            d.p_val = float(
                1.0 - local_ratio
            )

            if local_ratio > tau:

                d.y = torch.tensor(
                    [1],
                    dtype=torch.long,
                    device=DEVICE
                )

                pos_count += 1

            else:

                d.y = torch.tensor(
                    [0],
                    dtype=torch.long,
                    device=DEVICE
                )

                neg_count += 1

    else:

        raise ValueError(
            f"未知 label_method: "
            f"{label_method}"
        )

    # ==========================================================
    # 统计信息
    # ==========================================================
    total_nodes = sum(
        d.x.shape[0]
        for d in data_list
    )

    total_edges = sum(
        d.edge_index.shape[1]
        for d in data_list
    )

    avg_nodes = (
        total_nodes / len(data_list)
        if len(data_list) > 0
        else 0.0
    )

    avg_edges = (
        total_edges / len(data_list)
        if len(data_list) > 0
        else 0.0
    )

    avg_quality = (
        sum(
            d.community_quality.item()
            for d in data_list
        ) / len(data_list)
        if len(data_list) > 0
        else 0.0
    )

    print(
        f"\n✅ 生成完毕 | "
        f"耗时: {time.time() - start_time:.2f}s | "
        f"总社区数: {len(data_list)}"
    )

    print(
        f"📊 标签分布 | "
        f"正常: {neg_count} | "
        f"异常: {pos_count} | "
        f"监督样本(0/1): {pos_count + neg_count} | "
        f"灰区样本(-1): {gray_count}"
    )

    if label_method == 'fisher_gmm':

        print(
            "🧾 Fisher-GMM 灰区来源 | "
            f"n_L<2证据不足: "
            f"{insufficient_evidence_count} | "
            f"GMM后验模糊: "
            f"{ambiguous_posterior_count}"
        )

    print(
        f"📏 规模统计 | "
        f"平均节点数: {avg_nodes:.1f} | "
        f"平均边数: {avg_edges:.1f} | "
        f"纯度 (S_avg): {avg_quality:.4f}\n"
    )

    return data_list