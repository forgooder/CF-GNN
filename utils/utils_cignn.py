import torch
from torch.utils.data import random_split, Subset, DataLoader
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


class GraphData(object):
    def __init__(self, x, edge_index, y):
        self.x = x
        self.edge_index = edge_index
        self.y = y

    def __len__(self):
        # 返回节点数，供某些长度相关操作使用
        return self.x.size(0)


class GraphBatch(object):
    def __init__(self, x, edge_index, batch, y):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.y = y

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        if isinstance(self.y, torch.Tensor):
            self.y = self.y.to(device)
        return self


def dense_to_sparse(adj):
    """
    替代 torch_geometric.utils.dense_to_sparse
    adj: [N, N]
    return:
      edge_index: [2, E]
      edge_weight: [E]
    """
    idx = adj.nonzero().t().contiguous()
    vals = adj[idx[0], idx[1]]
    return idx.long(), vals


def collate_graphs(graph_list):
    """
    把若干 GraphData / PyG Data 合并成一个 GraphBatch，模拟 PyG 的 Batch 行为。
    关键点：把每个图的 y 都压成标量，最终 batch.y 形状是 [num_graph]。
    """
    if len(graph_list) == 0:
        return None

    xs = []
    edge_indices = []
    batch_index = []
    ys = []

    node_offset = 0
    for g_id, g in enumerate(graph_list):
        x = g.x
        edge_index = g.edge_index
        y = g.y

        # 节点数
        num_nodes = x.size(0)
        xs.append(x)

        # 偏移 edge_index
        edge_indices.append(edge_index + node_offset)
        batch_index.append(torch.full((num_nodes,), g_id, dtype=torch.long))

        # 图级标签：统一成标量，再堆叠成 [num_graph]
        if isinstance(y, torch.Tensor):
            # 兼容 y 可能是标量 / [1] / [B,1] 等形状
            y = y.view(-1)[0]
        else:
            y = torch.tensor(y)
        ys.append(y)

        node_offset += num_nodes

    x_cat = torch.cat(xs, dim=0)                    # [N_total, F]
    edge_index_cat = torch.cat(edge_indices, dim=1) # [2, E_total]
    batch_cat = torch.cat(batch_index, dim=0)       # [N_total]
    y_cat = torch.stack(ys, dim=0).long()           # [num_graph]

    return GraphBatch(x_cat, edge_index_cat, batch_cat, y_cat)


eps = 1e-8


def load_dataset(graph):
    print('loading data')
    num_graphs = graph["label"].size
    label1 = graph["label"]
    label = np.append(label1, label1)
    data_list = []
    for i in range(num_graphs):
        node_features = torch.FloatTensor(graph["graph_struct"][0][i][1])
        # node_features = (node_features - torch.mean(node_features))/(torch.std(node_features)+eps)
        tepk = node_features.reshape(-1, 1)
        tepk, indices = torch.sort(abs(tepk), dim=0, descending=True)
        mk = tepk[int(math.pow(node_features.shape[0], 2) / 20 * 2)]
        adj = torch.Tensor(np.where(node_features > mk, 1, 0))
        edge_index, _ = dense_to_sparse(adj)
        data_example = GraphData(x=node_features, edge_index=edge_index, y=label[i])
        data_list.append(data_example)

    return data_list


def BAnotif_load_dataset(graph):
    print('loading data')
    data_list = []
    for i in range(len(graph)):
        g, label = graph[i]
        adj = g.adjacency_matrix(transpose=True)._indices()
        node_features = g.ndata['feat']
        # node_features = (node_features - torch.mean(node_features))/(torch.std(node_features)+eps)
        data_example = GraphData(x=node_features, edge_index=adj, y=label[0])
        data_list.append(data_example)

    return data_list


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=None):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, 'supplement'):
        assert 'split_indices' in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement['split_indices']
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        eval_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, eval_indices)
        test = Subset(dataset, test_indices)
    else:
        assert data_split_ratio is not None, "split ratio"
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(
            dataset,
            lengths=[num_train, num_eval, num_test],
            generator=torch.Generator().manual_seed(seed)
        )
    num_eval = len(eval)
    num_test = len(test)
    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    dataloader['eval'] = DataLoader(eval, batch_size=num_eval, shuffle=False, collate_fn=collate_graphs)
    dataloader['test'] = DataLoader(test, batch_size=num_test, shuffle=False, collate_fn=collate_graphs)
    return dataloader


# ===== 以下是 MI / 熵 相关函数，已经兼容 torch==1.4（使用 symeig）=====

eps = 1e-8  # 如果前面已经定义了 eps，这里可以省略或保持一致


##############################################
#  距离 / 核函数 / 矩阵 Renyi 熵 & 互信息
##############################################

def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D]
    return: [N, N]，欧氏距离矩阵
    """
    x = x.view(x.size(0), -1)
    xx = (x ** 2).sum(dim=1, keepdim=True)          # [N, 1]
    dist = xx + xx.t() - 2.0 * x @ x.t()            # [N, N]
    dist = torch.clamp(dist, min=0.0)
    dist = torch.sqrt(dist + 1e-12)
    return dist


def calculate_sigma(Z: torch.Tensor) -> float:
    """
    计算高斯核带宽 sigma，使用最近邻距离的均值。
    """
    if Z.dim() == 1:
        Z = Z.unsqueeze(1)

    Z_np = Z.detach().cpu().numpy()
    k = squareform(pdist(Z_np, "euclidean"))  # [N, N]
    if k.shape[0] <= 1:
        return 0.1

    k_sorted = np.sort(k, axis=1)
    # 跳过第 0 个 self-distance=0，避免小 batch 下 sigma 被系统性压小。
    topk = k_sorted[:, 1:min(11, k_sorted.shape[1])]
    if topk.size == 0:
        return 0.1
    sigma = float(np.mean(topk))

    if sigma < 0.1:
        sigma = 0.1
    return sigma


def calculate_gram_mat(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    根据 matrix-based Renyi 熵构造 Gaussian 核矩阵：
      K_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))
    然后做 trace 归一化：K = K / trace(K)
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)

    # Keep this path differentiable so MI/CMI losses can train alpha/beta.
    x_flat = x.view(x.size(0), -1)
    xx = (x_flat ** 2).sum(dim=1, keepdim=True)
    sq_dist = xx + xx.t() - 2.0 * x_flat @ x_flat.t()
    sq_dist = torch.clamp(sq_dist, min=0.0)

    K = torch.exp(-sq_dist / (2.0 * (sigma ** 2) + 1e-12))

    trace = torch.trace(K)
    if trace.item() > 0:
        K = K / trace
    else:
        K = K / (K.sum() + 1e-12)

    K = 0.5 * (K + K.t())
    K = torch.clamp(K, min=0.0)
    return K


def _matrix_renyi_entropy(K: torch.Tensor, alpha: float = 1.01) -> torch.Tensor:
    """
    Matrix-based Renyi's alpha-order entropy:
        H_alpha(X) = 1/(1-alpha) * log2( sum_i lambda_i^alpha )
    其中 lambda_i 是 K 的特征值，K 已经 trace-normalized。
    在 torch1.4 下用 symeig 代替 torch.linalg.eigh。
    """
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "eigvalsh"):
        eigvals = torch.linalg.eigvalsh(K)
    else:
        # torch 1.4 中 symeig 的特征值梯度路径需要 eigenvectors=True。
        eigvals, _ = torch.symeig(K, eigenvectors=True)
    eigvals = torch.clamp(eigvals, min=0.0)
    eigvals = eigvals / (eigvals.sum() + 1e-12)

    if alpha == 1.0:
        ent = -torch.sum(eigvals * torch.log2(eigvals + 1e-12))
    else:
        ent = 1.0 / (1.0 - alpha) * torch.log2(
            torch.sum(eigvals ** alpha) + 1e-12
        )
    return ent


def reyi_entropy(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    对数据 x 计算矩阵 Renyi 熵。
    """
    K = calculate_gram_mat(x, sigma)
    H = _matrix_renyi_entropy(K)
    return H


def joint_entropy(x: torch.Tensor, y: torch.Tensor,
                  s_x: float, s_y: float) -> torch.Tensor:
    """
    联合熵 H(X, Y)：先拼接 [x, y]，再算一遍熵。
    """
    xy = torch.cat([x, y], dim=-1)
    sigma_xy = (s_x + s_y) / 2.0
    return reyi_entropy(xy, sigma_xy)


def joint_entropy3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                   s_x: float, s_y: float, s_z: float) -> torch.Tensor:
    """
    三元联合熵 H(X, Y, Z)
    """
    xyz = torch.cat([x, y, z], dim=-1)
    sigma_xyz = (s_x + s_y + s_z) / 3.0
    return reyi_entropy(xyz, sigma_xyz)


def calculate_MI(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    互信息 I(X; Y) = H(X) + H(Y) - H(X, Y)
    """
    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)

    Hx = reyi_entropy(x, s_x)
    Hy = reyi_entropy(y, s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)

    MI = Hx + Hy - Hxy
    return MI


def calculate_conditional_MI(x: torch.Tensor,
                             y: torch.Tensor,
                             z: torch.Tensor) -> torch.Tensor:
    """
    条件互信息 I(X; Y | Z)
      = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
    """
    s_x = calculate_sigma(x)
    s_y = calculate_sigma(y)
    s_z = calculate_sigma(z)

    Hxz = joint_entropy(x, z, s_x, s_z)
    Hyz = joint_entropy(y, z, s_y, s_z)
    Hz = reyi_entropy(z, s_z)
    Hxyz = joint_entropy3(x, y, z, s_x, s_y, s_z)

    CMI = Hxz + Hyz - Hz - Hxyz
    return CMI


def calculate_single_TC(x: torch.Tensor) -> torch.Tensor:
    """
    Total Correlation for a single multivariate X:
      TC(X) = sum_i H(X_i) - H(X)
    这里按列拆变量 X_i。
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)

    d = x.size(1)
    sigmas = []
    H_marg = 0.0
    for i in range(d):
        xi = x[:, i:i + 1]
        s_i = calculate_sigma(xi)
        sigmas.append(s_i)
        H_marg = H_marg + reyi_entropy(xi, s_i)

    s_all = float(np.mean(sigmas))
    H_all = reyi_entropy(x, s_all)
    TC = H_marg - H_all
    return TC


def calculate_Condition_TC(x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
    """
    条件 total correlation: TC(X | Y)
    简单近似为 TC(X) - I(X; Y)。
    """
    TC_x = calculate_single_TC(x)
    MI_xy = calculate_MI(x, y)
    return TC_x - MI_xy


def calculate_TC(x: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
    """
    TC(X, Y) 的近似：TC([X, Y])。
    """
    xy = torch.cat([x, y], dim=-1)
    return calculate_single_TC(xy)


def MI_Est(discriminator,
           embeddings: torch.Tensor,
           positive: torch.Tensor,
           batch_size: int) -> torch.Tensor:
    """
    基于判别器的 InfoNCE 风格互信息估计器。
    """
    device = embeddings.device
    N = embeddings.size(0)

    pos_i = positive[:, 0]
    pos_j = positive[:, 1]
    z_i = embeddings[pos_i]  # [B, D]
    z_j = embeddings[pos_j]  # [B, D]

    neg_j_idx = torch.randint(low=0, high=N, size=(batch_size,), device=device)
    z_neg = embeddings[neg_j_idx]

    pos_score = discriminator(z_i, z_j)       # [B]
    neg_score = discriminator(z_i, z_neg)     # [B]

    loss = -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-12))
    return loss
