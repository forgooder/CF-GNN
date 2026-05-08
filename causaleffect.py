import numpy as np
import torch
import torch.nn as nn
import dgl
from utils.utils_cignn import calculate_conditional_MI

#######################################################
# 适配 DGL 的 Readout 逻辑
#######################################################

def dgl_global_mean_pool(graph, feat):
    """
    替代原来的 global_mean_pool，直接使用 DGL 内置算子
    graph: DGLGraph (Batch 后的)
    feat: [N, F] 节点特征
    """
    return dgl.mean_nodes(graph, feat)

def dgl_global_add_pool(graph, feat):
    return dgl.sum_nodes(graph, feat)

#######################################################
# 条件互信息 + 因果效应计算 (DGL 适配版)
#######################################################

import dgl
import torch
from utils.utils_cignn import calculate_conditional_MI

def joint_uncond(alpha, beta, data, rel_labels, casual_decoder, classifier, device, k, compute_cmi=False):
    # 0) 脱壳与设备同步
    while isinstance(rel_labels, (tuple, list)):
        rel_labels = rel_labels[0]
    
    alpha = alpha.to(device)
    beta = beta.to(device)
    rel_labels = rel_labels.to(device)

    # 1) ★ 获取边信息 (修复 NameError 的关键) ★
    u, v = data.edges() 
    u, v = u.to(device), v.to(device)

    # 2) 图级 Readout (得到 Alpha_G, Beta_G)
    data.ndata['tmp_alpha'] = alpha
    data.ndata['tmp_beta'] = beta
    graph_alpha = dgl.mean_nodes(data, 'tmp_alpha') 
    graph_beta = dgl.mean_nodes(data, 'tmp_beta')
    data.ndata.pop('tmp_alpha')
    data.ndata.pop('tmp_beta')

    # 3) 标签处理
    y_float = rel_labels.float().view(-1, 1)

    # 4) 因果掩码生成
    z_casual = casual_decoder(alpha) # [N, d_hidden]
    # 计算边两端节点的点积得到边权重
    aedge_logits = torch.sum(z_casual[u] * z_casual[v], dim=1, keepdim=True)
    aedge = torch.sigmoid(aedge_logits) # [E, 1]

    # 5) 执行分类器 (因果干预后的 Forward)
    logits = classifier(
        data=data,
        rel_labels=rel_labels,
        edge_weight=aedge,
        edge_mask_logits=aedge_logits
    )

    # 6) 计算条件互信息 (CMI)
    if compute_cmi:
        causal_effect = calculate_conditional_MI(
            graph_alpha, y_float, graph_beta
        )
    else:
        causal_effect = torch.tensor(0.0, device=device)

    return causal_effect, logits

# 保留 no-op 函数避免 Trainer 报错
def clear_masks(model): return
def set_masks(model, edgemask): return
