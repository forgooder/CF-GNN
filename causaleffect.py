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

def _edge_mask_from_latent(latent, decoder, data, device):
    u, v = data.edges()
    u, v = u.to(device), v.to(device)
    z_edge = decoder(latent)
    mask_logits = torch.sum(z_edge[u] * z_edge[v], dim=1, keepdim=True)
    mask = torch.sigmoid(mask_logits)
    return mask, mask_logits


def joint_uncond(alpha, beta, data, rel_labels, causal_decoder, shortcut_decoder,
                 classifier, device, k, compute_cmi=False):
    # 0) 脱壳与设备同步
    while isinstance(rel_labels, (tuple, list)):
        rel_labels = rel_labels[0]
    
    alpha = alpha.to(device)
    beta = beta.to(device)
    rel_labels = rel_labels.to(device)

    # 1) 图级 Readout (得到 Alpha_G, Beta_G)
    data.ndata['tmp_alpha'] = alpha
    data.ndata['tmp_beta'] = beta
    graph_alpha = dgl.mean_nodes(data, 'tmp_alpha') 
    graph_beta = dgl.mean_nodes(data, 'tmp_beta')
    data.ndata.pop('tmp_alpha')
    data.ndata.pop('tmp_beta')

    # 2) 标签处理
    y_float = rel_labels.float().view(-1, 1)

    # 3) 双分支掩码：alpha 负责 causal graph，beta 负责 shortcut graph。
    mask_alpha, logits_alpha = _edge_mask_from_latent(alpha, causal_decoder, data, device)
    mask_beta, logits_beta = _edge_mask_from_latent(beta, shortcut_decoder, data, device)

    # 4) 同一个 GraIL predictor 分别在 causal/shortcut graph 上打分。
    score_alpha = classifier(
        data=data,
        rel_labels=rel_labels,
        edge_weight=mask_alpha,
        edge_mask_logits=logits_alpha
    )
    score_beta = classifier(
        data=data,
        rel_labels=rel_labels,
        edge_weight=mask_beta,
        edge_mask_logits=logits_beta
    )

    # 5) causal effect 是可反传的分数差，直接进入 Trainer 的 ranking loss。
    causal_effect = score_alpha - score_beta

    # 6) 可选 CMI；主训练里会用正负样本拼接后的 y 再计算一次。
    if compute_cmi:
        cmi = calculate_conditional_MI(
            graph_alpha, y_float, graph_beta
        )
    else:
        cmi = torch.tensor(0.0, device=device)

    return {
        "causal_score": score_alpha,
        "shortcut_score": score_beta,
        "causal_effect": causal_effect,
        "mask_alpha": mask_alpha,
        "mask_beta": mask_beta,
        "cmi": cmi,
    }

# 保留 no-op 函数避免 Trainer 报错
def clear_masks(model): return
def set_masks(model, edgemask): return
