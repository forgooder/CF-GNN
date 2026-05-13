import torch
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

    # 1) alpha 负责生成 causal graph；beta 只作为 CMI 条件变量。
    mask_alpha, logits_alpha = _edge_mask_from_latent(alpha, causal_decoder, data, device)

    # 2) GraIL predictor 只在 causal graph 上打分。
    score_alpha = classifier(
        data=data,
        rel_labels=rel_labels,
        edge_weight=mask_alpha,
        edge_mask_logits=logits_alpha
    )

    # 3) 可选 CMI；主训练会用正负样本拼接后的 binary y 计算。
    if compute_cmi:
        data.ndata['tmp_alpha'] = alpha
        data.ndata['tmp_beta'] = beta
        graph_alpha = dgl.mean_nodes(data, 'tmp_alpha')
        graph_beta = dgl.mean_nodes(data, 'tmp_beta')
        data.ndata.pop('tmp_alpha')
        data.ndata.pop('tmp_beta')
        y_float = rel_labels.float().view(-1, 1)
        cmi = calculate_conditional_MI(
            graph_alpha, y_float, graph_beta
        )
    else:
        cmi = torch.tensor(0.0, device=device)

    return {
        "causal_score": score_alpha,
        "causal_effect": cmi,
        "mask_alpha": mask_alpha,
        "cmi": cmi,
    }

# 保留 no-op 函数避免 Trainer 报错
def clear_masks(model): return
def set_masks(model, edgemask): return
