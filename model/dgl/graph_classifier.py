from .rgcn_model import RGCN
import dgl
from dgl import mean_nodes
import torch.nn as nn
import torch
from causaleffect import joint_uncond  
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def predict(self, data, device):
        """
        🌟 核心：推理专用接口。复刻 Trainer.py 中的因果推理流程。
        """
        self.eval()
        with torch.no_grad():
            # 兼容性解析数据
            if isinstance(data, (tuple, list)):
                g, rel_labels = data[0], data[1]
            else:
                g, rel_labels = data, data.edata['type'][0]

            g = g.to(device)
            rel_labels = rel_labels.to(device)
            use_cignn_mask = (
                getattr(self.params, 'use_cignn_mask', False)
                and getattr(self.params, 'cignn_mask_mode', 'none') != 'none'
            )
            if not use_cignn_mask:
                return self.forward(data=g, rel_labels=rel_labels)

            node_feats = g.ndata['feat']

            # VAE 编码分离 alpha/beta
            z, mu, _ = self.gnn.encoder(g, node_feats)
            mid = z.shape[1] // 2
            alpha = z[:, :mid]
            beta = z[:, mid:]

            # 调用因果干预函数获取最终 score
            # k=100 表示测试时使用稳定的干预强度
            _, scores = joint_uncond(alpha, beta, g, rel_labels, self.gnn.casual_decoder, self, device, k= 50)
            return scores

    def forward(self, data, *args, **kwargs):
        # 兼容性处理：从 args 或 kwargs 中提取 rel_labels 和 edge_weight
        edge_weight = kwargs.get('edge_weight', None)
        edge_mask_logits = kwargs.get('edge_mask_logits', None)
        rel_labels = kwargs.get('rel_labels', None)

        # 处理位置参数
        if rel_labels is None and len(args) > 0:
            rel_labels = args[0]
        if edge_weight is None and len(args) > 1:
            edge_weight = args[1]

        # 结构化解包 data
        if isinstance(data, (tuple, list)):
            g = data[0]
            if rel_labels is None and len(data) > 1:
                rel_labels = data[1]
        else:
            g = data

        # 确保 rel_labels 脱壳
        while isinstance(rel_labels, (tuple, list)):
            rel_labels = rel_labels[0]

        # --- 核心计算逻辑 ---
        node_repr_all = self.gnn(g, edge_weight=edge_weight, edge_mask_logits=edge_mask_logits) 
        node_repr_flat = node_repr_all.view(node_repr_all.shape[0], -1)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        
        head_embs = node_repr_flat[head_ids]
        tail_embs = node_repr_flat[tail_ids]

        g.ndata['tmp_h'] = node_repr_flat
        g_out = dgl.mean_nodes(g, 'tmp_h')
        g.ndata.pop('tmp_h')

        rel_embs = self.rel_emb(rel_labels)
        
        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out, head_embs, tail_embs, rel_embs], dim=1)
        else:
            g_rep = torch.cat([g_out, rel_embs], dim=1)

        return self.fc_layer(g_rep)
