import abc
import torch.nn as nn
import torch
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(nn.Module):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, node_batch):
        """
        node_batch.mailbox 包含了 msg_func 发送过来的所有数据
        'msg':      [batch_nodes, num_neighbors, out_dim] -> 邻居消息
        'alpha':    [batch_nodes, num_neighbors, 1]       -> 注意力权重 (包含因果权重)
        'curr_emb': [batch_nodes, num_neighbors, out_dim] -> 当前节点的自环特征
        """
        # 1. 提取邮件信息
        mailbox_msg = node_batch.mailbox['msg']      # 邻居发来的 RGCN 变换后的特征
        mailbox_alpha = node_batch.mailbox['alpha']  # 软权重 (已经包含了因果掩码)
        mailbox_curr = node_batch.mailbox['curr_emb'] # 节点的自环变换特征

        # 2. 加权聚合邻居消息 (维度相乘: [B, N, D] * [B, N, 1])
        # 这里执行了真正的因果过滤聚合
        neigh_msg = torch.sum(mailbox_msg * mailbox_alpha, dim=1)

        # 3. 处理自环特征
        # 自环消息在 mailbox 中通常对每个入边都存了一份，我们只需要取其中一份即可
        # (或者在 msg_func 里只给自环发一份，但 DGL 默认对每条边触发一次 msg_func)
        curr_emb = mailbox_curr[:, 0, :] 

        # 4. 最终融合：邻居聚合信息 + 节点自身信息
        new_emb = neigh_msg + curr_emb

        # 返回字典，对应 RGCNLayer 里的 node_repr = g.ndata['h']
        return {'h': new_emb}


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb
