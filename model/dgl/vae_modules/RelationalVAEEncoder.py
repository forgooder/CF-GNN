import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn  # 必须导入，用于 update_all 的内置聚合

class RelationalVAEEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_rels, num_bases, device):
        super(RelationalVAEEncoder, self).__init__()

        self.device = device
        self.num_rels = num_rels
        self.num_bases = num_bases

        self.in_channels = in_channels # 输入特征维度 (建议设为 8，适配 GraIL 的 hop=3 输出)
        self.out_channels = out_channels    # VAE 隐变量维度 (建议设为 64，其中 32 用于 alpha，32 用于 beta)

        # 🌟 新增：升维垫片，将 8 维输入线性映射到 64 维，
        self.input_projection = nn.Linear(in_channels, out_channels).to(device)
        
        # --- 基础矩阵设计 (同 GraIL Basis 逻辑) ---
        # weight: [num_bases, in_dim, 2 * out_dim]
        self.weight = nn.Parameter(torch.Tensor(num_bases, out_channels, 2 * out_channels))
        # w_comp: [num_rels, num_bases]
        self.w_comp = nn.Parameter(torch.Tensor(num_rels, num_bases))

        # 负责将隐变量 z [32维] 还原回原始特征维度 [108维]
        self.reconstruct_decoder = nn.Linear(out_channels, in_channels)
        
        # 潜在分布映射层
        self.lin_mu = nn.Linear(2 * out_channels, out_channels)
        self.lin_logstd = nn.Linear(2 * out_channels, out_channels)

        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h_input):
        """
        g: DGLGraph 子图
        h_input: 从 GraIL 截获的特征 [N, in_channels]
        """

        h_proj = self.input_projection(h_input)
        g.ndata['vae_h'] = h_proj
        
        #batch_in_dim = h_input.shape[1]
        # 1. 生成关系专属权重 W_r [num_rels, in_dim, 2 * out_dim]
        weight_flatten = self.weight.view(self.num_bases, -1)

        rel_weights = torch.matmul(self.w_comp, weight_flatten).view(
            self.num_rels, self.out_channels, 2*self.out_channels
        )

        # 2. 定义 DGL 消息函数 (使用隔离的特征键名 'vae_h' 和 'vae_m')
        def msg_func(edges):
            # 根据边的类型索引对应的 W_r
            w = rel_weights.index_select(0, edges.data['type'])
            # 矩阵乘法计算消息: [batch_e, 1, in_dim] * [batch_e, in_dim, out_dim]
            # 这里使用了 src['vae_h'] 避免覆盖 GraIL 的 'h'
            msg = torch.bmm(edges.src['vae_h'].unsqueeze(1), w).squeeze(1)
            return {'vae_m': msg}

        # 3. 消息传递与聚合
        # 将输入特征存入 vae_h，保护原始数据的干净

        
        # 使用 DGL 内置的 sum 函数聚合 vae_m 到 vae_h_agg
        g.update_all(msg_func, fn.sum(msg='vae_m', out='vae_h_agg'))
        
        h_agg = g.ndata['vae_h_agg']
        
        # 4. 归一化处理
        # 计算入度并进行倒数缩放，防止节点邻居多导致特征爆炸
        degs = g.in_degrees().float().clamp(min=1).to(self.device).unsqueeze(1)
        h_agg = h_agg / degs

        # 5. 生成 VAE 潜在分布参数 (均值和标准差对数)
        mu = self.lin_mu(h_agg)
        logstd = self.lin_logstd(h_agg)
        
        # 限制 logstd 范围防止早期随机采样把 mask 分支推到饱和区。
        logstd = torch.clamp(logstd, min=-5.0, max=5.0)
        std = torch.exp(logstd)
        
        # 训练时重参数化采样；评估时使用均值，保证验证/测试可复现。
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        self.x_est = self.reconstruct_decoder(z) 
        # 清理 DGL 图中的临时 VAE 特征，释放显存并保持图对象整洁
        if 'vae_h' in g.ndata: g.ndata.pop('vae_h')
        if 'vae_h_agg' in g.ndata: g.ndata.pop('vae_h_agg')
        
        return z, mu, logstd
