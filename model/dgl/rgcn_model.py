import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from .layers import RGCNBasisLayer as RGCNLayer
from .aggregators import SumAggregator, MLPAggregator, GRUAggregator
from .vae_modules.RelationalVAEEncoder import RelationalVAEEncoder

# 假设 RelationalVAEEncoder 定义在同一目录下或已导入
# from .vae_modules import RelationalVAEEncoder 

class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        # --- 基础参数初始化 ---
        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim # 输入特征维度 (通常是 GraIL 首层输出的维度，建议设为 8)
        self.emb_dim = params.emb_dim # RGCN 内部表示维度 (建议设为 64)
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn
        self.device = params.device

        self.latent_dim = getattr(params, 'latent_dim', 64) # VAE 隐变量维度（建议设为 64，其中 32 用于 alpha，32 用于 beta）
        self.encoder = RelationalVAEEncoder(
            in_channels=self.inp_dim, 
            out_channels=self.latent_dim, 
            num_rels=params.aug_num_rels, 
            num_bases=params.num_bases, 
            device=params.device
        )

        self.decoder = self._internal_decoder_wrapper

        self.casual_decoder = nn.Sequential(
            nn.Linear(params.latent_dim // 2, params.emb_dim),
            nn.ReLU(),
            nn.Linear(params.emb_dim, params.emb_dim) # 输出用于计算 aindex
        )

        self.mu = None
        self.logvar = None
        # --- 因果 VAE 相关参数 ---
        # 确保 params 中定义了 latent_dim (建议设为 32，即 16 alpha + 16 beta)
        self.latent_dim = getattr(params, 'latent_dim', 32) 

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)  # 注意：+1 是为了给 "no relation" 留出一个特殊 ID (通常为 num_rels)，以便 attention 机制使用
        else:
            self.attn_rel_emb = None

        # --- 聚合器初始化 ---
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # --- [关键修改] 初始化 VAE Encoder ---
        # 这里的 in_channels 对应 GraIL 首层输出的 emb_dim
        
        ''' self.vae_encoder = RelationalVAEEncoder(
           in_channels=self.emb_dim, 
            out_channels=self.latent_dim, 
            num_rels=self.aug_num_rels, 
            num_bases=self.num_bases, 
            device=self.device
        )'''

        # --- 构建 GraIL 骨干层 ---
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # Input 层 (i2h)
        self.layers.append(self.build_input_layer())
        # Hidden 层 (h2h)
        for idx in range(self.num_hidden_layers - 1):
            self.layers.append(self.build_hidden_layer(idx))

    def build_input_layer(self):
        return RGCNLayer(self.inp_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim,
                         self.aug_num_rels, self.num_bases, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout,
                         is_input_layer=True, has_attn=self.has_attn)

    def build_hidden_layer(self, idx):
        return RGCNLayer(self.emb_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim,
                         self.aug_num_rels, self.num_bases, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout,
                         has_attn=self.has_attn)

    def _internal_decoder_wrapper(self, z):
        """
        供 Trainer 调用：loss_recon = MSE(decoder(z), x_raw)
        """
        # 利用 RelationalVAEEncoder 内部自带的线性层还原特征
        x_rec = self.encoder.reconstruct_decoder(z)
        # 邻接矩阵重构：CIGNN 常用方式是隐向量内积
        adj_rec = torch.matmul(z, z.t()) 
        return x_rec, adj_rec

    def forward(self, g, edge_weight=None):
        # --- 1. 初始化掩码 ---
        # 这里的设备转换要确保和 g 一致
        device = next(self.parameters()).device
        
        h_raw = g.ndata['feat'].clone()

        if edge_weight is not None:
            g.edata['causal_mask'] = edge_weight.to(device).view(-1, 1)
        else:
            g.edata['causal_mask'] = torch.ones(g.number_of_edges(), 1).to(device)
        
        # 使用 8 维特征进行 VAE 编码，这样维度就匹配了 [N, 8] @ [8, 64]
        z, mu, logvar = self.encoder(g, h_raw) 
        self.mu, self.logvar = mu, logvar

        # --- 3. 因果掩码生成 (Alpha 逻辑) ---
        mid = z.shape[1] // 2
        alpha = z[:, :mid]
        u, v = g.edges()
        
        if edge_weight is None:
            # 没有外部干预权重时，使用 alpha 相似度生成默认因果掩码。
            e_score = torch.sum(alpha[u] * alpha[v], dim=1, keepdim=True)
            causal_mask = torch.sigmoid(e_score)
            g.edata['causal_mask'] = causal_mask

        # --- 4. 执行 RGCN 卷积推理 ---
        # 现在所有的卷积层（包括第一层）都会在 msg_func 里读取到刚才生成的 'causal_mask'
        for i, layer in enumerate(self.layers):
            layer(g, self.attn_rel_emb)

        return g.ndata['repr']  # 最终输出节点表示，供后续分类器使用
