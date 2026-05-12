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

        self.params = params
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
        if self.latent_dim % 2 != 0:
            raise ValueError('latent_dim must be even so it can be split into alpha/beta branches.')
        self.encoder = RelationalVAEEncoder(
            in_channels=self.inp_dim, 
            out_channels=self.latent_dim, 
            num_rels=params.aug_num_rels, 
            num_bases=params.num_bases, 
            device=params.device
        )

        self.decoder = self._internal_decoder_wrapper

        self.causal_decoder = nn.Sequential(
            nn.Linear(self.latent_dim // 2, params.emb_dim),
            nn.ReLU(),
            nn.Linear(params.emb_dim, params.emb_dim) # 输出用于计算 aindex
        )
        self.shortcut_decoder = nn.Sequential(
            nn.Linear(self.latent_dim // 2, params.emb_dim),
            nn.ReLU(),
            nn.Linear(params.emb_dim, params.emb_dim)
        )

        self.mu = None
        self.logvar = None

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

    @property
    def casual_decoder(self):
        # Backward-compatible spelling for older call sites.
        return self.causal_decoder

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

    def forward(self, g, edge_weight=None, edge_mask_logits=None):
        device = next(self.parameters()).device
        use_cignn_mask = (
            getattr(self.params, 'use_cignn_mask', False)
            and getattr(self.params, 'cignn_mask_mode', 'none') != 'none'
        )
        g.use_cignn_mask = use_cignn_mask
        g.cignn_mask_mode = getattr(self.params, 'cignn_mask_mode', 'none')
        g.vae_encoder_called = False
        g.message_mask_applied = False
        g.attention_mask_applied = False
        g.mask_logits_ref = None
        g.raw_mask_ref = None
        g.effective_mask_ref = None

        if 'causal_mask' in g.edata:
            g.edata.pop('causal_mask')
        if 'raw_cignn_mask' in g.edata:
            g.edata.pop('raw_cignn_mask')
        if 'effective_cignn_mask' in g.edata:
            g.edata.pop('effective_cignn_mask')

        if use_cignn_mask and edge_weight is not None:
            raw_mask = edge_weight.to(device).view(-1, 1)
            gamma = getattr(self.params, 'current_mask_gamma', getattr(self.params, 'mask_injection_gamma', 0.0))
            effective_mask = 1.0 - gamma + gamma * raw_mask
            g.edata['raw_cignn_mask'] = raw_mask
            g.edata['effective_cignn_mask'] = effective_mask
            g.mask_logits_ref = edge_mask_logits.to(device).view(-1, 1) if edge_mask_logits is not None else None
            g.raw_mask_ref = raw_mask
            g.effective_mask_ref = effective_mask
        
        if use_cignn_mask and edge_weight is None:
            h_raw = g.ndata['feat'].clone()
            z, mu, logvar = self.encoder(g, h_raw)
            g.vae_encoder_called = True
            self.mu, self.logvar = mu, logvar

            mid = z.shape[1] // 2
            alpha = z[:, :mid]
            u, v = g.edges()
            z_causal = self.causal_decoder(alpha)
            e_score = torch.sum(z_causal[u] * z_causal[v], dim=1, keepdim=True)
            raw_mask = torch.sigmoid(e_score)
            gamma = getattr(self.params, 'current_mask_gamma', getattr(self.params, 'mask_injection_gamma', 0.0))
            effective_mask = 1.0 - gamma + gamma * raw_mask
            g.edata['raw_cignn_mask'] = raw_mask
            g.edata['effective_cignn_mask'] = effective_mask
            g.mask_logits_ref = e_score
            g.raw_mask_ref = raw_mask
            g.effective_mask_ref = effective_mask

        for i, layer in enumerate(self.layers):
            layer(g, self.attn_rel_emb)

        return g.ndata['repr']  # 最终输出节点表示，供后续分类器使用
