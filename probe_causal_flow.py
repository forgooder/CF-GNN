import torch
import dgl
import logging
import sys
import os
from model.dgl.rgcn_model import RGCN
from model.dgl.layers import RGCNLayer
from model.dgl.aggregators import SumAggregator, MLPAggregator, GRUAggregator
from model.dgl.vae_modules.RelationalVAEEncoder import RelationalVAEEncoder


class DummyParams:
    """ 伪造参数类，模拟 main.py 传参 """
    def __init__(self):
        self.dataset = "WN18RR_v1"
        self.device = torch.device('cpu')
        self.max_label_value = 8
        self.inp_dim = 108   # 距离标签 + KGE
        self.emb_dim = 32    # RGCN 隐藏层
        self.attn_rel_emb_dim = 32
        self.num_rels = 11
        self.aug_num_rels = 22
        self.num_gcn_layers = 3
        self.num_bases = 4
        self.dropout = 0.1
        self.edge_dropout = 0.2
        self.has_attn = True
        self.gnn_agg_type = "sum"
        self.latent_dim = 32  # 16 alpha + 16 beta
        self.stage = 1
        self.load_model = False

def probe():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    params = DummyParams()

    # --- 步骤 1: 实例化模型 ---
    logging.info("--- 正在初始化 RGCN + VAE 模型 ---")
    try:
        model = RGCN(params)
        model.eval()
        logging.info("✅ 模型初始化：成功")
    except Exception as e:
        logging.error(f"❌ 模型初始化：失败 -> {e}")
        return

    # --- 步骤 2: 伪造输入数据 (模拟 DataLoader 产出) ---
    logging.info("--- 正在构建伪造 DGL 子图 ---")
    num_nodes = 10
    num_edges = 25
    g = dgl.graph((torch.randint(0, num_nodes, (num_edges,)), 
                   torch.randint(0, num_nodes, (num_edges,))),
                   num_nodes=num_nodes)
    
    # 填充必要特征
    g.ndata['feat'] = torch.randn(num_nodes, params.inp_dim)
    g.edata['type'] = torch.randint(0, params.aug_num_rels, (num_edges,))
    g.edata['label'] = torch.randint(0, params.num_rels, (num_edges,))
    g.ndata['h'] = torch.randn(num_nodes, params.emb_dim) # 模拟首层输入
    logging.info(f"子图构建完成: {num_nodes} 点, {num_edges} 边")

    # --- 步骤 3: 探测 VAE 编码流 ---
    logging.info("--- 探测 VAE 编码逻辑 ---")
    try:
        # 手动调用 VAE
        z, mu, logstd = model.vae_encoder(g, g.ndata['h'])
        logging.info(f"✅ VAE 隐变量维度: {z.shape} (预期: [10, 32])")
        logging.info(f"✅ 重构特征维度: {model.vae_encoder.x_est.shape} (预期: [10, 32])")
    except Exception as e:
        logging.error(f"❌ VAE 逻辑崩溃: {e}")

    # --- 步骤 4: 探测 掩码生成与注入 ---
    logging.info("--- 探测 Causal Mask 注入 ---")
    try:
        # 模拟 forward 内部的 alpha 提取
        alpha = z[:, :params.latent_dim // 2]
        u, v = g.edges()
        e_score = torch.sum(alpha[u] * alpha[v], dim=1, keepdim=True)
        mask = torch.sigmoid(e_score)
        g.edata['causal_mask'] = mask
        logging.info(f"✅ Causal Mask 维度: {mask.shape} (预期: [25, 1])")
    except Exception as e:
        logging.error(f"❌ Mask 生成失败: {e}")

    # --- 步骤 5: 探测 RGCN 消息传递 (Layer 注入检查) ---
    logging.info("--- 探测去噪消息传递 (Layer.forward) ---")
    try:
        g.ndata['repr'] = g.ndata['h'].unsqueeze(1) 
    
         # 现在运行第二层就不会报错了
        model.layers[1](g, model.attn_rel_emb)
        logging.info("✅ RGCN 去噪卷积运行：成功")
    except Exception as e:
        logging.error(f"❌ RGCN 卷积报错 (检查 layers.py 里的 msg_func): {e}")

if __name__ == "__main__":
    probe()