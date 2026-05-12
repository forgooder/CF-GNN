import os
import logging
import argparse
import random
import torch
import json
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import (
    initialize_experiment,
    initialize_model,
    load_pretrained_vae_branch,
    save_experiment_params,
)
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_causal_mode_defaults(params):
    """Use one switch to enable the full CIGNN-on-GraIL objective."""
    if getattr(params, 'causal_mode', 'none') != 'full':
        return

    params.use_cignn_mask = True
    params.use_causal_effect_loss = True
    params.use_vae_loss = True
    params.use_mi_loss = True
    params.use_cmi_loss = True
    # A single edge gate is enough; using "both" squares the same mask because
    # GraIL aggregates msg * attention.
    params.cignn_mask_mode = 'attention_only'
    params.mask_injection_gamma = 1.0
    params.mask_gamma_schedule = 'linear'
    params.mask_ramp_epochs = max(getattr(params, 'mask_ramp_epochs', 0), 10)
    params.lambda_effect = 0.5
    params.lambda_vae = 0.1
    params.lambda_mi = 0.05
    params.lambda_cmi = 0.05
    params.lambda_sparse = 0.001
    params.pretrain_vae_only = False
    params.warmup_epochs = max(getattr(params, 'warmup_epochs', 0), 50)


def infer_relation_count(file_paths, relation2id_path):
    if os.path.exists(relation2id_path):
        with open(relation2id_path) as f:
            return len(json.load(f))

    rels = set()
    for path in file_paths.values():
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    rels.add(parts[1])
    return len(rels)


def sync_params_from_dataset(params, dataset):
    params.num_rels = int(dataset.num_rels)
    params.aug_num_rels = int(dataset.aug_num_rels)
    params.inp_dim = int(dataset.n_feat_dim)
    params.max_label_value = [int(dataset.max_n_label[0]), int(dataset.max_n_label[1])]


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    # 1. 自动构建数据库路径
    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    # 2. 生成子图数据
    if not os.path.isdir(params.db_path):
        logging.info(f"正在为 {params.dataset} 生成子图数据库...")
        generate_subgraph_datasets(params)

    # 3. 加载训练集和验证集
    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file)

    sync_params_from_dataset(params, train)
    save_experiment_params(params)

    # 4. 初始化模型 (此时 params 已经包含了完整的动态参数，且已记录在 JSON)
    graph_classifier = initialize_model(params, dgl_model, params.load_model)
    load_pretrained_vae_branch(params, graph_classifier)

    logging.info(f"Device: {params.device}")
    logging.info(f"Exp Dir: {params.exp_dir}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}")

    # 5. 初始化评估器和训练器
    valid_evaluator = Evaluator(params, graph_classifier, valid)
    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info(f'--- Starting Unified Causal-GraIL Training (Dataset: {params.dataset}) ---')

    # 执行训练
    trainer.train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Causal-GraIL Training Script')
    
    # [A. 实验基础控制]
    parser.add_argument("--experiment_name", "-e", type=str, default="causal_run")
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--train_file", "-tf", type=str, default="train")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--causal_mode", type=str, choices=['none', 'full'], default='none')
    
    # [B. 维度占位符 - 核心修复：注册这些参数，使 initialize_experiment 能够识别并保存它们]
    parser.add_argument("--num_rels", type=int, default=None)
    parser.add_argument("--aug_num_rels", type=int, default=None)
    parser.add_argument("--inp_dim", type=int, default=8)
    parser.add_argument("--max_label_value", type=int, default=None)

    # [C. 训练与超参数]
    parser.add_argument("--lambda_vae", type=float, default=0.5)
    parser.add_argument("--lambda_mi", type=float, default=0.3)
    parser.add_argument("--lambda_cmi", type=float, default=0.1)
    parser.add_argument("--lambda_effect", type=float, default=0.5)
    parser.add_argument("--lambda_sparse", type=float, default=0.001)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--num_epochs", "-ne", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--clip", type=int, default=1000)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=10)
    parser.add_argument("--max_links", type=int, default=1000000)
    parser.add_argument("--hop", type=int, default=3)
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--enclosing_sub_graph', '-en', type=str2bool, default=True)
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0)
    parser.add_argument('--add_traspose_rels', '-tr', type=str2bool, default=False)
    
    # [D. 模型架构相关]
    parser.add_argument("--use_kge_embeddings", "-kge", type=str2bool, default=False)
    parser.add_argument("--kge_model", type=str, default="TransE")
    parser.add_argument('--model_type', '-m', type=str, default='dgl')
    
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=64)
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=64)
    parser.add_argument("--emb_dim", "-dim", type=int, default=64)
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3)
    parser.add_argument("--num_bases", "-b", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--edge_dropout", type=float, default=0.5)
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum')
    parser.add_argument('--add_ht_emb', '-ht', type=str2bool, default=True)
    parser.add_argument('--has_attn', '-attn', type=str2bool, default=True)
    parser.add_argument('--use_cignn_mask', type=str2bool, default=False)
    parser.add_argument('--use_causal_effect_loss', type=str2bool, default=False)
    parser.add_argument('--use_vae_loss', type=str2bool, default=False)
    parser.add_argument('--use_mi_loss', type=str2bool, default=False)
    parser.add_argument('--use_cmi_loss', type=str2bool, default=False)
    parser.add_argument('--pretrain_vae_only', type=str2bool, default=False)
    parser.add_argument('--load_pretrained_vae', type=str, default='')
    parser.add_argument('--freeze_vae_after_pretrain', type=str2bool, default=False)
    parser.add_argument('--cignn_mask_mode', type=str,
                        choices=['none', 'message_only', 'attention_only', 'both'],
                        default='none')
    parser.add_argument('--mask_injection_gamma', type=float, default=0.0)
    parser.add_argument('--mask_gamma_schedule', type=str, choices=['none', 'linear'], default='none')
    parser.add_argument('--mask_ramp_epochs', type=int, default=0)
    parser.add_argument('--debug_baseline_check', type=str2bool, default=False)
    parser.add_argument('--debug_grad_check', type=str2bool, default=False)

    params = parser.parse_args()
    apply_causal_mode_defaults(params)
    set_random_seed(params.seed)

    # --- 1. 初始化基础路径 ---
    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    # --- 2. 动态预扫描 (必须在 initialize_experiment 之前执行) ---
    logging.info(f"📊 正在预扫描数据集 {params.dataset} 以动态确定模型维度...")
    try:
        if not os.path.exists(params.file_paths['train']):
             raise FileNotFoundError(f"未找到训练文件: {params.file_paths['train']}")

        relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
        
        # 将结果显式注入 params 对象；最终维度会在 SubgraphDataset 加载后再次校准。
        params.num_rels = infer_relation_count(params.file_paths, relation2id_path)
        params.aug_num_rels = params.num_rels * (2 if params.add_traspose_rels else 1)
        params.max_label_value = params.hop
        params.inp_dim = 2 * (params.hop + 1)
        
        logging.info(f"✅ 维度参数锁定: Relations={params.num_rels}, MaxLabel={params.max_label_value}")
    except Exception as e:
        logging.error(f"❌ 预扫描失败: {e}")
        exit(1)

    # --- 3. 初始化实验：此时 params 已包含扫描结果，initialize_experiment 会将其完整写入 JSON ---
    params.experiment_name = os.path.join(params.experiment_name, f"{params.dataset}_exp")
    initialize_experiment(params, __file__)
    
    # --- 4. 运行环境配置 ---
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)
