import json
import os
import argparse
import logging
import random
import torch
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
from managers.evaluator import Evaluator

from warnings import simplefilter
from model.dgl.graph_classifier import GraphClassifier


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


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    # 1. 强制确定真实的训练输出目录
    actual_train_dir = os.path.join(os.getcwd(), 'experiments', params.experiment_name)
    config_path = os.path.join(actual_train_dir, 'params.json')
    
    # 2. 从 JSON 同步配置
    if os.path.exists(config_path):
        logging.info(f"正在同步配置: {config_path}")
        with open(config_path, 'r') as f:
            train_params = json.load(f)
        for key, value in train_params.items():
            if key != 'exp_dir':
                setattr(params, key, value)
    
    # 3. 🌟 核心修复：注入动态参数 (骨架图纸)
    # 必须在 initialize_model 之前，防止 RGCN 初始化报 AttributeError
    if not hasattr(params, 'max_label_value') or params.max_label_value is None:
        params.max_label_value = params.hop 
        logging.info(f"✅ 参数注入: max_label_value = {params.max_label_value}")
        
    

    # 4. 补全 RGCN 必须的所有超参数（保底补丁）
    must_have = {
        'emb_dim': 64, 
        'latent_dim': 64,
        'rel_emb_dim': 64, 
        'attn_rel_emb_dim': 64, 
        'num_gcn_layers': 3, 'num_bases': 4, 'dropout': 0, 
        'edge_dropout': 0.5, 'gnn_agg_type': 'sum', 'has_attn': True, 
        'add_ht_emb': True, 'inp_dim': 8, 'stage': 1,
        'max_links': 1000000 ,# 🌟 加上这一行，解决当前的报错
        'max_nodes_per_hop': None , # 🌟 修复：对应当前的报错，None 表示不限制
        'use_kge_embeddings': False, 
        'kge_model': 'TransE'
    }
    for k, v in must_have.items():
        if not hasattr(params, k) or getattr(params, k) is None:
            setattr(params, k, v)

    # 5. 维度确认：确保 num_rels 正确
    if not hasattr(params, 'num_rels') or params.num_rels is None:
        rels = set()
        train_path = os.path.join(os.getcwd(), f'data/{params.dataset}/train.txt')
        with open(train_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3: rels.add(parts[1])
        params.num_rels = len(rels)
        params.aug_num_rels = params.num_rels

    logging.info(f"✅ 配置锁定：Relations={params.num_rels}, MaxLabel={params.max_label_value}, Emb={params.emb_dim}")

    # 6. 🌟 加载模型 (现在它会先建骨架，再填权重)
    params.exp_dir = actual_train_dir 
    graph_classifier = initialize_model(params, GraphClassifier, load_model=True)

    logging.info(f"Device: {params.device}")
    
    # --- 后续测试逻辑 ---
    all_auc = []
    all_auc_pr = []
    
    for r in range(1, params.runs + 1):
        # 路径确保：测试子图存放位置
        params.db_path = os.path.join(os.getcwd(), f'data/{params.dataset}/test_subgraphs_en_{params.enclosing_sub_graph}')
        
        # 默认复用已生成的测试 LMDB，保证多次测试使用同一批负样本。
        if params.regenerate_test_subgraphs or not os.path.isdir(params.db_path):
            logging.info(f"Generating test subgraphs at {params.db_path}")
            generate_subgraph_datasets(params, splits=['test'],
                                       saved_relation2id=graph_classifier.relation2id,
                                       max_label_value=graph_classifier.gnn.max_label_value)
        else:
            logging.info(f"Reusing fixed test subgraphs at {params.db_path}")
        
        test = SubgraphDataset(params.db_path, 'test_pos', 'test_neg', params.file_paths, graph_classifier.relation2id,
                               add_traspose_rels=params.add_traspose_rels,
                               num_neg_samples_per_link=params.num_neg_samples_per_link,
                               use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                               kge_model=params.kge_model, file_name=params.test_file)
        
        test_evaluator = Evaluator(params, graph_classifier, test)
        
        # 这里 Evaluator.eval 内部应该已经改为调用 model.predict()
        result = test_evaluator.eval(save=False)

        logging.info(f'Run {r} Result: {result}')
        
        all_auc.append(result['auc'])
        all_auc_pr.append(result['auc_pr'])
    
    logging.info(f'Final Mean AUC: {np.mean(all_auc):.4f}')
    logging.info(f'Final Mean AUC-PR: {np.mean(all_auc_pr):.4f}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", type=str, default="causal_run")
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--test_file", "-t", type=str, default="test")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hop", type=int, default=3)
    parser.add_argument('--enclosing_sub_graph', '-en', type=str2bool, default=True)
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0)
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1)
    parser.add_argument('--add_traspose_rels', '-tr', type=str2bool, default=False)
    parser.add_argument('--regenerate_test_subgraphs', action='store_true')

    params = parser.parse_args()
    set_random_seed(params.seed)
    params.main_dir = os.getcwd()
    
    # 路径拼接逻辑：对应 experiments/MyRun/WN18RR_v1_exp
    params.experiment_name = os.path.join(params.experiment_name, f"{params.dataset}_exp")
    
    # 初始化实验目录 (生成 test 相关日志目录)
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/train.txt'.format(params.dataset)),
        'test': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_file))
    }
    
    params.device = torch.device(f'cuda:{params.gpu}' if not params.disable_cuda and torch.cuda.is_available() else 'cpu')
    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl
    
    main(params)
