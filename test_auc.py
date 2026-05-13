import json
import os
import argparse
import logging
import random
import hashlib
import torch
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_model
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


def apply_causal_mode_defaults(params, force=True):
    """Keep evaluation on the same full causal branch used in training."""
    if getattr(params, 'causal_mode', 'none') != 'full':
        return

    def set_default(name, value, baseline=None):
        current = getattr(params, name, None)
        if force or current is None or current == baseline:
            setattr(params, name, value)

    set_default('use_cignn_mask', True, False)
    set_default('use_causal_effect_loss', False, True)
    set_default('use_vae_loss', True, False)
    set_default('use_mi_loss', True, False)
    set_default('use_cmi_loss', True, False)
    set_default('cignn_mask_mode', 'attention_only', 'none')
    set_default('mask_injection_gamma', 1.0, 0.0)
    set_default('mask_gamma_schedule', 'linear', 'none')
    set_default('mask_ramp_epochs', max(getattr(params, 'mask_ramp_epochs', 0), 10), 0)
    set_default('lambda_effect', 0.0)
    set_default('lambda_vae', 0.05)
    set_default('lambda_mi', 0.01)
    set_default('lambda_cmi', 0.05)
    set_default('lambda_sparse', 0.001)
    set_default('pretrain_vae_only', False, True)
    set_default('warmup_epochs', getattr(params, 'warmup_epochs', 0), 0)


def _with_default(params, name, value):
    if not hasattr(params, name) or getattr(params, name) is None:
        setattr(params, name, value)


def _log_eval_configuration(params):
    logging.info("Loaded params.json: %s", params.config_path)
    logging.info("Checkpoint path: %s", params.checkpoint_path)
    logging.info("Train dataset: %s", params.train_dataset)
    logging.info("Test dataset: %s", params.test_dataset)
    logging.info(
        "Model config: emb_dim=%s, rel_emb_dim=%s, attn_rel_emb_dim=%s, "
        "num_gcn_layers=%s, num_bases=%s, has_attn=%s, add_ht_emb=%s",
        params.emb_dim, params.rel_emb_dim, params.attn_rel_emb_dim,
        params.num_gcn_layers, params.num_bases, params.has_attn,
        params.add_ht_emb
    )
    logging.info("CIGNN mask enabled: %s", params.use_cignn_mask)
    logging.info("CIGNN mask mode: %s", params.cignn_mask_mode)
    logging.info("Causal effect loss enabled: %s", params.use_causal_effect_loss)
    logging.info("VAE loss enabled: %s", params.use_vae_loss)
    logging.info("MI loss enabled: %s", params.use_mi_loss)
    logging.info("CMI loss enabled: %s", params.use_cmi_loss)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_train_dataset(test_dataset):
    return test_dataset[:-4] if test_dataset.endswith('_ind') else test_dataset


def _infer_relation_count(params):
    relation2id_path = os.path.join(params.main_dir, f'data/{params.train_dataset}/relation2id.json')
    if os.path.exists(relation2id_path):
        with open(relation2id_path) as f:
            return len(json.load(f))

    rels = set()
    for split in ('train', 'valid'):
        path = os.path.join(params.main_dir, f'data/{params.train_dataset}/{split}.txt')
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    rels.add(parts[1])
    return len(rels)


def _resolve_train_dir(experiment_name, train_dataset, test_dataset):
    candidates = [
        os.path.join(os.getcwd(), 'experiments', experiment_name, f'{train_dataset}_exp'),
        os.path.join(os.getcwd(), 'experiments', experiment_name, f'{test_dataset}_exp'),
        os.path.join(os.getcwd(), 'experiments', experiment_name),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'params.json')):
            return candidate
    return candidates[0]


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    cli_params = vars(params).copy()
    test_dataset = cli_params['dataset']
    train_dataset = cli_params['train_dataset'] or _infer_train_dataset(test_dataset)
    if cli_params.get('checkpoint'):
        checkpoint_path = os.path.abspath(cli_params['checkpoint'])
        actual_train_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(actual_train_dir, 'params.json')
    else:
        actual_train_dir = _resolve_train_dir(params.experiment_name, train_dataset, test_dataset)
        config_path = os.path.join(actual_train_dir, 'params.json')
        checkpoint_path = os.path.join(actual_train_dir, 'best_model.pth')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Training params.json not found: {config_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'best_model.pth not found: {checkpoint_path}')

    before_params_hash = _file_sha256(config_path)
    logging.info("Before test params hash: %s", before_params_hash)

    logging.info("Reading training configuration from %s", config_path)
    with open(config_path, 'r') as f:
        train_params = json.load(f)

    for key, value in train_params.items():
        if key not in {'exp_dir', 'device', 'collate_fn', 'move_batch_to_device'}:
            setattr(params, key, value)

    params.train_dataset = train_params.get('dataset', train_dataset)
    params.test_dataset = test_dataset
    params.dataset = params.train_dataset
    params.test_file = cli_params['test_file']
    params.runs = cli_params['runs']
    params.seed = cli_params['seed']
    params.gpu = cli_params['gpu']
    params.disable_cuda = cli_params['disable_cuda']
    params.num_workers = cli_params['num_workers']
    params.batch_size = cli_params['batch_size']
    params.regenerate_test_subgraphs = cli_params['regenerate_test_subgraphs']
    params.causal_mode = (
        cli_params['causal_mode']
        if cli_params['causal_mode'] != 'none'
        else train_params.get('causal_mode', 'none')
    )
    params.main_dir = os.getcwd()
    params.exp_dir = actual_train_dir
    params.config_path = config_path
    params.checkpoint_path = checkpoint_path
    
    defaults = {
        'emb_dim': 64, 'latent_dim': 64, 'rel_emb_dim': 64,
        'attn_rel_emb_dim': 64, 'num_gcn_layers': 3, 'num_bases': 4,
        'dropout': 0, 'edge_dropout': 0.5, 'gnn_agg_type': 'sum',
        'has_attn': True, 'add_ht_emb': True, 'inp_dim': 8, 'stage': 1,
        'max_links': 1000000, 'max_nodes_per_hop': None,
        'use_kge_embeddings': False, 'kge_model': 'TransE',
        'use_cignn_mask': False, 'use_vae_loss': False,
        'use_causal_effect_loss': False, 'use_mi_loss': False, 'use_cmi_loss': False,
        'pretrain_vae_only': False, 'load_pretrained_vae': '',
        'freeze_vae_after_pretrain': False, 'cignn_mask_mode': 'none',
        'mask_injection_gamma': 0.0, 'mask_gamma_schedule': 'none',
        'mask_ramp_epochs': 0, 'debug_baseline_check': False,
        'debug_grad_check': False, 'lambda_effect': 0.0,
        'lambda_sparse': 0.001, 'warmup_epochs': 0,
    }
    for key, value in defaults.items():
        _with_default(params, key, value)

    apply_causal_mode_defaults(params, force=cli_params['causal_mode'] == 'full')
    if getattr(params, 'use_causal_effect_loss', False) or getattr(params, 'use_cmi_loss', False):
        if not getattr(params, 'use_cignn_mask', False) or getattr(params, 'cignn_mask_mode', 'none') == 'none':
            raise RuntimeError(
                'Causal checkpoint/config requested, but CIGNN mask is disabled. '
                'Use --causal_mode full or fix params.json.'
            )

    if not hasattr(params, 'num_rels') or params.num_rels is None:
        params.num_rels = _infer_relation_count(params)
        params.aug_num_rels = params.num_rels * (2 if params.add_traspose_rels else 1)

    if not hasattr(params, 'max_label_value') or params.max_label_value is None:
        params.max_label_value = params.hop

    _log_eval_configuration(params)

    graph_classifier = initialize_model(params, GraphClassifier, load_model=True)
    params.dataset = params.test_dataset
    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/train.txt'.format(params.test_dataset)),
        'test': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.test_dataset, params.test_file))
    }

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

    after_params_hash = _file_sha256(config_path)
    logging.info("After test params hash: %s", after_params_hash)
    if after_params_hash != before_params_hash:
        raise RuntimeError(
            f'params.json changed during test: before={before_params_hash}, after={after_params_hash}'
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", type=str, default="causal_run")
    parser.add_argument("--dataset", "-d", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--causal_mode", type=str, choices=['none', 'full'], default='none')
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
    
    params.file_paths = {
        'train': os.path.join(params.main_dir, 'data/{}/train.txt'.format(params.dataset)),
        'test': os.path.join(params.main_dir, 'data/{}/{}.txt'.format(params.dataset, params.test_file))
    }
    
    params.device = torch.device(f'cuda:{params.gpu}' if not params.disable_cuda and torch.cuda.is_available() else 'cpu')
    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl
    
    main(params)
