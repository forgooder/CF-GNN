import os
import random
import argparse
import logging
import json
import time
import hashlib

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl

from model.dgl.graph_classifier import GraphClassifier
from utils.initialization_utils import initialize_model


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
    if getattr(params, 'causal_mode', 'none') != 'full':
        return

    def set_default(name, value, baseline=None):
        current = getattr(params, name, None)
        if force or current is None or current == baseline:
            setattr(params, name, value)

    set_default('use_cignn_mask', True, False)
    set_default('use_causal_effect_loss', True, False)
    set_default('use_vae_loss', True, False)
    set_default('use_mi_loss', True, False)
    set_default('use_cmi_loss', True, False)
    set_default('cignn_mask_mode', 'attention_only', 'none')
    set_default('mask_injection_gamma', 1.0, 0.0)
    set_default('mask_gamma_schedule', 'linear', 'none')
    set_default('mask_ramp_epochs', max(getattr(params, 'mask_ramp_epochs', 0), 10), 0)
    set_default('lambda_effect', 0.5)
    set_default('lambda_vae', 0.1)
    set_default('lambda_mi', 0.05)
    set_default('lambda_cmi', 0.05)
    set_default('lambda_sparse', 0.001)
    set_default('pretrain_vae_only', False, True)
    set_default('warmup_epochs', max(getattr(params, 'warmup_epochs', 0), 50), 0)


def _with_default(params, name, value):
    if not hasattr(params, name) or getattr(params, name) is None:
        setattr(params, name, value)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_train_dataset(test_dataset):
    return test_dataset[:-4] if test_dataset.endswith('_ind') else test_dataset


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


def _normalize_max_label(max_label_value, hop):
    if isinstance(max_label_value, (list, tuple, np.ndarray)):
        if len(max_label_value) >= 2:
            return [int(max_label_value[0]), int(max_label_value[1])]
        if len(max_label_value) == 1:
            value = int(max_label_value[0])
            return [value, value]
    if max_label_value is None:
        value = int(hop)
    else:
        value = int(max_label_value)
    return [value, value]


def _load_training_config(params):
    cli_params = vars(params).copy()
    test_dataset = cli_params['dataset']
    train_dataset = cli_params['train_dataset'] or _infer_train_dataset(test_dataset)
    if cli_params.get('checkpoint'):
        checkpoint_path = os.path.abspath(cli_params['checkpoint'])
        train_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(train_dir, 'params.json')
    else:
        train_dir = _resolve_train_dir(params.experiment_name, train_dataset, test_dataset)
        config_path = os.path.join(train_dir, 'params.json')
        checkpoint_path = os.path.join(train_dir, 'best_model.pth')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Training params.json not found: {config_path}')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')

    before_params_hash = _file_sha256(config_path)
    with open(config_path, 'r') as f:
        train_params = json.load(f)

    for key, value in train_params.items():
        if key not in {'exp_dir', 'device', 'collate_fn', 'move_batch_to_device'}:
            setattr(params, key, value)

    params.train_dataset = train_params.get('dataset', train_dataset)
    params.test_dataset = test_dataset
    params.dataset = params.train_dataset
    params.experiment_name = cli_params['experiment_name']
    params.checkpoint_path = checkpoint_path
    params.config_path = config_path
    params.exp_dir = train_dir
    params.main_dir = os.getcwd()
    params.mode = cli_params['mode']
    params.runs = cli_params['runs']
    params.seed = cli_params['seed']
    params.gpu = cli_params['gpu']
    params.disable_cuda = cli_params['disable_cuda']
    params.num_workers = cli_params['num_workers']
    params.batch_size = cli_params['batch_size']
    params.test_file = cli_params['test_file']
    params.causal_mode = (
        cli_params['causal_mode']
        if cli_params['causal_mode'] != 'none'
        else train_params.get('causal_mode', 'none')
    )

    defaults = {
        'emb_dim': 64, 'latent_dim': 64, 'rel_emb_dim': 64,
        'attn_rel_emb_dim': 64, 'num_gcn_layers': 3, 'num_bases': 4,
        'dropout': 0, 'edge_dropout': 0.5, 'gnn_agg_type': 'sum',
        'has_attn': True, 'add_ht_emb': True, 'inp_dim': 2 * (params.hop + 1),
        'stage': 1, 'max_links': 1000000, 'max_nodes_per_hop': None,
        'use_kge_embeddings': False, 'kge_model': 'TransE',
        'use_cignn_mask': False, 'use_vae_loss': False,
        'use_causal_effect_loss': False, 'use_mi_loss': False, 'use_cmi_loss': False,
        'pretrain_vae_only': False, 'load_pretrained_vae': '',
        'freeze_vae_after_pretrain': False, 'cignn_mask_mode': 'none',
        'mask_injection_gamma': 0.0, 'mask_gamma_schedule': 'none',
        'mask_ramp_epochs': 0, 'debug_baseline_check': False,
        'debug_grad_check': False, 'lambda_effect': 0.5,
        'lambda_sparse': 0.001, 'warmup_epochs': 0,
    }
    for key, value in defaults.items():
        _with_default(params, key, value)

    apply_causal_mode_defaults(params, force=cli_params['causal_mode'] == 'full')
    if getattr(params, 'use_causal_effect_loss', False):
        if not getattr(params, 'use_cignn_mask', False) or getattr(params, 'cignn_mask_mode', 'none') == 'none':
            raise RuntimeError('Causal-effect ranking requires CIGNN mask. Use --causal_mode full or fix params.json.')

    if not hasattr(params, 'num_rels') or params.num_rels is None:
        params.num_rels = _infer_relation_count(params)
        params.aug_num_rels = params.num_rels * (2 if params.add_traspose_rels else 1)
    if not hasattr(params, 'max_label_value') or params.max_label_value is None:
        params.max_label_value = params.hop
    params.max_label_value = _normalize_max_label(params.max_label_value, params.hop)

    return before_params_hash


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split('\n')[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet['head'][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet['tail'][1] = j

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, node_features=None, kge_entity2id=None):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling(
            (head, tail),
            rel,
            adj_list,
            h=params_.hop,
            enclosing_sub_graph=params_.enclosing_sub_graph,
            max_nodes_per_hop=getattr(params_, 'max_nodes_per_hop', None),
            max_node_label_value=max_node_label_value
        )

        subgraph = dgl.DGLGraph(dgl_adj_list.subgraph(nodes))
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels)


def get_rank(neg_links):
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        with torch.no_grad():
            head_scores = model_.predict(data, params_.device).squeeze(1).detach().cpu().numpy()
        head_rank = int(np.argwhere(np.argsort(head_scores)[::-1] == head_target_id)[0][0] + 1)
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        with torch.no_grad():
            tail_scores = model_.predict(data, params_.device).squeeze(1).detach().cpu().numpy()
        tail_rank = int(np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id)[0][0] + 1)
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        offset = 0
        for i, neg_triplet in enumerate(neg_triplets):
            next_offset = offset + len(neg_triplet['head'][0])
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[offset:next_offset]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')
            offset = next_offset

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        offset = 0
        for i, neg_triplet in enumerate(neg_triplets):
            next_offset = offset + len(neg_triplet['tail'][0])
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[offset:next_offset]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')
            offset = next_offset


def save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_head_predictions.txt'), "w") as f:
        offset = 0
        for i, neg_triplet in enumerate(neg_triplets):
            next_offset = offset + len(neg_triplet['head'][0])
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[offset:next_offset]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')
            offset = next_offset

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_tail_predictions.txt'), "w") as f:
        offset = 0
        for i, neg_triplet in enumerate(neg_triplets):
            next_offset = offset + len(neg_triplet['tail'][0])
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[offset:next_offset]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')
            offset = next_offset


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def main(params):
    before_params_hash = _load_training_config(params)
    params.device = torch.device(
        f'cuda:{params.gpu}'
        if not params.disable_cuda and torch.cuda.is_available()
        else 'cpu'
    )

    logging.info("Loaded params.json: %s", params.config_path)
    logging.info("Checkpoint path: %s", params.checkpoint_path)
    logging.info("Train dataset: %s", params.train_dataset)
    logging.info("Ranking dataset: %s", params.test_dataset)
    logging.info("CIGNN mask enabled: %s", getattr(params, 'use_cignn_mask', False))
    logging.info("CIGNN mask mode: %s", getattr(params, 'cignn_mask_mode', 'none'))
    logging.info("Causal effect loss enabled: %s", getattr(params, 'use_causal_effect_loss', False))

    model = initialize_model(params, GraphClassifier, load_model=True)
    model.eval()

    params.dataset = params.test_dataset
    params.file_paths = {
        'graph': os.path.join(params.main_dir, 'data', params.test_dataset, 'train.txt'),
        'links': os.path.join(params.main_dir, 'data', params.test_dataset, params.test_file + '.txt')
    }

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
        params.file_paths,
        model.relation2id,
        params.add_traspose_rels
    )

    node_features, kge_entity2id = get_kge_embeddings(params.dataset, params.kge_model) if params.use_kge_embeddings else (None, None)

    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
        save_to_file(neg_triplets, id2entity, id2relation)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    elif params.mode == 'ruleN':
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(params.ruleN_pred_path, entity2id, relation2id)

    ranks = []
    all_head_scores = []
    all_tail_scores = []

    def consume_rank_results(results_iter):
        for head_scores, head_rank, tail_scores, tail_rank in tqdm(results_iter, total=len(neg_triplets)):
            ranks.append(head_rank)
            ranks.append(tail_rank)

            all_head_scores += head_scores.tolist()
            all_tail_scores += tail_scores.tolist()

    if params.num_workers > 0 and params.device.type == 'cpu':
        with mp.Pool(
            processes=params.num_workers,
            initializer=intialize_worker,
            initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)
        ) as p:
            consume_rank_results(p.imap(get_rank, neg_triplets))
    else:
        intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)
        consume_rank_results(map(get_rank, neg_triplets))

    if params.mode == 'ruleN':
        save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)
    else:
        save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)

    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    logger.info(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')

    after_params_hash = _file_sha256(params.config_path)
    logger.info("Before ranking params hash: %s", before_params_hash)
    logger.info("After ranking params hash: %s", after_params_hash)
    if after_params_hash != before_params_hash:
        raise RuntimeError(
            f'params.json changed during ranking: before={before_params_hash}, after={after_params_hash}'
        )


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--causal_mode", type=str, choices=['none', 'full'], default='none')
    parser.add_argument("--test_file", "-t", type=str, default="test")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--use_kge_embeddings", "-kge", type=str2bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=str2bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=3,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None)
    parser.add_argument('--add_traspose_rels', '-tr', type=str2bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')

    params = parser.parse_args()
    set_random_seed(params.seed)
    params.main_dir = os.getcwd()
    params.ruleN_pred_path = os.path.join(params.main_dir, 'data', params.dataset, 'pos_predictions.txt')

    train_dataset = params.train_dataset or _infer_train_dataset(params.dataset)
    train_dir = (
        os.path.dirname(os.path.abspath(params.checkpoint))
        if params.checkpoint
        else _resolve_train_dir(params.experiment_name, train_dataset, params.dataset)
    )
    rank_log_dir = os.path.join(train_dir, f"rank_{params.dataset}")
    if not os.path.exists(rank_log_dir):
        os.makedirs(rank_log_dir)
    file_handler = logging.FileHandler(os.path.join(rank_log_dir, f'log_rank_test_{time.time()}.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)
