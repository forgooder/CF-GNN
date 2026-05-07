import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx

#负采样
def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def links2subgraphs(A, graphs, params, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        # 🌟 核心修复：兼容处理 int 和 array 类型
        current_max_v = max_n_label['value']
        
        # 判断是数组还是单个整数
        if isinstance(current_max_v, (list, np.ndarray)) and len(current_max_v) >= 2:
            val_sub = int(current_max_v[0])
            val_obj = int(current_max_v[1])
        elif isinstance(current_max_v, (int, np.integer)):
            val_sub = int(current_max_v)
            val_obj = int(current_max_v)
        else:
            # 最后的兜底逻辑：如果类型实在奇怪，用 params.hop 作为默认值
            val_sub = int(params.hop)
            val_obj = int(params.hop)

        # 计算 bit 长度 (int.bit_length 至少需要 1 bit，防止 0 的情况)
        bit_len_label_sub = max(int.bit_length(val_sub), 1)
        bit_len_label_obj = max(int.bit_length(val_obj), 1)

        # 写入数据库
        txn.put('max_n_label_sub'.encode(), val_sub.to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), val_obj.to_bytes(bit_len_label_obj, byteorder='little'))

        # --- 以下存储统计信息的代码保持不变 ---
        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))
        
#在正式的大规模数据处理开始前，先通过小样本地抽样调查，算出每个子图平均占用的磁盘空间，以便合理地设置 LMDB 数据库的 map_size 参数，确保在存储大量子图数据时不会遇到空间不足的问题。
def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size

#在每个“工人进程”（Worker Process）启动时，把主进程里的重要数据（如邻接矩阵、配置参数）同步到该进程的“全局变量”中，从而避免重复传输大数据导致的性能瓶颈。
def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value

#每个进程池里的工人都会领到一个三元组任务，然后利用刚才通过 intialize_worker 领到的“全局图纸” A_，把这个三元组变成一个包含丰富特征的子图数据包。
def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)
    #双半径标签（Double-Radius Labeling）是一种节点标记方法，用于在图神经网络中为节点分配标签，以捕捉节点之间的相对位置关系。max_label_value_ 参数用于设置节点标签的最大可能值，以避免标签过大导致模型训练困难或过拟合。通过限制标签的最大值，可以确保模型能够更好地学习节点之间的关系，同时提高训练效率和模型性能。
    #邻居搜索过程中，如果某些节点与根节点的距离超过了 max_label_value_，则将这些节点的标签值限制在 max_label_value_，以确保标签值不会过大，从而帮助模型更好地学习节点之间的关系。
    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)

#从给定的根节点集合出发，在邻接矩阵上进行宽度优先搜索（BFS），以获取指定跳数内的邻居节点集合。通过设置 max_nodes_per_hop 参数，可以限制每跳采样的最大节点数量，从而控制子图的规模和复杂度。
def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    #这个函数是整个子图提取和标记流程的核心。它首先通过 get_neighbor_nodes 函数在邻接矩阵上进行宽度优先搜索，获取指定跳数内的邻居节点集合。
    #然后，根据是否提取封闭子图（enclosing_sub_graph 参数），决定是取邻居节点的交集还是并集来构建子图。接着，利用 node_label 函数为子图中的节点分配标签，
    #以捕捉节点之间的相对位置关系。最后，返回提取的子图节点、对应的标签、子图大小、封闭比率以及被剪枝掉的节点数量等信息，以供后续模型训练使用。
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

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

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    #双半径标签（Double-Radius Labeling）是一种节点标记方法，用于在图神经网络中为节点分配标签，以捕捉节点之间的相对位置关系。
    # max_distance 参数用于设置节点标签的最大距离值，以限制标签的范围，避免标签过大导致模型训练困难或过拟合。
    #通过限制标签的最大距离值，可以确保模型能够更好地学习节点之间的关系，同时提高训练效率和模型性能。
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes
