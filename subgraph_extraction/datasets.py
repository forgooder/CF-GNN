from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *
import pdb

#生成子图数据集。
def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):

    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, saved_relation2id)
    #邻接表   ，三元组，    实体id，     关系id，  id到实体的映射，id到关系的映射
    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    #路径拼接函数，将主目录、数据目录、数据集名称和文件名拼接成完整的路径
    
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:     #安全打开文件的方式，执行完会自动关闭文件，防止内存泄漏。
            json.dump(relation2id, f)       #将关系id的字典保存为json文件

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}
    #把所有数据（训练、验证、测试）都塞进一个 graphs 变量里，后续处理时只需要写一个循环 for split_name, split in graphs.items(): 就能一次性处理完所有数据，不需要为 train 写一遍代码，再为 valid 复制一遍。
    #'max_size': params.max_links 是一个参数，指定了每个数据集（训练、验证、测试）中要处理的最大链接数量。这是为了控制数据集的规模，避免处理过多的数据导致计算资源不足或训练时间过长。通过设置 max_size，可以确保每个数据集中的链接数量不会超过指定的上限，从而提高模型训练的效率和效果。

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)
    # sample_neg 函数的作用是为每个正样本生成负样本。它会根据给定的邻接表和三元组列表，随机采样负样本，并确保这些负样本不在原始数据中。参数 num_neg_samples_per_link 指定了每个正样本需要生成多少个负样本，max_size 参数限制了生成的负样本数量，constrained_neg_prob 参数控制了生成负样本时是否考虑某些约束条件。
    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)
    #把生成的负样本保存到文件中，文件名包含测试文件名和约束负样本概率，以便后续分析和使用。
    #constrained_neg_prob 是一个参数，控制生成负样本时是否考虑某些约束条件。具体来说，如果 constrained_neg_prob 的值较高，那么在生成负样本时会更多地考虑这些约束条件，从而生成更具挑战性的负样本；如果值较低，则生成的负样本可能更随机，缺乏挑战性。
    links2subgraphs(adj_list, graphs, params, max_label_value)
    #max_label_value 参数用于限制子图中节点标签的最大值，确保生成的子图不会过大或过于复杂，从而提高模型训练的效率和效果。


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))#entity_embedding.npy 文件包含了从 KGE 模型中生成的实体嵌入向量。这些嵌入向量是通过训练 KGE 模型得到的，旨在捕捉实体之间的语义关系和结构信息。通过加载这个文件，我们可以将这些嵌入向量作为节点特征，应用于子图数据集中，从而增强模型对实体之间关系的理解和预测能力。
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)    #id2entity.json 文件包含了从 KGE 模型中生成的实体 ID 到实体名称的映射关系。这个映射关系对于将 KGE 模型中的实体与子图数据集中的实体进行对齐非常重要。通过加载这个文件，我们可以确保在子图数据集中使用的实体 ID 与 KGE 模型中的实体 ID 一致，从而正确地将 KGE 嵌入向量与子图数据集中的实体关联起来。
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}
        #kge_entity2id 是一个字典，它将实体名称映射到对应的实体 ID。这个映射关系是通过反转 kge_id2entity 字典实现的，其中 kge_id2entity 是从 KGE 模型中加载的实体 ID 到实体名称的映射关系。通过创建 kge_entity2id 字典，我们可以方便地根据实体名称获取对应的实体 ID，从而在子图数据集中正确地使用 KGE 嵌入向量。
    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""
    # SubgraphDataset 类是一个 PyTorch 数据集类，用于存储和处理从原始图数据中提取的子图数据。它继承自 torch.utils.data.Dataset 类，提供了 __init__、__getitem__ 和 __len__ 方法，以便在训练过程中能够方便地加载和使用子图数据。这个类主要负责从 LMDB 数据库中读取预处理好的子图数据，并将其转换为 DGL 图格式，以供模型训练使用。
    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        #从 LMDB 数据库中取出第 $i$ 个正样本及其对应的 $N$ 个负样本，并把它们“解压”成模型能看懂的图对象。
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):  
        #告诉程序这个数据集到底有多少组“考题”。
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):

        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes)) #从全局大图中，根据节点 ID 把这一块局部的边和点切出来。
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        #subgraph.edata['type'] 是一个张量，存储了子图中每条边的类型信息。这个信息是从全局大图中提取出来的，通过使用 self.graph.subgraph(nodes).parent_eid 获取子图中边在全局大图中的对应边 ID，然后根据这些边 ID 从全局大图的边数据中提取出边的类型信息，并将其赋值给子图的边数据 'type'。这样，子图中的每条边就有了对应的类型信息，可以在后续的模型训练过程中使用这些信息来区分不同类型的边。
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        #subgraph.edata['label'] 是一个张量，存储了子图中每条边的标签信息。这个标签信息是通过将关系标签 r_label 乘以一个全为 1 的数组（其长度与子图中边的数量相同）来生成的。这样，子图中的每条边都被赋予了相同的标签 r_label，这个标签可以在后续的模型训练过程中用来区分不同类型的边或关系。
        edges_btw_roots = subgraph.edge_id(0, 1)
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)
        '''拓扑重构：把点变成图，并把原来的关系类型找回来。连通保障：确保我们要研究的那两个“主角”（节点 0 和 1）之间有路可走。身份赋予：通过 ID 对齐，把外部高质量的 KGE 特征贴到每个节点上。'''
        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
