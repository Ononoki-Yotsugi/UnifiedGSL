import torch.nn as nn
import torch
from data import load_dataset
from data.pyg_load import pyg_load_dataset
import dgl
from data.split import get_split
from dgl.data.utils import generate_mask_tensor
from utils.utils import sample_mask, set_seed
import numpy as np
from utils.logger import Logger


class BaseSolver(nn.Module):
    def __init__(self, args, conf):
        super().__init__()
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.prepare_data(args.data, mode=self.args.data_load)
        self.split_seeds = [i for i in range(20)]
        self.train_seeds = [i for i in range(400)]

    def prepare_data(self, ds_name, mode='dgl'):
        if mode =='pyg':
            self.data_raw = pyg_load_dataset(ds_name)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # 这个feats尚未经过归一化
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse.FloatTensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                                [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges
            if not ('data_cpu' in self.conf and self.conf.data_cpu):
                self.feats = self.feats.to(self.device)
                self.labels = self.labels.to(self.device)
                self.adj = self.adj.to(self.device)
        else:
            self.data_raw, g = load_dataset(ds_name)
            self.g = dgl.remove_self_loop(g)  # this operation is aimed to get a adj without self loop
            self.adj = self.g.adj()
            if not ('data_cpu' in self.conf and self.conf.data_cpu):
                self.g = self.g.int().to(self.device)
                self.adj = self.adj.to(self.device)
            self.feats = self.g.ndata['feat']  # 这个feats已经经过归一化了
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.ndata['label']
            self.n_edges = self.g.number_of_edges()
        self.n_classes = self.data_raw.num_classes
        if self.args.verbose:
            print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d""" %
                  (self.n_nodes, self.n_edges, self.n_classes))

    def split_data(self, ds_name, seed, mode='dgl'):
        if ds_name in ['coauthorcs', 'coauthorph', 'amazoncom', 'amazonpho']:
            np.random.seed(seed)
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), 20, 30)  # 默认采取20-30-rest这种划分
            self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes))
            self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes))
            self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes))
        elif ds_name == 'wikics':
            assert seed <= 19 and seed >= 0
            if mode == 'pyg':
                self.train_mask = self.g.train_mask[:, seed].bool()
                self.val_mask = self.g.val_mask[:, seed].bool()
                self.test_mask = self.g.test_mask.bool()
            else:
                self.train_mask = self.g.ndata['train_mask'][:, seed].bool()
                self.val_mask = self.g.ndata['val_mask'][:, seed].bool()
                self.test_mask = self.g.ndata['test_mask'].bool()
        elif ds_name in ['cora', 'citeseer', 'pubmed']:
            if 're_split' in self.conf and self.conf.re_split:
                np.random.seed(seed)
                train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(), 20, 30)  # 默认采取20-30-rest这种划分
                self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes))
                self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes))
                self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes))
            else:
                if mode == 'pyg':
                    self.train_mask = self.g.train_mask
                    self.val_mask = self.g.val_mask
                    self.test_mask = self.g.test_mask
                else:
                    self.train_mask = self.g.ndata['train_mask']
                    self.val_mask = self.g.ndata['val_mask']
                    self.test_mask = self.g.ndata['test_mask']
        elif ds_name == 'ogbn-arxiv':
            if mode == 'pyg':
                self.train_mask = self.g.train_mask.to(self.device)
                self.val_mask = self.g.val_mask.to(self.device)
                self.test_mask = self.g.test_mask.to(self.device)
            else:
                self.train_mask = self.g.ndata['train_mask']
                self.val_mask = self.g.ndata['val_mask']
                self.test_mask = self.g.ndata['test_mask']
        else:
            print('dataset not implemented')
            exit(0)
        self.train_mask = torch.nonzero(self.train_mask, as_tuple=False).squeeze()
        self.val_mask = torch.nonzero(self.val_mask, as_tuple=False).squeeze()
        self.test_mask = torch.nonzero(self.test_mask, as_tuple=False).squeeze()

        if self.args.verbose:
            print("""----Split statistics------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (len(self.train_mask), len(self.val_mask), len(self.test_mask)))

    def run(self):
        total_runs = self.args.n_runs * self.args.n_splits
        assert self.args.n_splits <= len(self.split_seeds)
        assert total_runs <= len(self.train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(self.args.n_splits):
            self.split_data(self.args.data, self.split_seeds[i], mode=self.args.data_load)   # split the data
            for j in range(self.args.n_runs):
                k = i * self.args.n_runs + j
                print("Exp {}/{}".format(k, total_runs))
                set_seed(self.train_seeds[k])
                result = self.train()
                logger.add_result(k, result)
        logger.print_statistics()