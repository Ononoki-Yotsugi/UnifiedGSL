import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.data.utils import generate_mask_tensor
from data import load_dataset
from data.pyg_load import pyg_load_dataset
from data.split import get_split
from copy import deepcopy
from models.grcn import GRCN
import torch
import numpy as np
import time
from utils.utils import accuracy, set_seed, sample_mask
from utils.logger import Logger
import os

# split_seeds = [0,1,2,3,4]
# train_seeds = [0,1,2,3,4,5,6,7,8,9,
#                10,11,12,13,14,15,16,17,18,19,
#                20,21,22,23,24,25,26,27,28,29,
#                30,31,32,33,34,35,36,37,38,39,
#                40,41,42,43,44,45,46,47,48,49]
split_seeds = [i for i in range(20)]
train_seeds = [i for i in range(400)]


class Solver(nn.Module):
    def __init__(self, args, conf):
        super().__init__()
        print("Solver Version : [{}]".format("grcn"))
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.prepare_data(args.data)
        if conf.save_graph:
            self.graph_loc = 'records/graph/{}_{}.pth'.format(args.solver, args.data)
            if not os.path.exists('records/graph'):
                os.makedirs('records/graph')

    def prepare_data(self, ds_name):
        dataset_raw = pyg_load_dataset(ds_name)
        self.g = dataset_raw[0]
        self.feats = self.g.x.to(self.device)   #这个feats已经经过归一化了
        self.n_nodes = self.feats.shape[0]
        self.labels = self.g.y.to(self.device)
        self.dim_feats = self.feats.shape[1]
        self.n_classes = dataset_raw.num_classes
        loop_edge_index = torch.stack([torch.arange(self.n_nodes), torch.arange(self.n_nodes)])
        edges = torch.cat([self.g.edge_index, loop_edge_index], dim=1)
        self.adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [self.n_nodes, self.n_nodes]).to(self.device).coalesce()
        self.n_edges = self.g.num_edges
        if self.args.verbose:
            print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d"""%
                  (self.n_nodes, self.n_edges, self.n_classes))

    def split_data(self, ds_name, seed):
        if ds_name in ['coauthorcs', 'coauthorph', 'amazoncom', 'amazonpho'] or self.conf.re_split:
            np.random.seed(seed)
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),20,30)   # 默认采取20-30-rest这种划分
            self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes)).to(self.device)
            self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes)).to(self.device)
            self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes)).to(self.device)
        elif ds_name == 'wikics':
            assert seed <= 19 and seed >=0
            self.train_mask = self.g.train_mask[:,seed].bool()
            self.val_mask = self.g.val_mask[:,seed].bool()
            self.test_mask = self.g.test_mask.bool()
        else:
            self.train_mask = self.g.train_mask.to(self.device)
            self.val_mask = self.g.val_mask.to(self.device)
            self.test_mask = self.g.test_mask.to(self.device)
        self.train_mask = torch.nonzero(self.train_mask, as_tuple=False).squeeze()
        self.val_mask = torch.nonzero(self.val_mask, as_tuple=False).squeeze()
        self.test_mask = torch.nonzero(self.test_mask, as_tuple=False).squeeze()

        if self.args.verbose:
            print("""----Split statistics------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (len(self.train_mask), len(self.val_mask), len(self.test_mask)))

    def train(self):
        self.reset()
        self.start_time = time.time()

        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim1.zero_grad()
            self.optim2.zero_grad()

            # forward and backward
            output = self.model(self.feats, self.adj)
            loss_train = F.nll_loss(output[self.train_mask], self.labels[self.train_mask])
            acc_train = accuracy(output[self.train_mask], self.labels[self.train_mask])
            loss_train.backward()
            self.optim1.step()
            self.optim2.step()

            # Evaluate
            loss_val, acc_val, output = self.evaluate(self.val_mask)

            # save
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())

            # print
            if self.args.debug:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        if self.conf.save_graph:
            torch.save(self.best_graph.cpu(), self.graph_loc)
        return self.result

    def evaluate(self, test_mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.feats, self.adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=F.nll_loss(logits, labels)
        return loss, accuracy(logits, labels), output

    def test(self):
        self.model.load_state_dict(self.weights)
        return self.evaluate(self.test_mask)

    def run(self):
        total_runs = self.args.n_runs * self.args.n_splits
        assert self.args.n_splits <= len(split_seeds)
        assert total_runs <= len(train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(self.args.n_splits):
            self.split_data(self.args.data, split_seeds[i])   # split the data
            for j in range(self.args.n_runs):
                k = i * self.args.n_runs + j
                print("Exp {}/{}".format(k, total_runs))
                set_seed(train_seeds[k])
                result = self.train()
                logger.add_result(k, result)
        logger.print_statistics()

    def reset(self):
        # 这里使用reset的方式，否则train_gcn等函数需要大量参数
        self.model = GRCN(self.n_nodes, self.feats.shape[1], self.n_classes, self.device, self.conf)
        self.model = self.model.to(self.device)
        self.optim1 = torch.optim.Adam(self.model.base_parameters(), lr=self.conf.lr, weight_decay=self.conf.wd)
        self.optim2 = torch.optim.Adam(self.model.graph_parameters(), lr=self.conf.lr_graph)

        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}


