import random
import numpy as np
from data import load_dataset
from data.split import get_split
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.GCN3 import GCN
from utils.utils import normalize, normalize_sp_tensor, accuracy, set_seed, sample_mask
from copy import deepcopy
import time
import dgl
from dgl.data.utils import generate_mask_tensor
from utils.logger import Logger

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
        print("Solver Version : [{}]".format("gcndense"))
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.normalize = normalize_sp_tensor if self.args.sparse_adj else normalize
        self.prepare_data(args.data)

    def prepare_data(self, ds_name):
        if "reload_gs" in self.conf and self.conf.reload_gs:
            self.data_raw, g = load_dataset(ds_name, reload_gs=True, graph_fn=self.conf.graph_fn)
        else:
            self.data_raw, g = load_dataset(ds_name)

        self.g = g.int().to(self.device)
        self.g = dgl.remove_self_loop(self.g)   # this operation is aimed to get a adj without self loop
        self.feats = self.g.ndata['feat']   #这个feats已经经过归一化了
        self.n_nodes = self.feats.shape[0]
        self.labels = self.g.ndata['label']
        self.dim_feats = self.feats.shape[1]
        self.n_classes = self.data_raw.num_classes
        if self.args.sparse_adj:
            self.adj = self.g.adj().to(self.device)
        else:
            self.adj = self.g.adj().to_dense().to(self.device)
        self.n_edges = self.g.number_of_edges()

    def split_data(self, ds_name, seed):
        if ds_name in ['coauthorcs', 'coauthorph', 'amazoncom', 'amazonpho'] or self.conf.re_split:
            np.random.seed(seed)
            train_indices, val_indices, test_indices = get_split(self.labels.cpu().numpy(),20,30)   # 默认采取20-30-rest这种划分
            self.train_mask = generate_mask_tensor(sample_mask(train_indices, self.n_nodes)).to(self.device)
            self.val_mask = generate_mask_tensor(sample_mask(val_indices, self.n_nodes)).to(self.device)
            self.test_mask = generate_mask_tensor(sample_mask(test_indices, self.n_nodes)).to(self.device)
        if ds_name == 'wikics':
            print(seed)
            assert seed <= 19 and seed >=0
            self.train_mask = self.g.ndata['train_mask'][:,seed].bool()
            self.val_mask = self.g.ndata['val_mask'][:,seed].bool()
            self.test_mask = self.g.ndata['test_mask'].bool()
        self.train_mask = torch.nonzero(self.train_mask, as_tuple=False).squeeze()
        self.val_mask = torch.nonzero(self.val_mask, as_tuple=False).squeeze()
        self.test_mask = torch.nonzero(self.test_mask, as_tuple=False).squeeze()

        if self.args.verbose:
            print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (self.n_nodes,self.n_edges, self.n_classes, len(self.train_mask), len(self.val_mask), len(self.test_mask)))

    def train_gcn(self, adj):
        model = GCN(self.dim_feats, self.conf.n_hidden, self.n_classes, dropout=self.conf.dropout,
                    input_dropout=self.conf.input_dropout, norm=self.conf.norm).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        total_time = 0
        best_val_loss = 10
        weights = None
        result = {'train': 0, 'valid': 0, 'test': 0}
        best_acc_val = 0
        normalized_adj = self.normalize(adj)
        start_time = time.time()
        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()
            model.train()
            optim.zero_grad()

            # forward and backward
            x, output = model(self.feats, normalized_adj)

            loss_train = F.cross_entropy(output[self.train_mask], self.labels[self.train_mask])
            acc_train = accuracy(output[self.train_mask], self.labels[self.train_mask])
            loss_train.backward()
            optim.step()

            # Evaluate
            loss_val, acc_val, _ = self.evaluate(model, self.val_mask, normalized_adj)
            loss_test, acc_test, _ = self.evaluate(model, self.test_mask, normalized_adj)

            # save
            if acc_val > result['valid']:
                improve = '*'
                weights = deepcopy(model.state_dict())
                total_time = time.time() - start_time
                best_val_loss = loss_val
                result['valid'] = acc_val
                result['train'] = acc_train

            # print
            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(total_time))
        loss_test, acc_test, _ = self.test(model, weights)
        result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return result

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
                result = self.train_gcn(self.adj)
                logger.add_result(i * self.args.n_runs + j, result)
        logger.print_statistics()

    def evaluate(self, model, test_mask, normalized_adj):
        model.eval()
        with torch.no_grad():
            x, output = model(self.feats, normalized_adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss = F.cross_entropy(logits, labels)
        return loss, accuracy(logits, labels), output

    def test(self, model, weights):
        model.load_state_dict(weights)
        normalized_adj = self.normalize(self.adj)
        return self.evaluate(model, self.test_mask, normalized_adj)






