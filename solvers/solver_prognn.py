import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.data.utils import generate_mask_tensor
from data import load_dataset
from data.split import get_split
from copy import deepcopy
import torch.optim as optim
from models.GCN2 import GCN
from models.prognn import PGD, prox_operators, EstimateAdj, feature_smoothing
import torch
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
        print("Solver Version : [{}]".format("prognn"))
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.prepare_data(args.data)
        if conf.save_graph:
            self.graph_loc = 'records/graph/{}_{}.pth'.format(args.solver, args.data)
            if not os.path.exists('records/graph'):
                os.makedirs('records/graph')

    def prepare_data(self, ds_name):
        if "reload_gs" in self.conf and self.conf.reload_gs:
            self.data_raw, g = load_dataset(ds_name, reload_gs=True, graph_fn=self.conf.graph_fn)
        else:
            self.data_raw, g = load_dataset(ds_name)
        
        self.g = g.int().to(self.device)
        self.g = dgl.remove_self_loop(self.g)

        self.adj = self.g.adj().to_dense().to(self.device)   # dense
        self.feats = self.g.ndata['feat']
        self.n_nodes = self.feats.shape[0]
        self.labels = self.g.ndata['label']
        self.dim_feats = self.feats.shape[1]
        self.n_classes = self.data_raw.num_classes
        self.n_edges = self.g.number_of_edges()
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
            self.train_mask = self.g.ndata['train_mask'][:,seed].bool()
            self.val_mask = self.g.ndata['val_mask'][:,seed].bool()
            self.test_mask = self.g.ndata['test_mask'].bool()
        else:
            self.train_mask = self.g.ndata['train_mask'].to(self.device)
            self.val_mask = self.g.ndata['val_mask'].to(self.device)
            self.test_mask = self.g.ndata['test_mask'].to(self.device)
        self.train_mask = torch.nonzero(self.train_mask, as_tuple=False).squeeze()
        self.val_mask = torch.nonzero(self.val_mask, as_tuple=False).squeeze()
        self.test_mask = torch.nonzero(self.test_mask, as_tuple=False).squeeze()

        if self.args.verbose:
            print("""----Split statistics------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (len(self.train_mask), len(self.val_mask), len(self.test_mask)))

    def train_gcn(self, epoch):
        normalized_adj = self.estimator.normalize()

        t = time.time()
        improve = ''
        self.model.train()
        self.optimizer.zero_grad()

        # forward and backward
        output = self.model(self.feats, normalized_adj)[-1]
        loss_train = F.nll_loss(output[self.train_mask], self.labels[self.train_mask])
        acc_train = accuracy(output[self.train_mask], self.labels[self.train_mask])
        loss_train.backward()
        self.optimizer.step()

        # evaluate
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)

        # save best model
        if loss_val < self.best_val_loss:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.best_graph = self.estimator.estimated_adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
            epoch+1, time.time() -t, loss_train.item(), acc_train, loss_val, acc_val, improve))

    def train_adj(self, epoch):
        estimator = self.estimator
        t = time.time()
        improve = ''
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - self.adj, p='fro')
        normalized_adj = estimator.normalize()

        if self.conf.lambda_:
            loss_smooth_feat = feature_smoothing(estimator.estimated_adj, self.feats)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(self.feats, normalized_adj)[-1]
        loss_gcn = F.nll_loss(output[self.train_mask], self.labels[self.train_mask])
        acc_train = accuracy(output[self.train_mask], self.labels[self.train_mask])

        #loss_symmetric = torch.norm(estimator.estimated_adj - estimator.estimated_adj.t(), p="fro")
        #loss_differential =  loss_fro + self.conf.gamma * loss_gcn + self.conf.lambda_ * loss_smooth_feat + args.phi * loss_symmetric
        loss_differential = loss_fro + self.conf.gamma * loss_gcn + self.conf.lambda_ * loss_smooth_feat
        loss_differential.backward()
        self.optimizer_adj.step()
        # we finish the optimization of the differential part above, next we need to do the optimization of loss_l1 and loss_nuclear

        loss_nuclear =  0 * loss_fro
        if self.conf.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                     + self.conf.gamma * loss_gcn \
                     + self.conf.alpha * loss_l1 \
                     + self.conf.beta * loss_nuclear
                     #+ self.conf.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj.data, min=0, max=1))

        # evaluate
        self.model.eval()
        normalized_adj = estimator.normalize()
        loss_val, acc_val = self.evaluate(self.val_mask, normalized_adj)

        # save the best model
        if loss_val < self.best_val_loss:
            self.total_time = time.time()-self.start_time
            self.improve = True
            self.best_val_loss = loss_val
            self.result['train'] = acc_train
            self.result['valid'] = acc_val
            improve = '*'
            self.best_graph = estimator.estimated_adj.clone().detach()
            self.weights = deepcopy(self.model.state_dict())

        #print
        print("Epoch {:05d} | Time(s) {:.4f} | Loss(adj) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
            epoch+1, time.time() - t, total_loss.item(), loss_val, acc_val, improve))

    def train(self):
        self.reset()
        self.start_time = time.time()
        for epoch in range(self.conf.n_epochs):
            if self.conf.only_gcn:
                self.train_gcn(epoch)
            else:
                for i in range(int(self.conf.outer_steps)):
                    self.train_adj(epoch)

                for i in range(int(self.conf.inner_steps)):
                    self.train_gcn(epoch)
            if self.improve:
                self.wait = 0
                self.improve = False
            else:
                self.wait += 1
                if self.wait == self.conf.patience:
                    print('Early stop!')
                    break

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        if self.conf.save_graph:
            torch.save(self.best_graph.cpu(), self.graph_loc)
        return self.result

    def evaluate(self, test_mask, normalized_adj):
        self.model.eval()
        self.estimator.eval()
        with torch.no_grad():
            logits = self.model(self.feats, normalized_adj)[-1]
        logits = logits[test_mask]
        labels = self.labels[test_mask]
        loss=F.nll_loss(logits, labels)
        return loss, accuracy(logits, labels)

    def test(self):
        self.model.load_state_dict(self.weights)
        self.estimator.estimated_adj.data.copy_(self.best_graph)
        normalized_adj = self.estimator.normalize()
        return self.evaluate(self.test_mask, normalized_adj)

    def run(self):
        total_runs = self.args.n_runs * self.args.n_splits
        assert self.args.n_splits <= len(split_seeds)
        assert total_runs <= len(train_seeds)
        logger = Logger(runs=total_runs)
        for i in range(self.args.n_splits):
            self.split_data(self.args.data, split_seeds[i])  # split the data
            for j in range(self.args.n_runs):
                k = i * self.args.n_runs + j
                print("Exp {}/{}".format(k, total_runs))
                set_seed(train_seeds[k])
                result = self.train()
                logger.add_result(k, result)
        logger.print_statistics()

    def reset(self):
        self.model = GCN(self.dim_feats, self.conf.n_hidden, self.n_classes, dropout=self.conf.dropout)
        self.model = self.model.to(self.device)
        self.estimator = EstimateAdj(self.adj, symmetric=self.conf.symmetric, device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        self.optimizer_adj = optim.SGD(self.estimator.parameters(), momentum=0.9, lr=self.conf.lr_adj)
        self.optimizer_l1 = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_l1], lr=self.conf.lr_adj, alphas=[self.conf.alpha])
        self.optimizer_nuclear = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_nuclear],
                                     lr=self.conf.lr_adj, alphas=[self.conf.beta])
        self.wait = 0
        self.improve = False
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}


