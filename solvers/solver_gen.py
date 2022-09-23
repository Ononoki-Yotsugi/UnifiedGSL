import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.data.utils import generate_mask_tensor
from data import load_dataset
from data.split import get_split
from copy import deepcopy
from models.gen import EstimateAdj, prob_to_adj
from models.GCN3 import GCN
import torch
import numpy as np
import time
from utils.utils import accuracy, normalize_sp_tensor, get_node_homophily, set_seed, sample_mask
from utils.logger import Logger
from sklearn.metrics.pairwise import cosine_similarity as cos
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
        print("Solver Version : [{}]".format("gen"))
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.normalize = normalize_sp_tensor
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
        self.g = dgl.remove_self_loop(self.g)   # this operation is aimed to get a adj without self loop
        self.feats = self.g.ndata['feat']   #这个feats已经经过归一化了
        self.n_nodes = self.feats.shape[0]
        self.labels = self.g.ndata['label']
        self.dim_feats = self.feats.shape[1]
        self.n_classes = self.data_raw.num_classes
        self.adj = self.g.adj().to(self.device)   # sparse
        self.homophily = get_node_homophily(self.labels.cpu().numpy(), self.adj.to_dense().cpu().numpy())
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

    def knn(self, feature):
        # Generate a knn graph for input feature matrix. Note that the graph contains self loop.
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
        dist = cos(feature.detach().cpu().numpy())
        col = np.argpartition(dist, -(self.conf.k + 1), axis=1)[:, -(self.conf.k + 1):].flatten()
        adj[np.arange(self.n_nodes).repeat(self.conf.k + 1), col] = 1
        return adj

    def train_gcn(self, iter, adj):
        if self.args.debug:
            print('==== Iteration {:04d} ===='.format(iter+1))
        t = time.time()
        improve_1 = ''
        best_loss_val = 10
        best_acc_val = 0
        normalized_adj = self.normalize(adj)
        for epoch in range(self.conf.n_epochs):
            improve_2 = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()

            # forward and backward
            hidden_output, output = self.model(self.feats, normalized_adj)
            loss_train = F.cross_entropy(output[self.train_mask], self.labels[self.train_mask])
            acc_train = accuracy(output[self.train_mask], self.labels[self.train_mask])
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val, hidden_output, output = self.evaluate(self.val_mask, normalized_adj)

            # save
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                improve_2 = '*'
                if acc_val > self.result['valid']:
                    self.total_time = time.time()-self.start_time
                    improve_1 = '*'
                    self.best_val_loss = loss_val
                    self.result['valid'] = acc_val
                    self.result['train'] = acc_train
                    self.best_iter = iter+1
                    self.best_graph = adj.clone().detach()
                    self.hidden_output = hidden_output
                    self.output = F.softmax(output, dim=1)
                    self.weights = deepcopy(self.model.state_dict())

            # print
            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch+1, time.time() -t0, loss_train.item(), acc_train, loss_val, acc_val, improve_2))

        print('Iteration {:04d} | Time(s) {:.4f} | Loss(val):{:.4f} | Acc(val):{:.4f} | {}'.format(iter+1,time.time()-t, best_loss_val, best_acc_val, improve_1))

    def structure_learning(self, iter):
        t=time.time()
        self.estimator.reset_obs()
        self.estimator.update_obs(self.knn(self.feats))   # 2
        self.estimator.update_obs(self.knn(self.hidden_output))   # 3
        self.estimator.update_obs(self.knn(self.output))   # 4
        alpha, beta, O, Q, iterations = self.estimator.EM(self.output.max(1)[1].detach().cpu().numpy(), self.conf.tolerance)
        adj = prob_to_adj(Q, self.conf.threshold).clone().detach().to(self.device)
        print('Iteration {:04d} | Time(s) {:.4f} | EM step {:04d}'.format(iter+1,time.time()-t,self.estimator.count))
        return adj

    def train(self):
        self.reset()
        self.start_time = time.time()
        adj = self.adj

        for iter in range(self.conf.n_iters):
            self.train_gcn(iter, adj)
            adj = self.structure_learning(iter)

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        loss_test, acc_test, _, _ = self.test()
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        if self.conf.save_graph:
            torch.save(self.best_graph.cpu(), self.graph_loc)
        return self.result

    def evaluate(self, test_mask, normalized_adj):
        self.model.eval()
        with torch.no_grad():
            hidden_output, output = self.model(self.feats, normalized_adj)
        logits = output[test_mask]
        labels = self.labels[test_mask]
        loss=F.cross_entropy(logits, labels)
        return loss, accuracy(logits, labels), hidden_output, output

    def test(self):
        self.model.load_state_dict(self.weights)
        normalized_adj = self.normalize(self.best_graph)
        return self.evaluate(self.test_mask, normalized_adj)

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
        self.model = GCN(self.dim_feats, self.conf.n_hidden, self.n_classes, dropout=self.conf.dropout,
                         input_dropout=self.conf.input_dropout)
        self.model = self.model.to(self.device)
        self.estimator = EstimateAdj(self.n_classes, self.adj, self.train_mask, self.labels, self.homophily)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.best_iter = 0
        self.hidden_output = None
        self.output = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}


