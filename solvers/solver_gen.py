import torch.nn.functional as F
from copy import deepcopy
from models.gen import EstimateAdj, prob_to_adj
from models.GCN3 import GCN
import torch
import numpy as np
import time
from utils.utils import accuracy, normalize_sp_tensor, get_node_homophily
from sklearn.metrics.pairwise import cosine_similarity as cos
from .solver import BaseSolver


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("gen"))
        self.normalize = normalize_sp_tensor
        self.homophily = get_node_homophily(self.labels.cpu().numpy(), self.adj.to_dense().cpu().numpy())

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


