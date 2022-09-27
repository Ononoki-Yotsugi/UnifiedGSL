import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from data import load_dataset
from data.split import get_split
from copy import deepcopy
from models.idgl import IDGL, sample_anchors, diff, compute_anchor_adj
import torch
import numpy as np
import time
from utils.utils import normalize, normalize_sp_tensor, accuracy, set_seed, sample_mask
# from dgl.transform import add_reverse_edges, to_simple
from dgl.data.utils import generate_mask_tensor
from utils.logger import Logger
import random

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
        print("Solver Version : [{}]".format("idgl"))
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.normalize = normalize_sp_tensor if self.conf.scalable_run else normalize
        self.prepare_data(args.data)
        self.run_epoch = self._scalable_run_whole_epoch if self.conf.scalable_run else self._run_whole_epoch

    def prepare_data(self, ds_name):
        if "reload_gs" in self.conf and self.conf.reload_gs:
            self.data_raw, g = load_dataset(ds_name, reload_gs=True, graph_fn=self.conf.graph_fn)
        else:
            self.data_raw, g = load_dataset(ds_name)

        self.g = g.int().to(self.device)
        self.g = dgl.remove_self_loop(self.g)  # this operation is aimed to get a adj without self loop

        # ogb is currently not implemented
        # if self.args.data == 'ogbn-arxiv' and "to_symmetric" in self.conf and self.conf.to_symmetric:
        #     self.g = to_simple(add_reverse_edges(self.g))

        self.feats = self.g.ndata['feat']  # 这个feats已经经过归一化了
        self.n_nodes = self.feats.shape[0]
        self.labels = self.g.ndata['label']
        self.dim_feats = self.feats.shape[1]
        self.n_classes = self.data_raw.num_classes
        self.adj = self.g.adj().to(self.device)  # sparse
        self.n_edges = self.g.number_of_edges()
        if self.args.verbose:
            print("""----Data statistics------'
                        #Nodes %d
                        #Edges %d
                        #Classes %d""" %
                  (self.n_nodes, self.n_edges, self.n_classes))
        if self.conf.scalable_run:
            self.normalized_adj = self.normalize(self.adj)
        else:
            self.adj = self.adj.to_dense()
            self.normalized_adj = normalize(self.adj)

        if "lpcls" in self.conf and self.conf.lpcls > 0:
            labs_sp = [self.conf.lpcls for _ in range(self.n_classes)]
            ind_list = list(range(self.train_mask.shape[0]))
            random.shuffle(ind_list)
            for i in ind_list:
                if self.train_mask[i] and labs_sp[self.labels[i]] > 0:
                    labs_sp[self.labels[i]] -= 1
                else:
                    self.train_mask[i] = False
            print(self.train_mask.sum())

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


    def _run_whole_epoch(self, mode='train', debug=False):

        # prepare
        training = mode == 'train'
        if mode == 'train':
            idx = self.train_mask
        elif mode == 'valid':
            idx = self.val_mask
        else:
            idx = self.test_mask
        self.model.train(training)
        network = self.model

        # The first iter
        features = F.dropout(self.feats, self.conf.feat_adj_dropout, training=training)
        init_node_vec = features
        init_adj = self.normalized_adj
        cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner, init_node_vec, self.conf.graph_skip_conn, graph_include_self=self.conf.graph_include_self, init_adj=init_adj)
        # cur_raw_adj是根据输入Z直接产生的adj, cur_adj是前者归一化并和原始adj加权求和的结果
        cur_raw_adj = F.dropout(cur_raw_adj, self.conf.feat_adj_dropout, training=training)
        cur_adj = F.dropout(cur_adj, self.conf.feat_adj_dropout, training=training)
        node_vec, output = network.encoder(init_node_vec, cur_adj)
        score = accuracy(output[idx], self.labels[idx])
        loss1 = F.nll_loss(output[idx], self.labels[idx])
        loss1 += self.get_graph_loss(cur_raw_adj, init_node_vec)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        # the following iters
        if training:
            eps_adj = float(self.conf.eps_adj)
        else:
            eps_adj = float(self.conf.test_eps_adj)
        pre_raw_adj = cur_raw_adj
        pre_adj = cur_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj) and iter_ < self.conf.max_iter:
            iter_ += 1
            pre_adj = cur_adj
            pre_raw_adj = cur_raw_adj
            cur_raw_adj, cur_adj = network.learn_graph(network.graph_learner2, node_vec, self.conf.graph_skip_conn, graph_include_self=self.conf.graph_include_self, init_adj=init_adj)
            update_adj_ratio = self.conf.update_adj_ratio
            cur_adj = update_adj_ratio * cur_adj + (1 - update_adj_ratio) * first_adj   # 这里似乎和论文中有些出入？？
            node_vec, output = network.encoder(init_node_vec, cur_adj, self.conf.gl_dropout)
            score = accuracy(output[idx], self.labels[idx])
            loss += F.nll_loss(output[idx], self.labels[idx])
            loss += self.get_graph_loss(cur_raw_adj, init_node_vec)

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, score

    def _scalable_run_whole_epoch(self, mode='train', debug=False):

        # prepare
        training = mode == 'train'
        if mode == 'train':
            idx = self.train_mask
        elif mode == 'valid':
            idx = self.val_mask
        else:
            idx = self.test_mask
        self.model.train(training)
        network = self.model

        # init
        init_adj = self.normalized_adj
        features = F.dropout(self.feats, self.conf.feat_adj_dropout, training=training)
        init_node_vec = features
        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, self.conf.num_anchors)

        # the first iter
        # Compute n x s node-anchor relationship matrix
        cur_node_anchor_adj = network.learn_graph(network.graph_learner, init_node_vec, anchor_features=init_anchor_vec)
        # Compute s x s anchor graph
        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)
        cur_node_anchor_adj = F.dropout(cur_node_anchor_adj, self.conf.feat_adj_dropout, training=training)
        cur_anchor_adj = F.dropout(cur_anchor_adj, self.conf.feat_adj_dropout, training=training)

        first_init_agg_vec, init_agg_vec, node_vec, output = network.encoder(init_node_vec, init_adj, cur_node_anchor_adj, self.conf.graph_skip_conn)
        anchor_vec = node_vec[sampled_node_idx]
        first_node_anchor_adj, first_anchor_adj = cur_node_anchor_adj, cur_anchor_adj
        score = accuracy(output[idx], self.labels[idx])
        loss1 = F.nll_loss(output[idx], self.labels[idx])
        loss1 += self.get_graph_loss(cur_anchor_adj, init_anchor_vec)

        # the following iters
        if training:
            eps_adj = float(self.conf.eps_adj)
        else:
            eps_adj = float(self.conf.test_eps_adj)

        pre_node_anchor_adj = cur_node_anchor_adj
        loss = 0
        iter_ = 0
        while (iter_ == 0 or diff(cur_node_anchor_adj, pre_node_anchor_adj, cur_node_anchor_adj).item() > eps_adj) and iter_ < self.conf.max_iter:
            iter_ += 1
            pre_node_anchor_adj = cur_node_anchor_adj
            # Compute n x s node-anchor relationship matrix
            cur_node_anchor_adj = network.learn_graph(network.graph_learner2, node_vec, anchor_features=anchor_vec)
            # Compute s x s anchor graph
            cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

            update_adj_ratio = self.conf.update_adj_ratio
            _, _, node_vec, output = network.encoder(init_node_vec, init_adj, cur_node_anchor_adj, self.conf.graph_skip_conn,
                                           first=False, first_init_agg_vec=first_init_agg_vec, init_agg_vec=init_agg_vec, update_adj_ratio=update_adj_ratio,
                                           dropout=self.conf.gl_dropout, first_node_anchor_adj=first_node_anchor_adj)
            anchor_vec = node_vec[sampled_node_idx]

            score = accuracy(output[idx], self.labels[idx])
            loss += F.nll_loss(output[idx], self.labels[idx])

            loss += self.get_graph_loss(cur_anchor_adj, init_anchor_vec)

        if iter_ > 0:
            loss = loss / iter_ + loss1
        else:
            loss = loss1
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, score

    def get_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.conf.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = torch.ones(out_adj.size(-1)).to(self.device)
        graph_loss += -self.conf.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        graph_loss += self.conf.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def train(self):
        self.reset()
        self.start_time = time.time()
        wait = 0

        for epoch in range(self.conf.max_epochs):
            t = time.time()
            improve = ''

            # training phase
            loss_train, acc_train = self.run_epoch(mode='train', debug=self.args.debug)

            # validation phase
            with torch.no_grad():
                loss_val, acc_val = self.run_epoch(mode='valid', debug=self.args.debug)

            if loss_val < self.best_val_loss:
                wait = 0
                self.total_time = time.time()-self.start_time
                self.best_val_loss = loss_val
                self.weights = deepcopy(self.model.state_dict())
                self.result['train'] = acc_train
                self.result['valid'] = acc_val
                improve = '*'
            else:
                wait += 1
                if wait == self.conf.patience:
                    print('Early stop!')
                    break

            # print
            print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch + 1, time.time() - t, loss_train.item(), acc_train, loss_val, acc_val, improve))

        # test
        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        self.model.load_state_dict(self.weights)
        with torch.no_grad():
            loss_test, acc_test = self.run_epoch(mode='test', debug=self.args.debug)
        self.result['test']=acc_test
        print(acc_test)
        return self.result

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
        self.model = IDGL(self.conf, self.dim_feats, self.n_classes)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        self.start_time = None
        self.total_time = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.result = {'train': 0, 'valid': 0, 'test': 0}

