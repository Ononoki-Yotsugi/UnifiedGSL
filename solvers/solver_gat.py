from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import torch

from data import load_dataset
from data.split import get_split
from models.gat import GAT
import numpy as np
import time
import random
from utils.utils import accuracy, AverageMeter, sample_mask, set_seed
from dgl.transforms import AddReverse, ToSimple        # dgl0.9
# from dgl.transform import add_reverse_edges, to_simple
from collections import Iterable
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
        print("Solver Version : [{}]".format("GAT"))
        self.args = args
        self.conf = conf
        self.device = torch.device('cuda')
        self.prepare_data(args.data)

        if isinstance(self.conf.num_heads, Iterable):
            self.heads = self.conf.num_heads + [self.conf.num_out_heads]
        else:
            self.heads = [self.conf.num_heads] * (self.conf.n_layers-1) + [self.conf.num_out_heads]

        self.use_scheduler = False                    
        if "scheduler" in self.conf:
            self.use_scheduler = True
            if self.conf.scheduler == "MultiStep":
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=self.optim,
                    milestones=self.conf.milestones,
                    gamma=self.conf.gamma
                )
            else:
                raise ValueError("Scheduler Type {} Has Not Been Supported.".format(self.conf.scheduler))
        
    def prepare_data(self, ds_name):
        if "reload_gs" in self.conf and self.conf.reload_gs:
            self.data_raw, self.g = load_dataset(ds_name, reload_gs=True, graph_fn=self.conf.graph_fn)
        else:
            self.data_raw, self.g = load_dataset(ds_name)

        if "to_symmetric" in self.conf and self.conf.to_symmetric:
            trans1 = AddReverse()
            trans2 = ToSimple()
            self.g = trans2(trans1(self.g))  # dgl0.9
            # self.g = to_simple(add_reverse_edges(self.g))

        if self.conf.self_loop:
            self.g = dgl.remove_self_loop(self.g)
            self.g = dgl.add_self_loop(self.g)

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

        if not self.conf.data_cpu:
            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)

        if self.conf.batch_size > 0:
            self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.conf.n_layers)
            self.dataloader = dgl.dataloading.NodeDataLoader(
                g=self.g,
                nids=self.train_nid,
                block_sampler=self.sampler,
                device='cpu',
                batch_size=self.conf.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.conf.num_workers
            )

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
        self.train_nid = torch.nonzero(self.train_mask, as_tuple=True)[0]
        self.val_nid = torch.nonzero(self.val_mask, as_tuple=True)[0]
        self.test_nid = torch.nonzero(self.test_mask, as_tuple=True)[0]

        if self.args.verbose:
            print("""----Split statistics------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (len(self.train_mask), len(self.val_mask), len(self.test_mask)))

    def train(self):
        # 这个还有待改动以适应ogbn-arxiv
        if self.conf.batch_size <= 0:
            result = self.train_non_batch()
            return result
        
        avg = 0
        iter_tput = []
        losses = AverageMeter()

        for epoch in range(self.conf.n_epochs):

            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            tic_step = time.time()
            losses.reset()

            for step, (input_nodes, seeds, blocks) in enumerate(self.dataloader):
                # Load the input features as well as output labels
                batch_inputs = self.feats[input_nodes].to(self.device)
                batch_labels = self.labels[seeds].to(self.device)
                blocks = [block.int().to(self.device) for block in blocks]

                # Compute loss and prediction
                batch_pred = self.model(blocks, batch_inputs)
                loss = self.loss_fcn(batch_pred, batch_labels)
                self.optim.zero_grad()
                loss.backward()
                losses.update(loss.item())

                self.optim.step()

                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % self.conf.log_every == 0:
                    acc = accuracy(batch_pred, batch_labels)
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, losses.avg, acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
                tic_step = time.time()

            if self.use_scheduler:
                self.scheduler.step()
                print("Current Learning Rate is {}".format(self.optim.param_groups[0]['lr']))
            
            toc = time.time()
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % self.conf.eval_every == 0 and epoch != 0:
                val_loss, val_acc = self.evaluate(self.val_mask)
                print('Eval Acc {:.4f}'.format(val_acc))

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.weights = deepcopy(self.model.state_dict())

        test_loss, test_acc = self.test()
        print('Test Acc on Best Val: {:.4f}'.format(test_acc))
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    
    def train_non_batch(self):

        total_time = 0
        best_val_loss = 10
        weights = None
        result = {'train': 0, 'valid': 0, 'test': 0}
        best_acc_val = 0
        start_time = time.time()

        model = GAT(g=self.g, num_layers=self.conf.n_layers - 1, in_dim=self.dim_feats, num_hidden=self.conf.n_hidden,
                    num_classes=self.n_classes, heads=self.heads, activation=F.elu, feat_drop=self.conf.feat_drop,
                    attn_drop=self.conf.attn_drop, negative_slope=0.2, residual=self.conf.residual).to(self.device)
        loss_fcn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)

        self.feats = self.feats.to(self.device)   # 在这里更改solver中的变量应该是不被允许的，这里不会造成负面结果所以例外
        self.labels = self.labels.to(self.device)
        self.g = self.g.int().to(self.device)

        speed_stata = AverageMeter()
        for epoch in range(self.conf.n_epochs):
            improve = ''
            t0 = time.time()

            # train
            model.train()
            pred = model(self.g, self.feats)
            loss_train = loss_fcn(pred[self.train_mask], self.labels[self.train_mask])

            optim.zero_grad()
            loss_train.backward()
            optim.step()

            acc_train = accuracy(pred[self.train_mask], self.labels[self.train_mask])
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

            # eval
            model.eval()
            pred_val = model([self.g], self.feats)
            loss_val = loss_fcn(pred_val[self.val_mask], self.labels[self.val_mask]).item()
            acc_val = accuracy(pred_val[self.val_mask], self.labels[self.val_mask])

            # print
            if self.args.debug:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                    epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve))
            
            if loss_val < best_val_loss:
                improve = '*'
                best_val_loss = loss_val
                weights = deepcopy(model.state_dict())
                result['valid'] = acc_val
                result['train'] = acc_train
                total_time = time.time()-start_time

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(total_time))
        loss_test, acc_test = self.test(model, weights)
        result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return result

    def evaluate(self, model, test_mask):
        model.eval()
        with torch.no_grad():
            if self.conf.batch_size < 0:
                logits = model(self.g, self.feats).cpu()
            else:
                logits = model.inference(self.g, self.feats, self.device, self.conf.batch_size, self.conf.num_workers)
        model.train()
        logits = logits[test_mask]
        labels = self.labels[test_mask].cpu()
        loss = F.cross_entropy(logits, labels)
        return loss, accuracy(logits, labels)

    def test(self, model, weights):
        model.load_state_dict(weights)
        return self.evaluate(model, self.test_mask)

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
