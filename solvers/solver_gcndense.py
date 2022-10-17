import torch.nn.functional as F
import torch
from models.GCN3 import GCN
from utils.utils import normalize_sp_tensor, accuracy, set_seed, sample_mask
from copy import deepcopy
import time
from .solver import BaseSolver


class Solver(BaseSolver):
    def __init__(self, args, conf):
        super().__init__(args, conf)
        print("Solver Version : [{}]".format("gcndense"))
        self.normalize = normalize_sp_tensor

    def train(self):
        model = GCN(self.dim_feats, self.conf.n_hidden, self.n_classes, dropout=self.conf.dropout,
                    input_dropout=self.conf.input_dropout, norm=self.conf.norm).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        total_time = 0
        best_val_loss = 10
        weights = None
        result = {'train': 0, 'valid': 0, 'test': 0}
        normalized_adj = self.normalize(self.adj)
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






