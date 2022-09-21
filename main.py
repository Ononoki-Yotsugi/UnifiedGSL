import os
import torch
import argparse
import numpy as np
import ruamel.yaml as yaml
import random
from utils.logger import Logger
from utils.utils import set_seed

def main(args):

    print(args)

    if args.solver == "gcn":
        from solvers.solver_gcn import Solver
    elif args.solver == 'gcndense':
        from solvers.solver_gcndense import Solver
    elif args.solver == 'ProGNN':
        from solvers.solver_ProGNN import Solver
    elif args.solver == 'GEN':
        from solvers.solver_GEN import Solver
    elif args.solver == 'GeomGCN':
        from solvers.solver_GeomGCN import Solver
    elif args.solver == 'GraphSAGE':
        from solvers.solver_sage import Solver
    elif args.solver == 'GAug':
        from solvers.solver_GAug import Solver
    elif args.solver == 'GAT':
        from solvers.solver_gat import Solver
    elif args.solver == 'IDGL':
        from solvers.solver_IDGL import Solver
    elif args.solver == 'SGSL':
        from solvers.solver_sgsl import Solver
    elif args.solver =='Dyn':
        from solvers.solver_gcndyn import Solver
    elif args.solver == 'CelGSL':
        from solvers.solver_CelGSL import Solver
    elif args.solver == 'LT':
        from solvers.solver_LT import Solver
    elif args.solver == 'LTGSL':
        from solvers.solver_LTGSL import Solver
    else:
        raise ValueError("Not Recognized Solver Version {}".format(args.solver_version))

    if args.config != '':
        conf = open(args.config, "r").read()
        conf = yaml.safe_load(conf)
        conf = argparse.Namespace(**conf)
        print(conf)
    else:
        conf = None

    solver = Solver(args, conf)
    solver.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora',
                        choices=['cora', 'pubmed', 'citeseer', 'raw_cora', 'ogbn-arxiv', 'amazoncom', 'amazonpho',
                                 'coauthorcs', 'coauthorph', 'wikics'], help='dataset')
    parser.add_argument('--solver', type=str, default='gcn',
                        choices=['gcn', 'gcndense', 'GEN', 'GeomGCN', 'ProGNN', 'GraphSAGE', 'GAug', 'GAT', 'IDGL', 'SGSL', 'LT', 'LTGSL'], help="The version of solver")
    parser.add_argument('--config', type=str, default='configs/gcn/gcn_template.yaml', help="Config file used for specific model training.")
    parser.add_argument('--n_runs', type=int, default=1,
                        help="number of exps per data split")
    parser.add_argument('--n_splits', type=int, default=1,
                        help="number of different data splits (For citation datasets you get the same split "
                             "unless you have re_split=true in the config file)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    main(args)
