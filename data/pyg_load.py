'''
Here we provide another way to load data with pyg. This is needed when running grcn because grcn uses the unnormalized
feature, which is not supported in dgl.
'''

from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS


def pyg_load_dataset(name):
    dic = {'cora': 'Cora',
           'citeseer': 'CiteSeer',
           'pubmed': 'PubMed',
           'amazoncom': 'Computers',
           'amazonpho': 'Photo',
           'coauthorcs': 'CS',
           'coauthorph': 'Physics',
           'wikics': 'WikiCS'}
    name = dic[name]

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root='./data/'+name, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root='./data/'+name, name=name)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root='./data/'+name, name=name)
    elif name in ['WikiCS']:
        dataset = WikiCS(root='./data/'+name)
    else:
        exit("wrong dataset")
    return dataset