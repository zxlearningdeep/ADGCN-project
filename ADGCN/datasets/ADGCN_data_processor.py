import datasets.utils as u
import os
import torch
import numpy as np


class args():
    def __init__(self):
        super().__init__()
        self.data = 'Wikipedia'

        self.datasetotc_args = u.Namespace({
            'folder': './datasets/',
            'edges_file': 'Wiki.csv',
            'aggr_time': 86400,  # one day in seconds
            'feats_per_node': 172

        })


def load_edges(dataset_args):
    file = os.path.join(dataset_args.folder, dataset_args.edges_file)
    with open(file) as f:
        lines = f.read().splitlines()
    edges = [[int(float(r)) for r in row.split(',')] for row in lines]
    edges = torch.tensor(edges, dtype=torch.long)
    return edges



args = args()
edges = load_edges(args.datasetotc_args)
ecols = u.Namespace({'FromNodeId': 0,
                     'ToNodeId': 1,
                     'Weight': 2,
                     'TimeStep': 3
                     })
timesteps = u.aggregate_by_time(edges[:, ecols.TimeStep], args.datasetotc_args.aggr_time)
max_time = timesteps.max()
min_time = timesteps.min()
edges[:, ecols.TimeStep] = timesteps

np.savetxt('./datasets/otc_edges_processed.csv', edges.numpy(), fmt='%.20f', delimiter=',')

num_nodes = edges[:, [ecols.FromNodeId,
                      ecols.ToNodeId]].unique().size(0)

edge_index = edges[:, ecols.FromNodeId, ecols.ToNodeId].t()

value_map = {el: i for i, el in enumerate(set(edges[:, 0]))}  # mapping

child2 = [value_map[el] for el in edges[:, 0]]

child2[0]

edges[:, 0] = child2

edges[:, 1] = [value_map[el] for el in edges[:, 1]]

np.savetxt('./datasets/otc_edges_processed.csv', edges, fmt='%.20f', delimiter=',')