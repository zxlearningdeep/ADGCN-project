import torch
import os
from models_adgcn import ADGCN_Trainer,DGraph
import warnings
warnings.filterwarnings("ignore")

features = 128
init_features = 172



class args():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.0001
        self.num_hidden = features
        self.init_dim = init_features
        self.num_proj_hidden = features
        self.drop_edge_rate_1 = 0.1
        self.drop_edge_rate_2 = 0.2
        self.drop_feature_rate_1 = 0.1
        self.drop_feature_rate_2 = 0.15
        self.tau = 0.5
        self.train_epochs = 1
        self.test_epochs = 1
        self.weight_decay = 1e-5
        self.alpha_1 = 0.2 # augmentation
        self.alpha_2 = 0.1 # con
        self.alpha_3 = 0.7 # MSE
        self.drop_scheme = 'degree'

        self.device = 'cuda:0'
        self.param = 'local:wikics.json'
        self.verbose = 'train,eval,final'
        self.weak_supervision_k1 = 40
        self.weak_supervision_k0 = 50






args = args()


device = torch.device(args.device)


base_dir = os.path.abspath(r"./datasets/Wikipedia/edges_processed.csv")


DGraph = DGraph(features)

renumbered_edges,renumbered_node_embeddings, old_graph_complexity= DGraph.adj_matrix_renumbered_global(base_dir)
adgcn = ADGCN_Trainer(renumbered_node_embeddings,renumbered_edges, device,args)
adgcn.get_embedding()

del  adgcn
torch.cuda.empty_cache()