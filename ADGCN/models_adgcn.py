import numpy as np
import torch
import utils as u
import nni
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE,GraphConvolution
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality

from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score


def data_normal_2d(orign_data):
    dim = 0
    d_min = torch.min(orign_data,dim=dim)[0]
    for idx,j in enumerate(d_min):
        if j < 0:
            orign_data[:,idx] += torch.abs(d_min[idx])
    d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    index = torch.nonzero(dst == 0)
    dst[index] = 1.0


    d_min = d_min.unsqueeze(0)
    dst = dst.unsqueeze(0)

    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    norm_data = (norm_data - 0.5).true_divide(0.5)

    return norm_data



class DGraph():
    def __init__(self, features_num):

        self.node_set = set()
        self.node_map = {}
        self.reverse_map = {}
        self.node_embeddings = {}
        self.node_feature_num = features_num


    def get_graph_complexity(self, old_graph, node_list):

        return (len(set((node_list))) + len(old_graph[0]) / 2) / 2


    def adj_matrix_renumbered_global(self, base_dir):

        features, old_graph_edges, old_graph_nodes = u.edges_loader_global(base_dir)
        old_graph_complexity = self.get_graph_complexity(old_graph_edges, old_graph_nodes)

        self.node_set.update(old_graph_nodes)
        my_list = [x for x in range(len(old_graph_nodes))]
        mapping = dict(zip(my_list, old_graph_nodes))
        reverse_map = {v: k for k, v in mapping.items()}

        node_label = features[:, 3].float().clone().detach()
        index = features[:, 0].int().clone().detach()
        index_item = features[:, 1].int().clone().detach()
        features = features[:,4:]
        features = data_normal_2d(features)
        features_size = features.size(1)
        self.init_feature_size = features_size
        renumbered_edges = torch.zeros_like(old_graph_edges)
        renumbered_node_embeddings = torch.zeros(size=(
            old_graph_nodes.shape[0], features_size + 2))


        renumbered_edges[0] = torch.tensor([reverse_map[x.item()] for x in old_graph_edges[0]])
        renumbered_edges[1] = torch.tensor([reverse_map[x.item()] for x in old_graph_edges[1]])

        self.node_map = mapping
        self.reverse_map = reverse_map


        for key in mapping.keys():

            node = np.array(mapping[key], dtype=int)

            new_index = torch.nonzero(index == node.item())
            if(new_index.numel() == 0):
                new_index = torch.nonzero(index_item == node.item())
                label = -1
            else:

                label = node_label[new_index[-1].item()]

            new_index = new_index.view(-1)
            # print(new_index, node.item())
            new_embedding = torch.mean(features[new_index], dim=0 , keepdim=True)
            # print(new_embedding, new_embedding.shape)
            new_embedding = torch.cat((new_embedding, torch.zeros(size=[1, 1], dtype=torch.float)),
                                      dim=1)
            new_embedding = torch.cat((new_embedding, torch.tensor([[label]], dtype=torch.float)),
                                      dim=1)
            # print(new_embedding.shape)
            renumbered_node_embeddings[key] = new_embedding

        return renumbered_edges, renumbered_node_embeddings.detach(), old_graph_complexity



class Balanced_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss_1, loss_2,loss_3 ,alpha1, alpha2):

        loss = loss_1 + alpha1 * loss_2 + loss_3 * alpha2

        loss = loss.requires_grad_()

        return loss

        
class ADGCN_Trainer():
    def __init__(self,node_features, edges, device, args):
        super().__init__()
        self.node_num = node_features.shape[0]
        self.device = device
        self.ecols = u.Namespace({'FromNodeId': 0,
                                      'ToNodeId': 1,
                                      'Weight': 2,
                                      'TimeStep': 3
                                  })
        self.args = args
        default_param = {
            'learning_rate': 0.005,
            'num_hidden': 128,
            'init_dim' : 172,
            'num_proj_hidden': 32,
            'drop_edge_rate_1': 0.3,
            'drop_edge_rate_2': 0.4,
            'drop_feature_rate_1': 0.1,
            'drop_feature_rate_2': 0.0,
            'tau': 0.1,
            'train_epochs':100,
            'test_epochs': 30,
            'weight_decay': 1e-5,
            'drop_scheme': 'pr',
            'weak_supervision_k1': 5,
            'weak_supervision_k0': 100,
            'alpha_1': 0.004,
            'alpha_2': 0.0001,
            'alpha_3': 0.0001,
        }
        param_keys = default_param.keys()

        sp = SimpleParam(default=default_param)
        self.param = sp(source=args.param, preprocess='nni')
        for key in param_keys: # 把参数重置为之前设定的值
            if getattr(self.args , key) is not None:
                self.param[ key] = getattr(self.args, key)

        feature_all_nums = node_features.size(1)

        self.weak_supervision_k1 = self.param['weak_supervision_k1']
        self.weak_supervision_k0 = self.param['weak_supervision_k0']

        label = node_features[:,-1]
        self.label = label.long()
        edges = edges.long()
        self.data = Data(x=node_features[:,:feature_all_nums-2],y =self.label ,edge_index=edges).to(device)

        self.autoencoder = torch.nn.Sequential(nn.Linear(feature_all_nums-2, self.param['num_hidden']),
                                               nn.ELU()).to(device)
        self.decoder = torch.nn.Sequential(nn.Linear(self.param['num_hidden'], self.param['num_hidden']),
                                           nn.ELU(),
                                           nn.Linear(self.param['num_hidden'], feature_all_nums-2),
                                           nn.Tanh()).to(device)


        self.encoder = Encoder(self.param['num_hidden']).to(device)
        self.model = GRACE(self.encoder, feature_all_nums-2, self.param['num_hidden'], self.param['num_proj_hidden'], self.param['tau']).to(device)
        # self.model.load_state_dict(torch.load("./final")['gca'])
        self.classifier_hidden = self.param['num_hidden'] * 2
        self.classifier = nn.Sequential(nn.Linear(self.param['num_hidden'], self.classifier_hidden),
                                        nn.ELU(),
                                        nn.Linear(self.classifier_hidden, 2)).to(device)
        self.balanced_loss = Balanced_loss().to(device)

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.criterion_2 = nn.MSELoss().to(device)
        self.optimizer = torch.optim.NAdam([
            {'params': self.autoencoder.parameters(), 'lr': self.param['learning_rate']},
            {'params': self.decoder.parameters(), 'lr': self.param['learning_rate']},
            {'params': self.model.parameters(), 'lr': self.param['learning_rate']},
            {'params': self.classifier.parameters(), 'lr': 5*1e-3},
            {'params': self.balanced_loss.parameters(), 'lr' : self.param['learning_rate']}
            ], weight_decay=0.00001)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)


        if self.param['drop_scheme'] == 'degree':
            self.drop_weights = degree_drop_weights(self.data.edge_index).to(device)
        elif self.param['drop_scheme'] == 'pr':
            self.drop_weights = pr_drop_weights(self.data.edge_index, aggr='sink', k=200).to(device)
        elif self.param['drop_scheme'] == 'evc':
            self.drop_weights = evc_drop_weights(self.data).to(device)
        else:
            self.drop_weights = None

        if self.param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(self.data.edge_index)
            node_deg = degree(edge_index_[1])
            self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_deg).to(device)


        elif self.param['drop_scheme'] == 'pr':
            node_pr = compute_pr(self.data.edge_index)
            self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_pr).to(device)

        elif self.param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(self.data)
            self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_evc).to(device)

        else:
            self.feature_weights = torch.ones((self.data.x.size(1),)).to(device)
            
    def train(self):
        self.autoencoder.train()
        self.decoder.train()
        self.model.train()
        self.classifier.train()
        self.optimizer.zero_grad()
        acc = 0.0

        def drop_edge(idx: int):
            global drop_weights

            if self.param['drop_scheme'] == 'uniform':
                return dropout_adj(self.data.edge_index, p= self.param[f'drop_edge_rate_{idx}'])[0]
            elif self.param['drop_scheme'] in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(self.data.edge_index, self.drop_weights, p= self.param[f'drop_edge_rate_{idx}'], threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: { self.param["drop_scheme"]}')

        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)
        x_1 = drop_feature(self.data.x, self.param['drop_feature_rate_1'])
        x_2 = drop_feature(self.data.x, self.param['drop_feature_rate_2'])


        if self.param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(self.data.x, self.feature_weights, self.param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted_2(self.data.x, self.feature_weights, self.param['drop_feature_rate_2'])

        z1 = self.model(self.autoencoder(x_1), edge_index_1)
        z2 = self.model(self.autoencoder(x_2), edge_index_2)

        edge_index = self.data.edge_index
        z_original = self.data.x
        z_encoder = self.autoencoder(z_original)
        z = self.model(z_encoder, edge_index)
        z_decoder = self.decoder(z_encoder)

        loss_contrastive = self.model.loss(z1, z2, batch_size=None)

        label = self.label.view(-1)
        index = torch.nonzero(label != -1)
        index = index.view(-1)
        label = label[index]
        label = label.to(self.device)

        z = z[index]
        z1_weak = z1[index]
        z2_weak = z2[index]
        pred = self.classifier(z)
        pred_1 = self.classifier(z1_weak)
        pred_2 = self.classifier(z2_weak)

        index_select_supervised_1 = torch.nonzero(label == 1).squeeze()
        index_select_supervised_0 = torch.nonzero(label == 0).squeeze()
        idx_1 = torch.randperm(self.weak_supervision_k1, dtype=torch.long).to(self.device)
        idx_2 = torch.randperm(self.weak_supervision_k0, dtype=torch.long).to(self.device)
        idx_1 = index_select_supervised_1[idx_1]
        idx_2 = index_select_supervised_0[idx_2]
        idx_weak_supervision = torch.cat([idx_1, idx_2])

        pred_weak_supervision = pred[idx_weak_supervision]
        pred_weak_supervision_1 = pred_1[idx_weak_supervision]
        pred_weak_supervision_2 = pred_2[idx_weak_supervision]
        label_weak_supervision = label[idx_weak_supervision]

        loss = self.criterion(pred_weak_supervision, label_weak_supervision)
        loss = loss + self.param['alpha_1'] * (self.criterion(pred_weak_supervision_1, label_weak_supervision) +
                                               self.criterion(pred_weak_supervision_2, label_weak_supervision))
        loss_mse = self.criterion_2(z_decoder,z_original)
        loss_all = self.balanced_loss(loss, loss_contrastive, loss_mse, self.param['alpha_2'], self.param['alpha_3'])



        loss_all.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step()

        pred_1_index = torch.argmax(pred, 1)
        acc += (pred_1_index == label).sum().item()
        acc = acc / (1 * len(label))

        label = label.cpu().detach().numpy()

        pred_1 = F.softmax(pred, dim=1)
        pred_1 = pred_1[:, 1]
        pred_1 = pred_1.cpu().detach().numpy()
        auc = 1 * roc_auc_score(label, pred_1)

        return loss_all.item(), acc, auc, loss.item(), loss_contrastive.item(), loss_mse.item()

    def test(self, final=False):
        self.model.eval()
        self.autoencoder.eval()
        z = self.model(self.autoencoder(self.data.x), self.data.edge_index)


        evaluator = MulticlassEvaluator()
        auc = log_regression(z, self.data, evaluator, split='preloaded', num_epochs=1500)['auc']
        use_nni = True
        if final and use_nni:
            nni.report_final_result(auc)
            return auc,z
        elif use_nni:
            nni.report_intermediate_result(auc)

        return auc

    def final_test(self, epochs):
        best_all_auc = 0.0
        best_train_auc = []
        best_val_auc = []
        best_test_auc = []
        best_train_loss = []
        flag = 0
        for epoch in range(1, epochs + 1):
            self.model.eval()
            self.autoencoder.eval()
            self.decoder.eval()
            self.classifier.eval()
            z = self.model(self.autoencoder(self.data.x), self.data.edge_index)

            evaluator = MulticlassEvaluator()
            all = log_regression(z, self.data, evaluator, split='preloaded', num_epochs=3000)
            best_auc = all['auc']
            train_auc = all['train_auc']
            val_auc = all['val_auc']
            test_auc = all['test_auc']
            split = all['split']
            train_loss = all['train_loss']

            use_nni = True

            if use_nni:
                nni.report_intermediate_result(best_auc)

            if best_auc > best_all_auc:
                flag = 1
                best_all_auc = best_auc
                best_train_auc = train_auc
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_train_loss = train_loss
                split = split
                print('best auc:', best_all_auc)
        print('Best auc on all epoch:', best_all_auc)


    def get_embedding(self):
        log = self.args.verbose.split(',')
        label = self.label

        label_user_idx = torch.nonzero(label >= 0).squeeze()
        label_user = label[label_user_idx]
        print('num of label 1:', sum(label_user).item(), 'num of label 0:', len(label_user) - sum(label_user).item())
        print('rate of weak label 1:', self.weak_supervision_k1 / sum(label_user).item(),
              'rate of weak label 0:', self.weak_supervision_k0 / (len(label_user) - sum(label_user).item()),
              'rate of weak label all:', (self.weak_supervision_k1 + self.weak_supervision_k0) / len(label_user))

        loss_all = []
        acc_all = []
        auc_all = []
        loss_weak_all = []
        loss_con_all = []
        loss_mse_all = []
        best_auc = 0.0

        for epoch in range(1, self.param['train_epochs'] + 1):
            loss, acc, auc, loss_weak, loss_con, loss_mse = self.train()
            loss_all.append(loss)
            acc_all.append(acc)
            auc_all.append(auc)
            loss_weak_all.append(loss_weak)
            loss_con_all.append(loss_con)
            loss_mse_all.append(loss_mse)
            if epoch % 50 == 0:
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

            if epoch % 100 == 0:
                auc_this_test = self.test()
                if auc_this_test > best_auc:
                    best_auc = auc_this_test

                if 'eval' in log:
                    print(f'(E) | Epoch={epoch:04d}, auc = {auc_this_test}')

        self.final_test(self.param['test_epochs'])





