from typing import Optional
from torch.nn import functional as F
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import random_split
from pGRACE.model import LogReg
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR


def get_idx_split(y, split='preloaded'):

    num_nodes = y.size(0)
    if split == 'preloaded':


        x = torch.tensor(range(num_nodes)).unsqueeze(dim=1)
        x = np.array(x)
        y = np.array(y.cpu())
        X_train, X_validate_test, _, y_validate_test = train_test_split(x, y, test_size=0.5, stratify=y)

        X_valid, X_test, _, _ = train_test_split(X_validate_test, y_validate_test, test_size=0.5, stratify=y_validate_test)

        train_mask = torch.zeros((num_nodes,)).to(torch.bool)
        test_mask = torch.zeros((num_nodes,)).to(torch.bool)
        val_mask = torch.zeros((num_nodes,)).to(torch.bool)

        train_mask[X_train.ravel()] = True
        test_mask[X_test.ravel()] = True
        val_mask[X_valid.ravel()] = True

        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 4000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    test_device = z.device if test_device is None else test_device
    z = z.detach()

    y = dataset.y.view(-1)
    index = torch.nonzero(y != -1)
    index = index.view(-1)
    y = y[index]
    z = z[index]
    dataset_new = dataset.clone()
    dataset_new.x = z
    dataset_new.y = y

    split = get_idx_split(y)
    split = {k: v.to(test_device) for k, v in split.items()}

    z = z.to(test_device)
    num_hidden = z.size(1)
    classifier_hidden = num_hidden * 2
    y = y.to(test_device)
    n0 = torch.sum(y[split['train']] == 0)
    n1 = torch.sum(y[split['train']] == 1)
    n = len(y[split['train']])
    alpha0 = n1 / n
    alpha1 = n0 / n
    alpha = torch.tensor([alpha0, alpha1], dtype=torch.float).to(test_device)

    # num_classes = dataset_new.y.max().item() + 1
    num_classes = 2

    classifier = nn.Sequential(nn.Linear(num_hidden, classifier_hidden),
                               nn.ELU(),
                               nn.Linear(classifier_hidden, num_classes)).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.0002, weight_decay=0.000005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)


    loss_fun = nn.CrossEntropyLoss(weight=alpha)

    best_test_auc = 0
    best_test_recall = 0
    best_test_pre = 0
    best_test_f1 = 0
    best_epoch = 0

    auc_train = []
    recall_train = []
    precision_train = []

    auc_val = []
    recall_val = []
    precision_val = []

    auc_test = []
    recall_test = []
    precision_test = []

    loss_train = []
    f1_train = []
    f1_test = []
    f1_val = []

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(z[split['train']])
        loss = loss_fun(output, y[split['train']].long())
        loss_train.append(loss.item())

        train_all = evaluator.eval({
            'y_true': y[split['train']].view(-1, 1),
            'y_pred': F.softmax(classifier(z[split['train']]), dim=1)
        })['auc']
        train_auc = train_all[0]
        train_recall = train_all[1]
        train_pre = train_all[2]
        train_f1 = train_all[3]
        auc_train.append(train_auc)
        recall_train.append(train_recall)
        precision_train.append(train_pre)
        f1_train.append(train_f1)

        loss.backward()
        optimizer.step()
        scheduler.step()

        test_all = evaluator.eval({
            'y_true': y[split['test']].view(-1, 1),
            'y_pred': F.softmax(classifier(z[split['test']]), dim=1)
        })['auc']

        test_auc = test_all[0]
        test_recall = test_all[1]
        test_pre = test_all[2]
        test_f1 = test_all[3]
        auc_test.append(test_auc)
        recall_test.append(test_recall)
        precision_test.append(test_pre)
        f1_test.append(test_f1)

        val_all = evaluator.eval({
            'y_true': y[split['val']].view(-1, 1),
            'y_pred': F.softmax(classifier(z[split['val']]), dim=1)
        })['auc']

        val_auc = val_all[0]
        val_recall = val_all[1]
        val_pre = val_all[2]
        val_f1 = val_all[3]
        auc_val.append(val_auc)
        recall_val.append(val_recall)
        precision_val.append(val_pre)
        f1_val.append(val_f1)

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_recall = test_recall
            best_test_pre = test_pre
            best_test_f1 = test_f1
            best_epoch = epoch

    print(
        f'test auc {best_test_auc}, test recall {best_test_recall}, test pre {best_test_pre}, '
        f'test f1 {best_test_f1}')
    return {'auc': best_test_auc, 'train_auc': auc_train, 'val_auc': auc_val, 'test_auc': auc_test,
            'train_loss': loss_train, 'split': split}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.cpu().detach().numpy()
        y_pred_auc = y_pred[:, 1]
        y_pred_auc = y_pred_auc.cpu().detach().numpy()

        auc = roc_auc_score(y_true, y_pred_auc)

        y_pred = y_pred.argmax(-1).view(-1, 1).cpu().detach().numpy()
        outputs = y_pred

        precision = precision_score(y_true, outputs, average='weighted')

        recall = recall_score(y_true, outputs, average='weighted')

        f1score = f1_score(y_true, outputs, average='weighted')

        return [auc.item(), recall.item(), precision.item(), f1score.item()]

    def eval(self, res):
        return {'auc': self._eval(**res)}

