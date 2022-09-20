from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def evaluate(best_model, train_loader, test_loader):

    knn_eval_out = knn_evaluate(best_model, train_loader, test_loader) 

    return knn_eval_out


def knn_evaluate(model, train_loader, test_loader):

    train_data, train_labels = get_representation(model, train_loader)
    test_data, test_labels = get_representation(model, test_loader)

    nn = KNeighborsClassifier(n_neighbors=3)
    nn.fit(train_data, train_labels)

    return (100 * nn.score(test_data, test_labels))


def get_representation(model, test_loader):
    model.eval()

    features = []
    labels = []
    with torch.no_grad():
        for i, (data, cur_labels) in enumerate(test_loader):
            data = data.cuda()
            mu, _ = model.module.encode(data)
            features.append(mu.cpu())

            labels.append(cur_labels)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0).reshape((-1,))

    return features, labels