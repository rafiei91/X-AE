from __future__ import print_function
import torch
import shutil
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier

def get_dataset(seed=1, traindir='./'):
    np.random.seed(seed)
    
    trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(),
            transforms.RandomRotation(180),
            transforms.RandomRotation(degrees=[45, -45]),
            transforms.RandomRotation(degrees=[90, -90]),
            ]) 

    train_loader = datasets.ImageFolder(
        traindir,  
        transforms.Compose([
            transforms.RandomResizedCrop(244, scale=(1.0, 1.0)),
            trans,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]))

    test_loader = datasets.ImageFolder(
        traindir,  
        transforms.Compose([
            transforms.RandomResizedCrop(244, scale=(1.0, 1.0)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]))

    return train_loader, test_loader

def train(train_loader, model, optimizer, epoch, log):

    # switch to train mode
    model.train()
    train_loss = 0

    #end = time.time()
    count = 0
    for input, target in train_loader:
        count += len(input)

        #input_var = torch.autograd.Variable(input)
        input_var = input.cuda()

        # compute output
        recon_batch, mu, logvar = model(input_var)
        loss = model.module.loss_unsupervised(recon_batch, input_var, mu, logvar)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss/count


def test(test_loader, model, epoch, log):

    # switch to evaluate mode
    model.eval()
    test_loss = 0

    count = 0
    for input, target in test_loader:
        count += len(input)

        #input_var = torch.autograd.Variable(input, volatile=True)
        input_var = input.cuda()

        # compute output
        recon_batch, mu, logvar = model(input_var)
        loss = model.module.loss_unsupervised(recon_batch, input_var, mu, logvar)
        test_loss += loss.item()

    return test_loss/count


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def knn_evaluate(model, train_loader, test_loader):

    train_data, train_labels = get_representation(model, train_loader)
    test_data, test_labels = get_representation(model, test_loader)

    nn = KNeighborsClassifier(n_neighbors=3)
    nn.fit(train_data[:2000], train_labels[:2000])
    print(100 - 100 * nn.score(test_data[:5000], test_labels[:5000]))

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