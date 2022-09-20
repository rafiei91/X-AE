import argparse
import os, sys
import time
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import torchvision.transforms as transforms

import torchvision.models

#import models
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from models.modules import VAE_Xray
import utils.vae_utils as vu

'''model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))'''

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
'''parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')'''
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--k', default=5, type=int, metavar='N', help='Number of folds for cross validation')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--opt', default='A', type=str, metavar='N', help='A for Adam and S for SGD')
parser.add_argument('--pretrain', default='True', type=str, metavar='N', help='If use pretrained model')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

writer = SummaryWriter(args.save_dir)

def main():
    best_loss = 999999
    seed = 42

    if not os.path.isdir(args.save_dir):
      os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '.log'), 'w')

    # version information
    vu.print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    vu.print_log("PyTorch version : {}".format(torch.__version__), log)
    vu.print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    vu.print_log("Vision  version : {}".format(torchvision.__version__), log)
    

    # Data loading code
    train_set, test_set = vu.get_dataset(seed=seed, traindir=args.data)

    '''if args.evaluate:
        validate(val_loader, model, criterion)
        return'''

    filename = os.path.join(args.save_dir, 'checkpoint.pth.tar')
    bestname = os.path.join(args.save_dir, 'best.pth.tar')

    torch.manual_seed(seed)
    m=len(train_set)

    k = args.k

    splits=KFold(n_splits=k, shuffle=False, random_state=None)
    foldperf={}

    #torch.save(model, 'model_plane.pt')

    for fold, (train_idx,test_idx) in enumerate(splits.split(np.arange(m))):

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler)

        model = VAE_Xray(dist_weight=1)

        model = torch.nn.DataParallel(model).cuda()

        if args.opt=='A':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

        cudnn.benchmark = True

        history = {'train_loss': [], 'test_loss': []}

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_loss = vu.train(train_loader, model, optimizer, epoch, log)
             
            # evaluate on validation set
            test_loss = vu.test(test_loader, model, epoch, log)

            writer.add_scalars('train/test {} loss'.format(fold+1), {
                'train loss': train_loss,
                'test loss': test_loss,
            }, epoch)

            # remember best prec@1 and save checkpoint
            is_best = test_loss > best_loss
            best_loss = min(test_loss, best_loss)
            vu.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename, bestname)

            if is_best:
                best_model = model
            
            writer.flush()
            

            print("Fold:{} Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(fold + 1,
                                                                                                epoch + 1,
                                                                                                args.epochs,
                                                                                                train_loss,
                                                                                                test_loss))

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)

            writer.close()
            
        foldperf['fold{}'.format(fold+1)] = history 
        #torch.save(model,'k_cross_CNN{}.pt'.format(fold+1))
        torch.save(best_model,'./saved_models/k{}.pt'.format(args.opt, args.pretrain, fold+1)) 
     

    vall_f,tl_f=[],[]
    for f in range(1,k+1):

        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        vall_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

    vu.print_log('Performance of {} fold cross validation'.format(k),log)
    vu.print_log("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} ".format(np.mean(tl_f),np.mean(vall_f)),log) 
    log.close()    



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
