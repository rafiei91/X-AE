import argparse
import os, sys
import time
import copy
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.models

import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from models.modules import VAE_Xray
import utils.vae_utils as vu
import utils.evaluate as ev

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
parser.add_argument('--latent_size', default=50, type=int, metavar='N', help='Size of latent features')
parser.add_argument('--use_mse', dest='use_mse', action='store_true', help='mse(true) or cross entropy (false)')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

shutil.rmtree(args.save_dir)

writer = SummaryWriter(args.save_dir)

def main():
    total_best_loss = 999999
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


    filename = os.path.join(args.save_dir, 'checkpoint.pth.tar')
    bestname = os.path.join(args.save_dir, 'best.pth.tar')

    torch.manual_seed(seed)
    m=len(train_set)

    k = args.k

    splits=KFold(n_splits=k, shuffle=False, random_state=None)
    foldperf={}
    knn_eval=np.array([])

    #torch.save(model, 'model_plane.pt')

    for fold, (train_idx,test_idx) in enumerate(splits.split(np.arange(m))):

        best_loss = 999999
        bestt_loss = 999999
        check_epoch = args.epochs

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler)

        model = VAE_Xray(latent_size=args.latent_size, dist_weight=1, use_mse=args.use_mse)

        model = torch.nn.DataParallel(model).cuda()

        if args.opt=='A':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

        cudnn.benchmark = True

        history = {'train_loss': [], 'test_loss': [], 'eval_knn': []}

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_loss = vu.train_vae(train_loader, model, optimizer, epoch, log)
             
            # evaluate on validation set
            test_loss = vu.test_vae(test_loader, model, epoch, log)

            writer.add_scalars('train/test {} loss'.format(fold+1), {
                'train loss': train_loss,
                'test loss': test_loss,
            }, epoch)

            if test_loss < best_loss:
                best_loss = test_loss
                bestt_loss = train_loss
                check_epoch = epoch
                if best_loss < total_best_loss:
                    total_best_loss = best_loss
                    best_model = model
                    torch.save(best_model,'/home/mehdi/X-AE-best/saved_models/best-vae.pt')

            if epoch-check_epoch > 20:
                break
                
            writer.flush()
            
            print("Fold:{} Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f}".format(fold + 1,
                                                                                                epoch + 1,
                                                                                                args.epochs,
                                                                                                train_loss,
                                                                                                test_loss))

            writer.close()
            
        knn_eval_out = ev.evaluate(best_model, train_loader, test_loader)
        history['train_loss'].append(bestt_loss)
        history['test_loss'].append(best_loss)
        history['eval_knn'].append(knn_eval_out)
        knn_eval = np.append(knn_eval, knn_eval_out)
        print(knn_eval)

        foldperf['fold{}'.format(fold+1)] = history 


        #vu.knn_evaluate(best_model, train_loader, test_loader) 
        #torch.save(model,'k_cross_CNN{}.pt'.format(fold+1))
        #torch.save(best_model,'./saved_models/k{}.pt'.format(fold+1)) 
     

    test_f,train_f,eval_knn = [],[],[]
    for f in range(1,k+1):

        train_f.append(foldperf['fold{}'.format(f)]['train_loss'])
        test_f.append(foldperf['fold{}'.format(f)]['test_loss'])
        eval_knn.append(foldperf['fold{}'.format(f)]['eval_knn'])

    vu.print_log('Performance of {} fold cross validation'.format(k),log)
    vu.print_log("Average Training Loss: {:.1f} \t Average Test Loss: {:.1f} \t Average knn validation: {:.1f} ".format(np.mean(train_f),np.mean(test_f),np.mean(eval_knn)),log) 
    log.close()  

    #best_model = torch.load('./saved_models/best.pt')

     



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
