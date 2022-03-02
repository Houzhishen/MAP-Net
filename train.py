from __future__ import print_function
from __future__ import division

import os
import sys
import time
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
sys.path.append('./torchFewShot')

from args_xent import argument_parser

from torchFewShot.models.trainer import ModelTrainer
from torchFewShot.models.propagation import Propagation
from torchFewShot.models.resnet12 import resnet12
from torchFewShot.models.conv4 import ConvNet
from torchFewShot.data_manager import DataManager

from torchFewShot.utils.logger import Logger

from torch.utils.tensorboard import SummaryWriter 
# from torchstat import stat
from thop import profile

parser = argument_parser()
args = parser.parse_args()

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.backbone, '%dway_%dshot'%(args.nKnovel, args.nExemplars), args.model_name)

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    args.log_dir = osp.join(args.save_dir, 'log')

    writer = SummaryWriter(log_dir=args.log_dir)

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    data_loader = {'train': trainloader,
                   'val': testloader
                   }

    if args.dataset == 'cub':
        word_dim = 312
    elif args.dataset == 'sun':
        word_dim = 102
    elif args.dataset == 'flower':
        word_dim = 1024
    
    if args.backbone == 'conv4':
        enc_module = ConvNet()
    elif args.backbone == 'resnet12':
        enc_module = resnet12()

    out_channels = enc_module.out_channels
    lp_module = Propagation(args, word_dim, out_channels)

    # create trainer
    trainer = ModelTrainer(args=args,
                           enc_module=enc_module,
                           lp_module=lp_module,
                           data_loader=data_loader,
                           writer=writer)

    if args.resume:
        checkpoint = torch.load(args.save_dir + '/model_best.pth.tar')
        trainer.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
        print("load pre-trained enc_nn done!")
        trainer.lp_module.load_state_dict(checkpoint['lp_module_state_dict'])
        print("load pre-trained lp_nn done!")


        trainer.val_acc = checkpoint['val_acc']
        trainer.global_step = checkpoint['iteration']
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        print(trainer.global_step)

    trainer.train()


if __name__ == '__main__':
    main()
