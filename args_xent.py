import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='cub')
    parser.add_argument('--load', default=True)

    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help="weight decay (default: 5e-04)")

    parser.add_argument('--max-epoch', default=60, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--schedule', type=int, nargs='+', default=[15, 30, 45, 60], 
                        help='Decrease learning rate at these number of tasks.')

    parser.add_argument('--train-batch', default=5, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=5, type=int,
                        help="test batch size")

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='./result/')
    parser.add_argument('--model_name', type=str, default='map')
    parser.add_argument('--backbone', type=str, default='conv4')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--gpu-devices', default='0', type=str)

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')

    parser.add_argument('--train_nTestNovel', type=int, default=5 * 15,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=1000,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=5 * 15,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=600,
                        help='number of batches per epoch')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='')

    parser.add_argument('--alpha', type=float, default=0.2,
                        help='')
    parser.add_argument('--miu', type=float, default=1,
                        help='')
    parser.add_argument('--phase', default='val', type=str,
                        help='use test or val dataset to early stop')

    return parser

