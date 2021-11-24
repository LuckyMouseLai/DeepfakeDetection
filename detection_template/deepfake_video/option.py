import argparse
from random import choices
import torch
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
"""  
    -result
        -exp_name(date + model_name + train_dataset)
            -log
                -log_name
            -model
                -epoch_
            -tensorboard
                -train
                -val
                -test
"""
parser = argparse.ArgumentParser() 

parser.add_argument('--device',type=str, default=torch.device('cuda:2' if (torch.cuda.is_available()) else "cpu"),
                    help='use cpu or gpu to run')
parser.add_argument('--seed',type=int, default=None,
                    help='seed, if None, denotes no seed')
parser.add_argument('--data_root', type=str, default='/home/TrainingData/laizhenqiang/train_test_val/FF++_all',
                    help='dataset save root dir')
parser.add_argument('--model_name', type=str, default='3dresnet50', choices=['xception', 'CGT', 'TS', '3dresnet50'],
                    help='whitch model used')
parser.add_argument('--dataset', type=str, default='Deepfakes', choices=['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'HQ', 'LQ', 'RAW'],
                    help='which dataset to train and test')
parser.add_argument('--compression', type=str, default='c23', choices=['c0', 'c23', 'c40'],
                    help='FF++ compression')
parser.add_argument('--nums', type=int, default=10,
                    help='random selection of image numbers in a video')

### train setting
parser.add_argument('--image_size', type=int, default=224,
                    help='the height of training images')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size of training')
parser.add_argument('--num_epochs', type=int, default=2,
                    help='the number of training epochs')
parser.add_argument('--lr_rate', type=float, default=5e-3,
                    help='learning rate')
parser.add_argument('--exp_name', type=str, default=None,
                    help='name of experiment', required=False)
                  
### device settings
parser.add_argument('--num_gpu', type=int, default=1,
                    help='number of gpu used in training')
                    
### log settings
parser.add_argument('--save_dir', type=str, default='/home/Users/laizhenqiang/AAAAA/video',
                    help='dir to save log, models, and figs')

### dataloader settings
parser.add_argument('--num_workers', type=int, default=4,
                    help='the number of workers to loading data')

### validation settings
parser.add_argument('--val_every', type=int, default=1,
                    help='validation period')

### optimizer settings
parser.add_argument('--beta1', type=float, default=0.9,
                    help='the beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='the beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='the eps in Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=0,  # 0.05
                    help='learning rate decay type')

parser.add_argument('--gamma', type=float, default=0.9,
                    help='learning rate decay dactor for step decay')
parser.add_argument('--step_size', type=float, default=5,
                    help='adjust lr each 20 epochs')


args = parser.parse_args()