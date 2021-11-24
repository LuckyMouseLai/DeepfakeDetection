import os
import time
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms

from utils import Logger, mkExpDir, printInfo, dataset_transforms
from trainer import Trainer
from dataset import FFPP_Dataset
from network.CT import CGTransformer
from network.xception import get_xcep_model
from loss import SparseCenterLoss
from option import args
import random
import torch.backends.cudnn as cudnn



def main():
    if args.seed is not None:
        # random seed
        SEED = args.seed
        # 设置随机种子
        random.seed(SEED)
        torch.manual_seed(SEED)
        cudnn.deterministic = True
        cudnn.benchmark = False
    # create exp dir
    mkExpDir(args)
    # logger
    logger = Logger(log_file_name= os.path.join(args.save_dir, args.model_name, args.exp_name, 'log', 'train_eval.log')).get_logger()
    # Tensorboard
    board_writer = SummaryWriter(logdir=os.path.join(args.save_dir, args.model_name, args.exp_name))
    
    # csv file path of dataset
    csv_file_path = {'train': os.path.join('./csvfile/images', '{}_{}_{}_train.csv'.format(args.dataset, args.compression, args.nums)), 
                    'test': os.path.join('./csvfile/images', '{}_{}_{}_test.csv'.format(args.dataset, args.compression, 30)),
                    'val': os.path.join('./csvfile/images', '{}_{}_{}_val.csv'.format(args.dataset, args.compression, 30)),
                    }
    # csv_file_path = {x: os.path.join('/home/Users/laizhenqiang/deepfake2/csvfile/images', '{}_{}_{}_{}.csv'.format(args.dataset, args.compression, args.nums, x)) for x in ['train', 'test', 'val']}
    for phase in ['train', 'test', 'val']:
        if not os.path.isfile(csv_file_path[phase]):
            raise OSError('请先生成数据集-{}-阶段的csv文件'.format(phase))
    # dataloader
    datasets = {x: FFPP_Dataset(csv_file=csv_file_path[x], transform=dataset_transforms[x])
                for x in ['train', 'test', 'val']}

    dataloader = {x: torch.utils.data.DataLoader(datasets[x],
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
                for x in ['train', 'test', 'val']}
    # model
    if args.model_name == 'CGT':
        model = CGTransformer(att_nums=3).to(args.device)
    elif args.model_name == 'xception':
        model = get_xcep_model().to(args.device)
    else:
        raise OSError('wrong model')
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    criterion = {
        'CE': torch.nn.CrossEntropyLoss().to(args.device),
        'MSE': torch.nn.MSELoss().to(args.device)
    }
    # optimizer--可以把loss看作网络中的一层
    optimizer = {
        'CE': torch.optim.Adam(model.parameters(), lr=args.lr_rate, betas=(args.beta1, args.beta2), eps=args.eps),
        # 'MSE': torch.optim.SGD(criterion['MSE'].parameters(), 0.1)  
    }
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer['CE'], step_size=args.step_size, gamma=args.gamma)
    ### 输出信息
    printInfo(logger=logger)
    ### 训练
    trainer = Trainer(args=args, model=model, optimizer=optimizer, scheduler=scheduler, dataloader=dataloader, criterion=criterion,  logger=logger, board_writer=board_writer)
    trainer.train_ffpp()

if __name__ == '__main__':
    main()


