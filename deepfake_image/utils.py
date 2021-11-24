import logging
import os
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torchvision import transforms
from option import args
import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset_transforms = {
    'train': transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

HQ_LQ_transforms = {
    'train': A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
    'val': A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
    'test': A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
}

def calc_metrics(y_true, y_pred, y_score):
        """  
            y_true: label
            y_pred: model's prediction classes [0, 1]
            y_score: model's outputs [numbers, 2]
        """
        # auc
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        AUC = roc_auc_score(y_true, y_score[:, 1])  # 1-fake作为positive label
        ACC = accuracy_score(y_true, y_pred, normalize=True)

        idx_real = np.where(y_true==0)[0]  # 获取label==real， id==0的索引
        idx_fake = np.where(y_true==1)[0]
        R_acc = accuracy_score(y_true[idx_real], y_pred[idx_real])  # 计算real label的准确率 TP / (TP + FN)
        F_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake])  # TN / (TN + FP)
        return ACC, AUC, R_acc, F_acc

def evaluate_ffpp(model, dataloader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch_id, sample in enumerate(tqdm(dataloader)):
            image_batch = sample['HQ']['image'].to(device)
            label_batch = sample['label'].to(device)

            _, outputs = model(image_batch)
            _, preds = torch.max(outputs, 1)
            y_score.extend(outputs.sigmoid().tolist())
            y_true.extend(label_batch.tolist())
            y_pred.extend(preds.tolist())

    ACC, AUC, R_acc, F_acc = calc_metrics(y_true, y_pred, y_score)
    return ACC, AUC, R_acc, F_acc

def mkExpDir(args):
    if args.exp_name is None:
        current_time = time.localtime()
        date = time.strftime('%m%d_%H-%M-%S', current_time)
        args.exp_name = '{}_{}_{}_{}_{}'.format(date, args.model_name, args.dataset, args.compression, args.nums)

    if (os.path.exists(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir, args.model_name, args.exp_name, 'model'))
        os.makedirs(os.path.join(args.save_dir, args.model_name, args.exp_name, 'log'))
    else:
        raise Exception('save_dir:{} is not exist'.format(args.save_dir))
    
    args_file = open(os.path.join(args.save_dir, args.model_name, args.exp_name, 'log', 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

def printInfo(logger):
    logger.info('实验结果将放保存至: {}'.format(os.path.join(args.save_dir, args.model_name, args.exp_name)))
    logger.info(' \
        \n-------------------------dataset info---------\n \
        model name: {}\n \
        dataset root: {}\n \
        dataset name: {}\n \
        compression: {}\n \
        nums: {}\n \
    '.format(args.model_name, args.data_root, args.dataset, args.compression, \
            args.nums))

    logger.info(' \
        \n-------------------------settings-------------\n \
        message: {}\n \
        use_sim_loss: {}\n \
        experimen name: {}\n \
        batch_size: {}\n \
        epoch: {}\n \
        lr: {}\n \
        image size: {}\n \
        \n---------------------------------------------- \
    '.format(args.message, args.use_sim, args.exp_name, args.batch_size, args.num_epochs, args.lr_rate, args.image_size))
    print('Press any key to continue, or CTRL-C to exit.')
    _ = input('')

class Logger():
    def __init__(self, log_file_name, log_level=logging.INFO):
        # 创建一个logger
        self.logger = logging.getLogger()
        # 设置log level
        self.logger.setLevel(log_level)
        # 设置一个handler来写日志
        file_handler = logging.FileHandler(log_file_name)
        # 设置一个handler来打印在控制台
        console_handler = logging.StreamHandler()
        # 定义handlers的输出格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 添加handler到logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def get_logger(self):
        return self.logger





    
        