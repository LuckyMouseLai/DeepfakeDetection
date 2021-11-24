import torch 
import torch.nn as nn
import os
import numpy as np

from utils import evaluate_ffpp, dataset_transforms, HQ_LQ_transforms
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from dataset import FFPP_Dataset, HQ_LQ_Dataset
from network.CT import CGTransformer
from network.xception import xception
from network.Resnet import resnet50

def main():
    """  
        ****注意事项
        使用之前，注意数据的transform是否一致，因为image size通过args传入
        如果评估2种不同输入size的模型，需要注意修改transform
        如果使用的模型发生过改变，image size可能不符合。
        最好评估一次，训练时已经得到的结果作为验证
    
    """
    
    device = 'cuda:0'
    # model = CGTransformer(att_nums=3).to(device)
    # model = xception(num_classes=2).to(device)
    model = resnet50(pretrained=False, num_classes=2).to(device)
    model_path = '/home/Users/laizhenqiang/AAAAA/image/Res50/1120_20-16-05_Res50_RAW_c0_100/model/best_Res50_RAW_epoch_28.pth'  # 0.0
    # csv_file = '/home/Users/laizhenqiang/Deepfake-Detection/deepfake_image/csvfile/images/RAW_c0_30_test.csv'
    csv_file = '/home/Users/laizhenqiang/Deepfake-Detection/deepfake_image/csvfile/images/LQ_c40_30_test.csv'
    # csv_file = '/home/Users/laizhenqiang/Deepfake-Detection/deepfake_image/csvfile/images/LQ_c40_30_test.csv'


    model.load_state_dict(torch.load(model_path))
    test_dataset = HQ_LQ_Dataset(csv_file=csv_file, transform=HQ_LQ_transforms['test'], phase='test')
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False)
    ACC, AUC, r_acc, f_acc = evaluate_ffpp(model=model, dataloader=dataloader, device=device)
    print('model: *{}*, test on: *{}* ———— acc: {:.4f}, auc: {:.4f}, r_acc: {:.4f}, f_acc: {:.4f}'.format(
        model_path.split('/')[-1], csv_file.split('/')[-1], ACC, AUC, r_acc, f_acc
    ))

if __name__ == '__main__':
    main()
