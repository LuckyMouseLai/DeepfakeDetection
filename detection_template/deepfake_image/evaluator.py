import torch 
import torch.nn as nn
import os
import numpy as np

from utils import evaluate_ffpp, dataset_transforms
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from dataset import FFPP_Dataset
from network.CT import CGTransformer


def main():
    device = 'cuda:0'
    model = CGTransformer(att_nums=3).to(device)
    model_path = '/home/Users/laizhenqiang/AAAAA/xception/0917_11-22-34_xception_face2face_c23_100/model/best_model_xception_face2face_epoch_0.pth'  # 0.0
    csv_file = '/home/TrainingData/laizhenqiang/train_test_val/FF++_all/c23/face2face_c23_100_test.csv'


    model.load_state_dict(torch.load(model_path))
    test_dataset = FFPP_Dataset(csv_file=csv_file, transform=dataset_transforms['test'])
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=False)
    ACC, AUC, r_acc, f_acc = evaluate_ffpp(model=model, dataloader=dataloader, device=device)
    print('model: *{}*, test on: *{}* ———— acc: {:.4f}, auc: {:.4f}, r_acc: {:.4f}, f_acc: {:.4f}'.format(
        model_path.split('/')[-1], csv_file.split('/')[-1], ACC, AUC, r_acc, f_acc
    ))

if __name__ == '__main__':
    main()
