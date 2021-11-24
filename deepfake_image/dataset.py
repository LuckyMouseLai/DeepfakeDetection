import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
# from albumentations import CenterCrop, Compose, Resize, RandomCrop
labels = ['Original', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']

class FFPP_Dataset(Dataset):
    """  
        csv_file: [img_path, label], header = None
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]
        
    def __getitem__(self, index: int):
        
        image_path = self.data.loc[index][0]
        label = self.data.loc[index][1]
        if label == 'Original':
            label = 0
        elif label in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            label = 1
        else:
            raise 'label is mismatch to [fake] and [real]'
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

class FFPP_videos_Dataset(Dataset):
    """  
        csv_file: [img_path, label], header = None
    """
    def __init__(self, csv_file, temporal, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.temporal = temporal

    def __len__(self) -> int:
        return self.data.shape[0]
        
    def __getitem__(self, index: int):
        
        clip = torch.tensor([])  # 创建一个空的tensor

        paths = self.data.loc[index][0][1:-1]  # 获得一个clip的 paths
        frame_paths = paths.split(', ')  ### 按照 {逗号和空格}来分，一定要空格, [1:-1] 去除头尾的 （'）
        label = self.data.loc[index][1]
        if label == 'Original':
            label = 0
        elif label in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            label = 1
        else:
            raise 'label is mismatch to [fake] and [real]'
        for image_path in frame_paths:
            image = Image.open(image_path[1:-1])
            if self.transform is not None:
                image = self.transform(image)
                image = torch.unsqueeze(image, 1)
                clip = torch.cat((clip, image), 1)
        sample = {'clip': clip, 'label': label}
        return sample

class HQ_LQ_Dataset(Dataset):
    """  
        csv_file: [img_path, label], header = None
    """
    def __init__(self, csv_file, transform=None, phase = 'train'):
        self.data = pd.read_csv(csv_file, header=None)
        self.phase = phase
        self.transform = transform
        self.aug = A.Compose([
            A.ImageCompression(30, 50, always_apply=True)
        ])

    def __len__(self) -> int:
        return self.data.shape[0]
        
    def __getitem__(self, index: int):
        
        image_path = self.data.loc[index][0]
        label = self.data.loc[index][1]
        if label == 'Original':
            label = 0
        elif label in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']: 
            label = 1
        else:
            raise 'label is mismatch to [fake] and [real]'
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.phase == 'train':
            aug_image = self.aug(image=image)['image']
            # cv2.imwrite('./te.png', aug_image)
            # cv2.imwrite('./tee.png', image)
        if self.phase == 'train' and self.transform is not None:
            image = self.transform(image=image)
            aug_image = self.transform(image=aug_image)
        elif self.phase != 'train' and self. transform is not None:
            image = self.transform(image=image)
            aug_image = image
        sample = {'HQ': image, 'LQ': aug_image, 'label': label}
        return sample

# aug = A.Compose([
#     A.ImageCompression(40, 50, always_apply=True)
# ])
# trans = A.Compose([
#     A.Resize(299, 299),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2(),
# ])
# a = FFPP_Dataset(csv_file='/home/Users/laizhenqiang/test.csv')
# for i in a:
#     print(i)