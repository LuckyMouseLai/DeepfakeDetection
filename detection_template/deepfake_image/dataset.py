import os

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