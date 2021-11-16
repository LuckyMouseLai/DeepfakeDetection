import sys
sys.path.append('..')
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import torch
from dataset import FFPP_Dataset
from model import Xception


device = 'cuda:0'  # id
weight_path = '/home/Users/laizhenqiang/AAAAA/xception/0917_11-22-34_xception_face2face_c23_100/model/xception_face2face_epoch_9.pth'  # 模型的权重文件路径
image_size = 299  # 输入模型图片大小
csv_path = '/home/TrainingData/laizhenqiang/train_test_val/FF++_all/c23/face2face_c23_100_val.csv'  # 数据集的csv 路径
name = csv_path.split('/')[-1].split('.')[0]  # tsne保存结果图片名称、无需后缀


### T-SNE
# features: 网络提取的特征向量(?, n) labels: 特征对应的标签(?, 1)
def tsne_2D(features, labels):
    # print('t-SNE fitting...')
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(features)
    data_tsne = np.vstack((Y.T, labels)).T
    # print('t-SNE fitting over.')
    return data_tsne

def visualization_2D_embedding(dataframe, name):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=dataframe, hue='Class', x='Dim1', y='Dim2')
    plt.savefig('./tsne_result/{}.png'.format(name))

def main():
    ### model
    model = Xception().to(device)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    ### dataset
    trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = FFPP_Dataset(csv_file=csv_path, transform=trans)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    data = np.empty((0, 3))  # 定义一个空数组 [dim1, dim2, label]
    with torch.no_grad():
        for batch_id, sample in enumerate(tqdm(data_loader)):

            image_batch = sample['image'].to(device)
            label_batch = sample['label'].to(device)
            features = model(image_batch)
            batch_data = tsne_2D(features.cpu(), label_batch.cpu())
            data = np.vstack((data, batch_data))

    df_tsne = pd.DataFrame(data=data, columns=['Dim1', 'Dim2', 'Class'])
    visualization_2D_embedding(df_tsne, name)

if __name__ == '__main__':
    main()

           
    

