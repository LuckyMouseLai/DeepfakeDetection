import json
import os
from unicodedata import name
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import argparse

"""  
    调用get_video_ids()，返回数据集中所有id，及其组合id，包含了数据集中所有的视频id
    return : real_ids, fake_ids
    args： json_path: json文件的路径
"""
def get_video_ids(json_path):
    with open(json_path) as inp:
        data = json.load(inp)
        real_ids = {x[0] for x in data} | {x[1] for x in data}  
        fake_ids = {'_'.join(x) for x in data} | {'_'.join(x[::-1]) for x in data}
    return real_ids, fake_ids


"""  
    phase = ['train', 'val', 'test'] train:前270帧，  val/test：前110帧
    root:
    json_path:

    return data [img_path, label]
"""
def get_full_video_data(root, json_path, phase, temporal):
    real_ids, fake_ids = get_video_ids(json_path)  # 获得json中的real fake id：即各个阶段的文件夹名字
    if phase == 'train':
        nums_frame = 270
    elif phase =='test' or phase =='val':
        nums_frame = 110
    data = []
    for dataset in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Original']:
        if dataset == 'Original':
            ids = real_ids
        else:
            ids = fake_ids
        
        for id in tqdm(ids):
            label = dataset
            dir_path = os.path.join(root, dataset, id)  # path to each dir in 1000 dirs
            nums_file = len(os.listdir(dir_path))
            assert nums_file >= nums_frame, f'the frames of video id: ({id}——{nums_file}) is less than 270'
            point = 0
            while point < nums_frame:
                clip = []
                for j in range(temporal):
                    img_path = os.path.join(dir_path, '{}_{:04d}.png'.format(id, point+j))
                    if not os.path.exists(img_path):
                        # print(img_path)
                        break
                    # clip.append('{}_{:04d}.png'.format(id, point+j))
                    clip.append(img_path)
                point = point + j + 1
                if len(clip)  == temporal:
                    data.append([clip, label])
    return data

def get_single_video_data(root, single_set, json_path, phase, temporal):
    real_ids, fake_ids = get_video_ids(json_path)  # 获得json中的real fake id：即各个阶段的文件夹名字
    if phase == 'train':
        nums_frame = 270
    elif phase =='test' or phase =='val':
        nums_frame = 110
    data = []
    for dataset in ['Original', single_set]:
        if dataset == 'Original':
            ids = real_ids
        else:
            ids = fake_ids
        
        for id in tqdm(ids):
            label = dataset
            dir_path = os.path.join(root, dataset, id)  # path to each dir in 1000 dirs
            nums_file = len(os.listdir(dir_path))
            assert nums_file >= nums_frame, f'the frames of video id: ({id}——{nums_file}) is less than 270'
            point = 0
            while point < nums_frame:
                clip = []
                for j in range(temporal):
                    img_path = os.path.join(dir_path, '{}_{:04d}.png'.format(id, point+j))
                    if not os.path.exists(img_path):
                        # print(img_path)
                        break
                    # clip.append('{}_{:04d}.png'.format(id, point+j))
                    clip.append(img_path)
                point = point + j + 1
                if len(clip)  == temporal:
                    data.append([clip, label])
    return data

def main_video(args):
    for phase in ['train', 'test', 'val']:
        if phase == 'train':
            print('train...')
            args.json_path = './splits/train.json'
        elif phase == 'test':
            print('test...')
            args.json_path = './splits/test.json'
        elif phase == 'val':
            print('val...')
            args.json_path = './splits/val.json'

        args.save_csv_file_path = os.path.join('../csvfile/videos', '{}_{}_{}_{}.csv'.format(args.dataset, args.compression, args.temporal, phase))
        print('-{}-的-{}-阶段csv文件将保存至-{}-下'.format(args.dataset, phase, args.save_csv_file_path))
    #     ### 判断是否存在
        if os.path.isfile(args.save_csv_file_path):
            raise OSError('csv file exists')
        if args.dataset in ['LQ', 'HQ', 'RAW']:
            data = get_full_video_data(os.path.join(args.root, args.compression), args.json_path, phase, args.temporal)
        elif args.dataset in ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']:
            data = get_single_video_data(os.path.join(args.root, args.compression), args.dataset, args.json_path, phase, args.temporal)

        pd.DataFrame(data, columns=['paths', 'label']).to_csv(args.save_csv_file_path, sep=',', index=False, header=None) # 保存为csv
        csv_file = pd.read_csv(args.save_csv_file_path, header=None)
        print('数据量', csv_file.shape)


if __name__ == '__main__':
    """  
        使用方法：修改一下四个内容即可

            --dataset：需要提取的数据集名称，root的子文件夹名称
            --compression：数据集的压缩率
            --temporal：一个clip中的帧数

            --save_csv_file_path：生成csv文件的保存路径，格式为 [数据集名称_压缩率_时序_阶段.csv, 例如：deepfake_c23_10_train.csv]

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='Deepfakes', choices=['LQ', 'RAW', 'HQ', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'Original'],
                    help='dataset name')
    parser.add_argument('-c', '--compression', type=str, default='c23', choices=['c23', 'c40', 'c0'],
                    help='dataset compression ratio')
    parser.add_argument('--root', type=str, default='/home/TrainingData/laizhenqiang/FF++_all',
                    help='dataset root directory')
    parser.add_argument('--temporal', type=int, default=10,
                    help='nums pics in each video')
    parser.add_argument('--json_path', type=str,
                    help='json file path')
    parser.add_argument('--save_csv_file_path', type=str,
                    help='csv file save path')
    args = parser.parse_args()
    if args.dataset == 'HQ' and args.compression != 'c23':
        raise OSError('HQ 与 压缩率需要匹配')
    if args.dataset == 'LQ' and args.compression != '40':
        raise OSError('LQ 与 压缩率需要匹配')
    if args.dataset == 'RAW' and args.compression != 'c0':
        raise OSError('RAW 与 压缩率需要匹配')

    # TOS
    print('数据集名称：', args.dataset)
    print('压缩率：', args.compression)
    print('数据集根目录：', args.root)
    print('时序：', args.temporal)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    _ = input('')
    main_video(args)


