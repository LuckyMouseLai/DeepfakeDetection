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
    root: path to dataset
    json_path: path to json
    dataset_name: dataset name

    return data: [img_path, label]
"""
# def get_data(root, json_path, dataset_name, nums):
#     real_ids, fake_ids = get_video_ids(json_path)  # 获得json中的real fake id：即各个阶段的文件夹名字
#     data = []
#     for id in tqdm(real_ids):
#         label = 'real'
#         path = os.path.join(root, 'Original', id)  # path to each dir
#         # nums_file = len(os.listdir(path))  # 单个文件夹下的图片数量
#         index = random.sample(os.listdir(path), nums)  # 随机生成的索引
#         for name in index:
#             # img_name = f'{id}_{i:04d}.png'  # 图片名称
#             img_path = os.path.join(path, name)  # 图片路径
#             data.append([img_path, label])  # 将sample添加到data中

#     for id in tqdm(fake_ids):
#         label = 'fake'
#         path = os.path.join(root, dataset_name, id)
#         nums_file = len(os.listdir(path))  # 单个文件夹下的图片数量
#         if nums_file < nums:
#             print(id, '文件夹下有', nums_file, '张图片')
#             continue
#         index = random.sample(os.listdir(path), nums)  # 随机生成的索引
#         for name in index:
#             # img_name = f'{id}_{i:04d}.png'  # 图片名称
#             img_path = os.path.join(path, name)  # 图片路径
#             data.append([img_path, label])  # 将sample添加到data中
#     return data

def get_full_image_data(root, json_path, nums):
    real_ids, fake_ids = get_video_ids(json_path)  # 获得json中的real fake id：即各个阶段的文件夹名字
    data = []
    for dataset in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Original']:
        if dataset == 'Original':
            ids = real_ids
        else:
            ids = fake_ids

        for id in tqdm(ids):
            label = dataset
            path = os.path.join(root, dataset, id)  # path to each dir
            # nums_file = len(os.listdir(path))  # 单个文件夹下的图片数量
            index = random.sample(os.listdir(path), nums)  # 随机生成的索引
            for name in index:
                # img_name = f'{id}_{i:04d}.png'  # 图片名称
                img_path = os.path.join(path, name)  # 图片路径
                data.append([img_path, label])  # 将sample添加到data中
    return data

def get_single_image_data(root, single_set, json_path, nums):
    real_ids, fake_ids = get_video_ids(json_path)  # 获得json中的real fake id：即各个阶段的文件夹名字
    data = []
    for dataset in ['Original', single_set]:
        if dataset == 'Original':
            ids = real_ids
        else:
            ids = fake_ids

        for id in tqdm(ids):
            label = dataset
            path = os.path.join(root, dataset, id)  # path to each dir
            # nums_file = len(os.listdir(path))  # 单个文件夹下的图片数量
            index = random.sample(os.listdir(path), nums)  # 随机生成的索引
            for name in index:
                # img_name = f'{id}_{i:04d}.png'  # 图片名称
                img_path = os.path.join(path, name)  # 图片路径
                data.append([img_path, label])  # 将sample添加到data中
    return data

def main_image(args):
    for phase in ['train', 'test', 'val']:
        if phase == 'train':
            print('train...')
            args.json_path = './splits/train.json'
            nums = args.nums
        elif phase == 'test':
            print('test...')
            args.json_path = './splits/test.json'
            nums = 30
        elif phase == 'val':
            print('val...')
            nums = 30
            args.json_path = './splits/val.json'
        
        args.save_csv_file_path = os.path.join('../csvfile/images', '{}_{}_{}_{}.csv'.format(args.dataset, args.compression, nums, phase))
        print('-{}-的-{}-阶段csv文件将保存至-{}-下'.format(args.dataset, phase, args.save_csv_file_path))
        ### 判断是否存在
        if os.path.isfile(args.save_csv_file_path):
            raise OSError('csv file exists')
        ### FF++ Single Dataset Image

        if args.dataset in ['LQ', 'HQ', 'RAW']:
            data = get_full_image_data(os.path.join(args.root, args.compression), args.json_path, nums)
        elif args.dataset in ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']:
            data = get_single_image_data(os.path.join(args.root, args.compression), args.dataset, args.json_path, nums)

        # data = get_data(os.path.join(args.root, args.compression), args.json_path, args.dataset, args.nums)
        pd.DataFrame(data, columns=['path', 'label']).to_csv(args.save_csv_file_path, sep=',', index=False, header=None) # 保存为csv
        csv_file = pd.read_csv(args.save_csv_file_path, header=None)
        for i in csv_file[0]:
            if not os.path.exists(i):
                raise OSError('path {} is not exists'.format(i))
        print('数据量', csv_file.shape)


if __name__ == '__main__':
    """  
        使用方法：修改一下四个内容即可
            --dataset：需要提取的数据集名称，root的子文件夹名称
            --compression：数据集的压缩率
            --nums：一个视频中随机提取的帧数

            --save_csv_file_path：生成csv文件的保存路径，格式为 [数据集名称_压缩率_帧数_阶段.csv, 例如：deepfake_c23_100_train.csv]

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='NeuralTextures', choices=['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'HQ', 'RAW', 'LQ'],
                    help='dataset name')
    parser.add_argument('-c', '--compression', type=str, default='c23', choices=['c23', 'c40', 'c0'],
                    help='dataset compression ratio')
    parser.add_argument('--root', type=str, default='/home/TrainingData/laizhenqiang/FF++_all',
                    help='dataset root directory')
    parser.add_argument('--nums', type=int, default=100,
                    help='nums pics in each video')
    parser.add_argument('--json_path', type=str,
                    help='json file path')
    parser.add_argument('--save_csv_file_path', type=str,
                    help='csv file save path')
    args = parser.parse_args()

    if args.dataset == 'HQ' and args.compression != 'c23':
        raise OSError('HQ 与 压缩率需要匹配')
    if args.dataset == 'LQ' and args.compression != 'c40':
        raise OSError('LQ 与 压缩率需要匹配')
    if args.dataset == 'RAW' and args.compression != 'c0':
        raise OSError('RAW 与 压缩率需要匹配')

    # TOS
    print('数据集名称：', args.dataset)
    print('压缩率：', args.compression)
    print('数据集根目录：', args.root)
    print('数量：', args.nums)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    _ = input('')
    main_image(args)



