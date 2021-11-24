import argparse
import os
from os.path import join

import cv2
import mmcv
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm


def extract_frames(face_detector, data_path, output_path, file_prefix):
    os.makedirs(output_path, exist_ok=False)
    video = mmcv.VideoReader(data_path)  # 视频读取
    length = video.frame_cnt  # 视频帧数
    for frame_num, frame in enumerate(video):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_file_path = join(output_path, '{}{:04d}.png'.format(file_prefix, frame_num))
        if not os.path.exists(out_file_path):
            face_detector(image, save_path=out_file_path)
    return length


def extract_images(device, videos_path, out_path, num_videos):
    print('extracting video frames from {} to {}'.format(videos_path, out_path))

    video_files = os.listdir(videos_path)
    print('total videos found - {}, extracting from - {}'.format(len(video_files), min(len(video_files), num_videos)))
    video_files = video_files[:num_videos]
    def get_video_input_output_pairs():
        for index, video_file in enumerate(video_files):
            video_file_name = video_file.split('.')[0]  # 获取文件名称
            v_out_path = os.path.join(out_path, video_file_name)  # 图片存放的文件夹 
            v_path = os.path.join(videos_path, video_file)  # 视频的路径
            f_prefix = '{}_'.format(video_file_name)  # 图片名称的前缀
            yield v_path, v_out_path, f_prefix

    face_detector = MTCNN(device=device, margin=16)
    face_detector.eval()

    for data_path, output_path, file_prefix in tqdm(get_video_input_output_pairs(), total=len(video_files)):
        extract_frames(face_detector, data_path, output_path, file_prefix)
        


def parse_args():
    args_parser = argparse.ArgumentParser()
    # /home/TrainingData/laizhenqiang/FF++/manipulated_sequences/Deepfakes/c23/videos  [Deepfakes, face2face, FaceShifter, faceswap, NeuralTextures]
    args_parser.add_argument('--path_to_videos', type=str, help='path for input videos', default='/home/TrainingData/laizhenqiang/FF++/original_sequences/Youtube/c23/videos')
    args_parser.add_argument('--output_path', type=str, help='path to output path', default='/home/TrainingData/laizhenqiang/train_test_val/FF++_all/c23/real/')
    args_parser.add_argument('--num_videos', type=int, default=1000, help='number of videos to extract images from')
    args = args_parser.parse_args()
    return args


def main():
    # device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('Running on device: {}'.format(device))

    args = parse_args()
    videos_path = args.path_to_videos
    out_path = args.output_path
    num_videos = args.num_videos
    extract_images(device, videos_path, out_path, num_videos)


if __name__ == '__main__':
    main()
