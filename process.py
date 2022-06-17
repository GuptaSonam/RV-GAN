#      Convert videos to frames for a dataset

import argparse
import os
import cv2
import h5py
import imageio
import numpy
import pandas
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, default='/media/vplab/Sonam_HDD/sonam-arti/Unconditional_Video_genration/g4an-3gpu/demo', help='')
parser.add_argument('--dst_dir', type=str, default='/media/vplab/Sonam_HDD/sonam-arti/Unconditional_Video_genration/g4an-3gpu/demo_frames', help='')
args = parser.parse_args()

path = args.src_dir
files = os.listdir(path)
n_files= len(files)
# For each video extract and save the frames
for i in range(n_files):
    fname = files[i].split('.')
    video_path_src = os.path.join(path, files[i])
    video_path_dst = os.path.join(args.dst_dir, fname[0])
    if not os.path.exists(video_path_dst):
        os.mkdir(video_path_dst)
    try:
        video_reader = imageio.get_reader(video_path_src)
    except Exception as e:
        print(e)
        print(video_path_src)
        continue

    #video = []
    n_frame = 1
    while True:
        try:
            img = video_reader.get_next_data()
            imageio.imwrite(video_path_dst + '/frame' +"{:04d}".format(n_frame)+'.jpg', img)
            n_frame +=1
        except IndexError:
            break
        
    video_reader.close()