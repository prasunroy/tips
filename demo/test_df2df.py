"""TIPS: Text-Induced Pose Synthesis

Test TIPS inference pipeline
Created on Thu Nov 18 10:00:00 2021
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/tips

"""


import datetime
import numpy as np
import os
import pandas as pd
from PIL import Image
from tips import TIPS
from tips import visualize_skeletons, visualize


# -----------------------------------------------------------------------------
prng = np.random.default_rng(1)

ckpt_text2pose = './checkpoints/text2pose_75000.pth'
ckpt_refinenet = './checkpoints/refinenet_100.pth'
ckpt_pose2pose = './checkpoints/pose2pose_260500.pth'

data_root = './data'
save_root = f'./output/df2df_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

keypoints = pd.read_csv('./data/keypoints.csv', index_col='file_id')
encodings = pd.read_csv('./data/encodings.csv', index_col='file_id')
img_pairs = pd.read_csv('./data/img_pairs_df2df.csv')

font = './data/FreeMono.ttf'
bbox = (40, 0, 216, 256)
# -----------------------------------------------------------------------------


def file_id(path):
    return os.path.splitext(os.path.basename(path))[0]


if not os.path.isdir(save_root):
    os.makedirs(save_root)


tips = TIPS(ckpt_text2pose, ckpt_refinenet, ckpt_pose2pose)


z = prng.normal(size=128).astype(np.float32)


layout = [
    ['iA', 'kA',    'iB', 'kB',    'iB_k0'],
    ['iA', 'kA',    'iB', 'kB_c1', 'iB_c1'],
    ['iA', 'kA',    'iB', 'kB_f1', 'iB_f1'],
    ['iA', 'kA_c2', 'iB', 'kB_c2', 'iB_c2'],
    ['iA', 'kA_f2', 'iB', 'kB_f2', 'iB_f2']
]


for i in range(len(img_pairs)):
    fpA = img_pairs.iloc[i].imgA
    fpB = img_pairs.iloc[i].imgB
    
    source_text_encoding = encodings.loc[file_id(fpA)].values[0:84].astype(np.float32)
    target_text_encoding = encodings.loc[file_id(fpB)].values[0:84].astype(np.float32)
    
    source_keypoints = keypoints.loc[file_id(fpA)].values[2:38].astype(np.int32)
    target_keypoints = keypoints.loc[file_id(fpB)].values[2:38].astype(np.int32)
    
    source_image = Image.open(f'{data_root}/{fpA}')
    target_image = Image.open(f'{data_root}/{fpB}')
    
    iB_k = tips.benchmark(source_image, source_keypoints, target_keypoints)
    out1 = tips.pipeline(source_image, source_keypoints, target_text_encoding, z)
    out2 = tips.pipeline_full(source_image, source_text_encoding, target_text_encoding, z)
    
    images_dict = {
        'iA': source_image.crop(bbox),
        'iB': target_image.crop(bbox),
        'iB_k0': iB_k.crop(bbox),
        'iB_c1': out1['iB_c'].crop(bbox),
        'iB_f1': out1['iB_f'].crop(bbox),
        'iB_c2': out2['iB_c'].crop(bbox),
        'iB_f2': out2['iB_f'].crop(bbox),
        'kA': Image.fromarray(visualize_skeletons([source_keypoints], head_color=(100, 255, 100))).crop(bbox),
        'kB': Image.fromarray(visualize_skeletons([target_keypoints], head_color=(100, 255, 100))).crop(bbox),
        'kA_c2': Image.fromarray(visualize_skeletons([out2['kA_c']], head_color=(255, 100, 100))).crop(bbox),
        'kA_f2': Image.fromarray(visualize_skeletons([out2['kA_f']], head_color=(100, 100, 255))).crop(bbox),
        'kB_c1': Image.fromarray(visualize_skeletons([out1['kB_c']], head_color=(255, 100, 100))).crop(bbox),
        'kB_f1': Image.fromarray(visualize_skeletons([out1['kB_f']], head_color=(100, 100, 255))).crop(bbox),
        'kB_c2': Image.fromarray(visualize_skeletons([out2['kB_c']], head_color=(255, 100, 100))).crop(bbox),
        'kB_f2': Image.fromarray(visualize_skeletons([out2['kB_f']], head_color=(100, 100, 255))).crop(bbox),
    }
    
    grid = visualize(images_dict, layout, True, font)
    grid.save(f'{save_root}/{file_id(fpA)}____{file_id(fpB)}.png')
    print(f'\r[TIPS] Testing inference pipeline... {i+1}/{len(img_pairs)}', end='')

print('')
