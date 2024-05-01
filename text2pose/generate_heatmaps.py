import numpy as np
import os
import pandas as pd
from utils.heatmap import create_gaussian_heatmap, create_isotropic_image


def generate_heatmaps(out_dir, keypoint_data, scale=1.0, spread=1.0):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    kp_data = pd.read_csv(keypoint_data)
    for i in range(len(kp_data)):
        file_id = kp_data.iloc[i]['file_id']
        grid_size = np.max(kp_data.iloc[i, 1:3].values.astype(np.int32))
        grid_size = np.int32(np.round(scale * grid_size))
        keypoints = kp_data.iloc[i, 3:39].values.astype(np.int32).reshape(-1, 2)
        keypoints = np.int32(np.floor(scale * keypoints))
        n = keypoints.shape[0]
        heatmaps = np.zeros((grid_size, grid_size, n), dtype=np.float32)
        for k in range(n):
            if keypoints[k, 0] == -1 or keypoints[k, 1] == -1:
                continue
            gaussian_heatmap = create_gaussian_heatmap(grid_size, keypoints[k], spread)[0]
            isotropic_heatmap = create_isotropic_image(gaussian_heatmap)[1]
            heatmaps[:, :, k] = isotropic_heatmap.astype(np.float32) / 255.0
        np.savez_compressed(f'{out_dir}/{file_id}.npz', heatmaps)
        print(f'\rGenerating heatmaps... {i+1}/{len(kp_data)} [{(i+1)*100.0/len(kp_data):.0f}%]', end='')
    print('')


if __name__ == '__main__':
    out_dir = '../datasets/DF-PASS/gaussian_heatmaps'
    keypoint_data_train = '../datasets/DF-PASS/train_img_keypoints.csv'
    keypoint_data_test = '../datasets/DF-PASS/test_img_keypoints.csv'
    generate_heatmaps(out_dir, keypoint_data_train, 0.25, 0.1)
    generate_heatmaps(out_dir, keypoint_data_test, 0.25, 0.1)
