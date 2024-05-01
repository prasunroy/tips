# imports
import numpy as np
import os
import pandas as pd
import torch
from refinenet import RefineNet
from PIL import Image
from visualization import visualize_skeletons


# configurations
# -----------------------------------------------------------------------------
run_id = 'xxxx-xx-xx-xx-xx-xx' # from ../output/refinenet/xxxx-xx-xx-xx-xx-xx

data_root = '../datasets/DF-PASS'
test_images = f'{data_root}/test_img_list.csv'
keypoints_data_test = f'{data_root}/test_img_keypoints.csv'
model_state_dict = f'../output/refinenet/{run_id}/refinenet_best.pth'
output_dir = f'../output/refinenet/{run_id}/test'
noise_range = (-5, 5)
use_gpu = True
# -----------------------------------------------------------------------------


# get file id of an image
def get_file_id(fp):
    return os.path.splitext(os.path.normpath(fp))[0].replace('/', '').replace('\\', '')


# create model
model = RefineNet(10, 10, bias=True)
if use_gpu and torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(model_state_dict))
model.eval()


# load data
images = pd.read_csv(test_images)
keypoints_data = pd.read_csv(keypoints_data_test, index_col='file_id')


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


skipped = 0
success = 0

for i in range(len(images)):
    fp = images.iloc[i].img
    file_id = get_file_id(fp)
    kp = keypoints_data.loc[file_id].values[2:38].reshape(-1, 2)[[0, 14, 15, 16, 17], :].astype(np.int32)
    if np.allclose(kp[0], [-1, -1]):
        skipped += 1
        continue
    z = np.random.randint(noise_range[0], noise_range[1], kp.shape)
    z = np.where(kp == [-1, -1], 0, z)
    z[0, :] = 0
    noisy_kp = kp + z
    x = torch.tensor(np.where(noisy_kp == [-1, -1], 0, noisy_kp-noisy_kp[0]).reshape(1, 1, 1, -1).astype(np.float32)) / 50
    if use_gpu and torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        p = model(x)
    p = (p.detach().cpu().squeeze().numpy() * 50).astype(np.int32).reshape(-1, 2)
    p = np.where(kp == [-1, -1], -1, p+kp[0])
    kp_18x = keypoints_data.loc[file_id].values[2:38].reshape(-1, 2).astype(np.int32)
    kp_18x[[0, 14, 15, 16, 17], :] = noisy_kp
    kp_18p = keypoints_data.loc[file_id].values[2:38].reshape(-1, 2).astype(np.int32)
    kp_18p[[0, 14, 15, 16, 17], :] = p
    kp_18y = keypoints_data.loc[file_id].values[2:38].reshape(-1, 2).astype(np.int32)
    kp_18y[[0, 14, 15, 16, 17], :] = kp
    img_x = Image.fromarray(visualize_skeletons([kp_18x], 3, 1, head_color=(255, 100, 100))).crop((40, 0, 216, 256))
    img_p = Image.fromarray(visualize_skeletons([kp_18p], 3, 1, head_color=(100, 100, 255))).crop((40, 0, 216, 256))
    img_y = Image.fromarray(visualize_skeletons([kp_18y], 3, 1, head_color=(100, 255, 100))).crop((40, 0, 216, 256))
    grid = Image.new('RGB', (528, 256))
    grid.paste(img_x, (0, 0))
    grid.paste(img_p, (176, 0))
    grid.paste(img_y, (352, 0))
    grid.save(f'{output_dir}/{file_id}.png')
    success += 1
    print(f'\rTesting RefineNet... {i+1}/{len(images)} [success: {success}] [skipped: {skipped}]', end='')
print('')
