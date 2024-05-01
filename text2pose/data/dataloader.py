import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Text2PoseDataset(Dataset):
    
    def __init__(self, img_list, text_encoding_data, pose_heatmaps_dir,
                 text_transform=None, pose_transform=None):
        super(Text2PoseDataset, self).__init__()
        self._img_list = pd.read_csv(img_list)
        self._text_encoding_data = pd.read_csv(text_encoding_data, index_col='file_id')
        self._pose_heatmaps_dir = pose_heatmaps_dir
        self._text_transform = text_transform or transforms.ToTensor()
        self._pose_transform = pose_transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self._img_list)
    
    def __getitem__(self, index):
        imgA = self._img_list.iloc[index].img
        fidA = os.path.splitext(imgA)[0].replace('/', '').replace('\\', '')
        textA = self._text_encoding_data.loc[fidA].values[:84].astype(np.float32).reshape(1, -1)
        poseA = np.load(f'{self._pose_heatmaps_dir}/{fidA}.npz')['arr_0']
        while True:
            imgB = self._img_list.iloc[np.random.randint(0, self.__len__())].img
            fidB = os.path.splitext(imgB)[0].replace('/', '').replace('\\', '')
            textB = self._text_encoding_data.loc[fidB].values[:84].astype(np.float32).reshape(1, -1)
            if (textB == textA).all():
                continue
            break
        poseA = self._pose_transform(poseA)
        textA = self._text_transform(textA)
        textB = self._text_transform(textB)
        return {'fidA': fidA, 'poseA': poseA, 'textA': textA, 'textB': textB}


def create_dataloader(img_list, text_encoding_data, pose_heatmaps_dir,
                      text_transform=None, pose_transform=None,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    dataset = Text2PoseDataset(img_list, text_encoding_data, pose_heatmaps_dir, text_transform, pose_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
