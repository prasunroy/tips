import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RefineNetDataset(Dataset):
    
    def __init__(self, keypoints_data, noise_range=(-1, 1), transform=None):
        super(RefineNetDataset, self).__init__()
        self.keypoints = pd.read_csv(keypoints_data).values[:, 3:39].reshape(-1, 18, 2)[:, [0, 14, 15, 16, 17], :].astype(np.float32)
        self.keypoints = np.float32([k for k in self.keypoints if not np.allclose(k[0], np.float32([-1.0, -1.0]))])
        self.noise = np.random.randint(noise_range[0], noise_range[1], self.keypoints.shape).astype(np.float32)
        self.noise = np.where(self.keypoints >= 0, self.noise, 0)
        self.noise[:, 0, :] = 0
        self.noisy_keypoints = self.keypoints + self.noise
        self.translations = np.float32([np.where(k == [-1.0, -1.0], -1.0, k[0]) for k in self.keypoints])
        self.transform = transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, index):
        keypoints = self.keypoints[index].reshape(-1, 2)
        keypoints = np.where(keypoints == [-1.0, -1.0], 0.0, keypoints-keypoints[0])
        noisy_keypoints = self.noisy_keypoints[index].reshape(-1, 2)
        noisy_keypoints = np.where(noisy_keypoints == [-1.0, -1.0], 0.0, noisy_keypoints-noisy_keypoints[0])
        return {
            'x': self.transform(noisy_keypoints.reshape(1, -1)),
            'y': self.transform(keypoints.reshape(1, -1)),
            't': self.translations[index]
        }


def create_dataloader(keypoints_data, noise_range=(-1, 1), transform=None,
                      batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    dataset = RefineNetDataset(keypoints_data, noise_range, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
