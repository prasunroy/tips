"""TIPS: Text-Induced Pose Synthesis

Inference pipeline
Created on Thu Nov 18 10:00:00 2021
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/tips

"""


import numpy as np
import torch
import torchvision
from PIL import Image
from .models.text2pose import NetG as Stage1
from .models.refinenet import RefineNet as Stage2
from .models.pose2pose import NetG as Stage3


class TIPS(object):
    
    def __init__(self, ckpt_text2pose, ckpt_refinenet, ckpt_pose2pose):
        self.stage1 = Stage1(128, 84, 18, 32).eval()
        self.stage2 = Stage2(10, 10, True).eval()
        self.stage3 = Stage3(3, 36, 3, 64).eval()
        
        self.stage1.load_state_dict(torch.load(ckpt_text2pose))
        self.stage2.load_state_dict(torch.load(ckpt_refinenet))
        self.stage3.load_state_dict(torch.load(ckpt_pose2pose))
        
        if torch.cuda.is_available():
            self.stage1.cuda()
            self.stage2.cuda()
            self.stage3.cuda()
        
        self.transforms1 = torchvision.transforms.ToTensor()
        self.transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
    
    def heatmaps2keypoints(self, heatmaps, confidence):
        keypoints = []
        for k in range(heatmaps.shape[2]):
            heatmap_k = heatmaps[:, :, k]
            proba_max = np.max(heatmap_k)
            if proba_max > confidence:
                y, x = np.where(heatmap_k == proba_max)
                y, x = y[0], x[0]
            else:
                y, x = -1, -1
            keypoints.append((x, y))
        return np.int32(keypoints).reshape(-1)
    
    def keypoints2heatmaps(self, keypoints, size):
        keypoints = keypoints.reshape(-1, 2).astype(np.int32)
        heatmaps = np.zeros(size + (keypoints.shape[0],), dtype=np.float32)
        for k in range(keypoints.shape[0]):
            x, y = keypoints[k]
            if x < 0 or y < 0:
                continue
            heatmaps[y, x, k] = 1
        return heatmaps
    
    def stage1_inference(self, z, text_encoding):
        if torch.cuda.is_available():
            z = z.cuda()
            text_encoding = text_encoding.cuda()
        with torch.no_grad():
            heatmaps = self.stage1(z, text_encoding)
        return (heatmaps.detach().cpu().squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
    
    def stage2_inference(self, keypoints):
        head_keypoints = keypoints.reshape(-1, 2)[[0, 14, 15, 16, 17], :].astype(np.int32)
        if np.allclose(head_keypoints[0], [-1, -1]):
            return keypoints
        x = np.where(head_keypoints == [-1, -1], 0, head_keypoints - head_keypoints[0])
        x = torch.tensor(x.reshape(1, -1).astype(np.float32)) / 50
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            p = self.stage2(x)
        p = (p.detach().cpu().squeeze().numpy() * 50).astype(np.int32).reshape(-1, 2)
        p = np.where(head_keypoints == [-1, -1], -1, p + head_keypoints[0])
        refined_keypoints = keypoints.reshape(-1, 2).astype(np.int32)
        refined_keypoints[[0, 14, 15, 16, 17], :] = p
        return refined_keypoints.reshape(keypoints.shape)
    
    def stage3_inference(self, source_image, source_heatmaps, target_heatmaps):
        x1 = source_image.unsqueeze(0)
        x2 = torch.cat((source_heatmaps, target_heatmaps), dim=0).unsqueeze(0)
        if torch.cuda.is_available():
            x1 = x1.cuda()
            x2 = x2.cuda()
        with torch.no_grad():
            p = self.stage3(x1, x2)
        p = (p.detach().cpu().squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        return np.clip(p * 255, 0, 255).astype(np.uint8)
    
    def benchmark(self, source_image, source_keypoints, target_keypoints):
        iA = self.transforms2(source_image)
        pA = self.transforms1(self.keypoints2heatmaps(source_keypoints, (256, 256)).astype(np.float32))
        pB = self.transforms1(self.keypoints2heatmaps(target_keypoints, (256, 256)).astype(np.float32))
        iB = self.stage3_inference(iA, pA, pB)
        return Image.fromarray(iB)
    
    def pipeline(self, source_image, source_keypoints, target_text_encoding, z):
        z = torch.tensor(z.reshape(1, -1).astype(np.float32))
        tB = (torch.tensor(target_text_encoding.reshape(1, -1).astype(np.float32)) - 0.5) / 0.5
        hB = self.stage1_inference(z, tB)
        kB = self.heatmaps2keypoints(hB, 0.2)
        kB = np.where(kB < 0, -1, kB * 4)
        pB = self.transforms1(self.keypoints2heatmaps(kB, (256, 256)).astype(np.float32))
        kB_f = self.stage2_inference(kB)
        pB_f = self.transforms1(self.keypoints2heatmaps(kB_f, (256, 256)).astype(np.float32))
        kA = source_keypoints.reshape(-1)
        pA = self.transforms1(self.keypoints2heatmaps(kA, (256, 256)).astype(np.float32))
        iA = self.transforms2(source_image)
        iB = self.stage3_inference(iA, pA, pB)
        iB_f = self.stage3_inference(iA, pA, pB_f)
        return {
            'kB_c': kB,
            'kB_f': kB_f,
            'iB_c': Image.fromarray(iB),
            'iB_f': Image.fromarray(iB_f)
        }
    
    def pipeline_full(self, source_image, source_text_encoding, target_text_encoding, z):
        z = torch.tensor(z.reshape(1, -1).astype(np.float32))
        tB = (torch.tensor(target_text_encoding.reshape(1, -1).astype(np.float32)) - 0.5) / 0.5
        hB = self.stage1_inference(z, tB)
        kB = self.heatmaps2keypoints(hB, 0.2)
        kB = np.where(kB < 0, -1, kB * 4)
        pB = self.transforms1(self.keypoints2heatmaps(kB, (256, 256)).astype(np.float32))
        kB_f = self.stage2_inference(kB)
        pB_f = self.transforms1(self.keypoints2heatmaps(kB_f, (256, 256)).astype(np.float32))
        tA = (torch.tensor(source_text_encoding.reshape(1, -1).astype(np.float32)) - 0.5) / 0.5
        hA = self.stage1_inference(z, tA)
        kA = self.heatmaps2keypoints(hA, 0.2)
        kA = np.where(kA < 0, -1, kA * 4)
        pA = self.transforms1(self.keypoints2heatmaps(kA, (256, 256)).astype(np.float32))
        kA_f = self.stage2_inference(kA)
        pA_f = self.transforms1(self.keypoints2heatmaps(kA_f, (256, 256)).astype(np.float32))
        iA = self.transforms2(source_image)
        iB = self.stage3_inference(iA, pA, pB)
        iB_f = self.stage3_inference(iA, pA_f, pB_f)
        return {
            'kA_c': kA,
            'kA_f': kA_f,
            'kB_c': kB,
            'kB_f': kB_f,
            'iB_c': Image.fromarray(iB),
            'iB_f': Image.fromarray(iB_f)
        }
