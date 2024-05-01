"""
References:
  [1] https://github.com/clovaai/CRAFT-pytorch/issues/3#issuecomment-504548693
  [2] https://colab.research.google.com/drive/1TQ1-BTisMYZHIRVVNpVwDFPviXYMhT7A

"""


import cv2
import numpy as np


# probability density function that returns a probability value within the range [0, 1]
# from a Gaussian distribution with mean = 0 and standard deviation = 1 (Normal distribution)
def gaussian(x):
    return np.exp(-(x**2)/2)


# estimate heatmap as the Gaussian probability distribution function of the Euclidean distance
# from a given center on a 2D plane
def create_gaussian_heatmap(grid_size=512, center=(256, 256), spread=1.0):
    k, (x, y), c = grid_size, center, spread
    assert k > 0 and x in range(k) and y in range(k) and c > 0.0 and c <= 1.0
    distmap = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            distmap[i, j] = np.linalg.norm(np.float32([x-j, y-i])) / np.float32(c * k/2)
    heatmap = gaussian(distmap)
    return heatmap, distmap


# create isotropic image representation of a given probability distribution
def create_isotropic_image(distribution):
    isotropic_gray = np.uint8(np.clip(255 * distribution, 0, 255))
    isotropic_cmap = cv2.applyColorMap(isotropic_gray, cv2.COLORMAP_JET)
    return isotropic_cmap, isotropic_gray


# find location of the highest probability in a given heatmap
def find_heatmap_peak(heatmap, confidence_cutoff=0.0):
    probability_max = np.max(heatmap)
    if probability_max > confidence_cutoff:
        y, x = np.where(heatmap == probability_max)
        y, x = y[0], x[0]
    else:
        y, x = -1, -1
    return probability_max, (x, y)
