import cv2
import numpy as np
from .heatmap import create_isotropic_image, find_heatmap_peak


def _draw_circle(image, point, color, radius=1):
    x, y = point
    if x >= 0 and y >= 0:
        cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return image


def _draw_line(image, point1, point2, color, thickness=1):
    x1, y1 = point1
    x2, y2 = point2
    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image


def draw_keypoints(image, keypoints, radius=1, alpha=1.0):
    overlay = image.copy()
    for kp in keypoints:
        for x, y in kp.reshape(-1, 2):
            overlay = _draw_circle(overlay, (x, y), (0, 255, 0), radius)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_connections(image, keypoints, thickness=1, alpha=1.0):
    overlay = image.copy()
    conns_h = [(0, 14), (0, 15), (14, 16), (15, 17)]
    conns_b = [(0, 1), (1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]
    conns_l = [(5, 6), (6, 7), (11, 12), (12, 13)]
    conns_r = [(2, 3), (3, 4), (8, 9), (9, 10)]
    for kp in keypoints:
        kp = kp.reshape(-1, 2)
        for i, j in conns_h:
            overlay = _draw_line(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in conns_b:
            overlay = _draw_line(overlay, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in conns_l:
            overlay = _draw_line(overlay, kp[i], kp[j], (255, 255, 0), thickness)
        for i, j in conns_r:
            overlay = _draw_line(overlay, kp[i], kp[j], (255, 0, 255), thickness)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def translate_heatmap(heatmap, target_size):
    h1, w1 = heatmap.shape[:2]
    h2, w2 = target_size[1], target_size[0]
    scales = np.float32([w2/w1, h2/h1])
    kp_old = []
    for i in range(heatmap.shape[2]):
        kp_old.append(find_heatmap_peak(heatmap[:, :, i], 0.0)[1])
    kp_old = np.int32(kp_old)
    kp_new = np.int32(np.where(kp_old >= 0, scales * kp_old, 1.0 * kp_old))
    shifts = kp_new - kp_old
    heatmap_new = np.zeros((h2, w2, heatmap.shape[2]), dtype=heatmap.dtype)
    for i in range(heatmap.shape[2]):
        ys, xs = np.where(heatmap[:, :, i] > 0)
        coords_old = np.int32(np.concatenate((xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=1))
        coords_new = coords_old + shifts[i]
        for (x1, y1), (x2, y2) in zip(coords_old, coords_new):
            if x2 in range(w2) and y2 in range(h2):
                heatmap_new[y2, x2, i] = heatmap[y1, x1, i]
    return heatmap_new


def visualize_heatmap(heatmap, distribution=True, keypoints=True, connections=True, confidence_cutoff=0.5,
                      keypoint_radius=1, connection_thickness=1, alpha=1.0):
    image = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    if distribution:
        proba = np.sum(heatmap, axis=2)
        proba /= np.max(proba)
        image = create_isotropic_image(proba)[0]
    if keypoints or connections:
        kp = []
        for i in range(heatmap.shape[2]):
            kp.append(find_heatmap_peak(heatmap[:, :, i], confidence_cutoff)[1])
        kp = np.int32([kp])
    if connections:
        image = draw_connections(image, kp, connection_thickness, alpha)
    if keypoints:
        image = draw_keypoints(image, kp, keypoint_radius, alpha)
    return image
