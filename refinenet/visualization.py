import cv2
import numpy as np


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


def draw_keypoints(image, keypoints, radius=1, head_color=(128, 128, 128), alpha=1.0):
    overlay = image.copy()
    for kp in keypoints:
        for i, (x, y) in enumerate(kp.reshape(-1, 2)):
            if i in [0, 14, 15, 16, 17]:
                overlay = _draw_circle(overlay, (x, y), head_color, radius)
            else:
                overlay = _draw_circle(overlay, (x, y), (128, 128, 128), radius)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_connections(image, keypoints, thickness=1, head_color=(128, 128, 128), alpha=1.0):
    overlay = image.copy()
    conns_h = [(0, 14), (0, 15), (14, 16), (15, 17)]
    conns_b = [(0, 1), (1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]
    conns_l = [(5, 6), (6, 7), (11, 12), (12, 13)]
    conns_r = [(2, 3), (3, 4), (8, 9), (9, 10)]
    for kp in keypoints:
        kp = kp.reshape(-1, 2)
        for i, j in conns_h:
            overlay = _draw_line(overlay, kp[i], kp[j], head_color, thickness)
        for i, j in conns_b:
            overlay = _draw_line(overlay, kp[i], kp[j], (128, 128, 128), thickness)
        for i, j in conns_l:
            overlay = _draw_line(overlay, kp[i], kp[j], (128, 128, 128), thickness)
        for i, j in conns_r:
            overlay = _draw_line(overlay, kp[i], kp[j], (128, 128, 128), thickness)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def visualize_skeletons(keypoints, keypoint_radius=1, connection_thickness=1,
                        head_color=(128, 128, 128), grid_size=(256, 256), alpha=1.0):
    image = np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8) + 255
    image = draw_connections(image, keypoints, connection_thickness, head_color, alpha)
    image = draw_keypoints(image, keypoints, keypoint_radius, head_color, alpha)
    return image
