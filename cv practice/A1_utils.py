import os
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'A1_Images')
RESULT_DIR = os.path.join(BASE_DIR, 'result')

IMAGE_FILES = ['lenna.png', 'shapes.png']

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[ 1,  2,  1],
                    [ 0,  0,  0],
                    [-1, -2, -1]], dtype=np.float64)


# read an image as grayscale (float)
def load_gray(file_name):
    path = os.path.join(IMAGE_DIR, file_name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError('cannot read ' + path)
    return img.astype(np.float64)


# round, clip to 0~255 and convert to uint8
def to_uint8(img):
    img = np.round(img)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


# ./result/<prefix><file_name>
def result_path(prefix, file_name):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    return os.path.join(RESULT_DIR, prefix + file_name)


# show an image until a key is pressed
def show(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = img.shape[:2]
    # shrink big images so they fit on the screen
    scale = min(1.0, 900 / max(h, w))
    cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# call func and also return how long it took
def timed(func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    end = time.perf_counter()
    return out, end - start
